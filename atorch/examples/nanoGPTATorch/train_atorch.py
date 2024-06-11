"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import math
import os
import pickle
import time
from contextlib import nullcontext
from functools import partial

import numpy as np
import torch
from model import GPT, GPTConfig
from torch.utils.tensorboard import SummaryWriter

import atorch

amp = torch.cuda.amp

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
eval_interval = 2000
log_interval = 1
eval_iters = 2
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
init_from = "scratch"  # 'scratch' or 'resume' or 'gpt2*'
# data
dataset = "openwebtext/"
# gradient_accumulation_steps = 5 # used to simulate larger batch sizes
gradient_accumulation_steps = 1
batch_size = 12 * 5  # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias = False  # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4  # max learning rate
max_iters = 200000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# grad_clip = 0.0 # clip gradients at this value, or disable if == 0.0
# grad_clip = 0.1 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 2000  # how many steps to warm up for
lr_decay_iters = 200000  # should be ~= max_iters per Chinchilla
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = "nccl"  # 'nccl', 'gloo', etc.
# system
device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
# dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
# 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
dtype = "float16"
train_type = "fsdp"
tb_path = "/tmp"
compile = False  # use PyTorch 2.0 to compile the model to be faster
win = False  # the parameter for agd optimizer
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str))]
exec(open("configurator.py").read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging

# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup


def setup_seed(seed_offset):
    torch.manual_seed(1337 + seed_offset)
    np.random.seed(1337 + seed_offset)


device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = nullcontext()

# poor man's data loader
data_dir = dataset
train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix])
    if device_type == "cuda":
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y, ix


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)


def build_model():

    # attempt to derive vocab_size from the dataset
    meta_path = os.path.join(data_dir, "meta.pkl")
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        meta_vocab_size = meta["vocab_size"]
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

    # model init
    model_args = dict(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        bias=bias,
        vocab_size=None,
        dropout=dropout,
    )  # start with model_args from command line
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    return model


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y, ix = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def train():
    if len(tb_path) == 0:
        raise ValueError("you shoud pass `tb_path` like --tb_path=path/to/your/tb")
    iter_num = 0
    # restore_iter = 50000
    load_ckpt = False
    rank = atorch.rank()
    tb_metrics_prefix = f"{train_type}-{dtype}"
    if rank == 0:
        writer = SummaryWriter(f"{tb_path}/{tb_metrics_prefix}")

    model = build_model().to("cuda")
    if load_ckpt:
        pass
    else:
        setup_seed(rank)
    from atorch.auto import auto_accelerate  # noqa: E402

    fsdp_config = {
        "sync_module_states": True,
        "use_orig_params": True,
    }
    from model import Block  # , LayerNorm

    fsdp_config["atorch_wrap_cls"] = {
        Block,
    }
    p_mode = ([("data", atorch.world_size())], None)
    if train_type == "fsdp":
        strategy = [
            ("parallel_mode", p_mode),
            ("module_replace"),
            ("fsdp", fsdp_config),
            ("amp_native", {"dtype": ptdtype}),
        ]
    else:
        strategy = [
            ("parallel_mode", p_mode),
            ("module_replace"),
            ("amp_native", {"dtype": ptdtype}),
        ]
    N = model.get_num_params()
    print("Manually loaded auto acc strategy:", strategy)

    def optim_func(params, model):
        return model.configure_optimizers(
            weight_decay, learning_rate, (beta1, beta2), device_type, win, opt_name="adamw"
        )

    def my_loss_func(_, outputs):
        return outputs[1]

    def configure_optimizers_group(model, weight_decay):
        # start with all of the candidate parameters
        # print(model)
        param_dict = {pn: p for pn, p in model.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.

        def is_decay(name):
            if "wpe" in name or "bias" in name or "ln_" in name:
                return False
            return True

        decay_params = [p for n, p in param_dict.items() if is_decay(n)]
        nodecay_params = [p for n, p in param_dict.items() if not is_decay(n)]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        # print(optim_groups)
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        return optim_groups

    status, result, best_strategy = auto_accelerate(
        model,
        optim_func=torch.optim.AdamW,
        loss_func=my_loss_func,
        optim_args={
            "lr": learning_rate,
            "betas": (beta1, beta2),
        },  # "use_kahan_summation": True, "momentum_dtype": torch.float32},
        optim_param_func=partial(configure_optimizers_group, weight_decay=weight_decay),
        load_strategy=strategy,
        verbose=True,
        ignore_dryrun_on_load_strategy=True,
    )
    assert status, "auto_accelerate failed"
    X, Y, ix = get_batch("train")  # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0  # number of iterations in the lifetime of this process
    raw_model = model
    running_mfu = -1.0
    # best_val_loss = 1e9
    master_process = rank == 0
    model = result.model
    optimizer = result.optim
    loss_func = result.loss_func

    def clip_norm(model, optim, grad_clip):
        if grad_clip == 0.0:
            return 0.0
        if dtype == "float16":
            optim.unscale_()
        if train_type == "fsdp":
            return model.clip_grad_norm_(grad_clip)
        return torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    while True:
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0:
            losses = estimate_loss(model)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            print_dict = {
                "iter": iter_num,
                "train/loss": losses["train"],
                "val/loss": losses["val"],
                "lr": lr,
                "mfu": running_mfu * 100,  # convert to percentage
            }
            print_s = [f"{k}:{str(v)}" for k, v in print_dict.items()]
            print(" ".join(print_s))
        if iter_num == 0 and eval_only:
            break

        optimizer.zero_grad(True)
        outputs = model(X, Y)
        loss = loss_func(None, outputs)

        # loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y, ix = get_batch("train")
        loss.backward()
        norm = clip_norm(model, optimizer, grad_clip)
        optimizer.step()

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * gradient_accumulation_steps
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt, N=N)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            if rank == 0:
                print(
                    f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%, norm {norm}"
                )
                writer.add_scalar("Loss", loss, iter_num)
                writer.add_scalar("Grad norm", norm, iter_num)
                writer.add_scalar("MFU", running_mfu, iter_num)
            if iter_num % 50 == 0:
                writer.flush()

        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > max_iters:
            break


if __name__ == "__main__":
    atorch.init_distributed("nccl", set_cuda_device_using_local_rank=True)
    train()
    atorch.reset_distributed()
