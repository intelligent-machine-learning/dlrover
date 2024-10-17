import os
import subprocess
from functools import partial
from typing import Union
from urllib.request import urlretrieve

import pytest

import atorch

torch = pytest.importorskip("torch", minversion="2.0.9")
if torch.version.git_version != "7bcf7da3a268b435777fe87c7794c382f444e86d" or not torch.cuda.is_available():
    pytest.skip("requires pytorch 2.1 stable release", allow_module_level=True)

from atorch.common.log_utils import default_logger as logger  # noqa: E402
from atorch.common.util_func import find_free_port  # noqa: E402
from atorch.trainer import AtorchTrainerV2, AtorchTrainingArgs  # noqa: E402
from atorch.trainer.megatron import MegatronTrainStep  # noqa: E402
from atorch.utils.import_util import is_coverage_available, is_megatron_lm_available  # noqa: E402
from atorch.utils.version import torch_version  # noqa: E402

if is_megatron_lm_available():
    import megatron.legacy.model
    from megatron.core import mpu
    from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
    from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_local_spec,
        get_gpt_layer_with_transformer_engine_spec,
    )
    from megatron.core.transformer.spec_utils import import_module
    from megatron.training import get_args, get_tokenizer, print_rank_0
    from megatron.training.arguments import core_transformer_config_from_args
    from megatron.training.utils import (
        average_losses_across_data_parallel_group,
        get_batch_on_this_cp_rank,
        get_batch_on_this_tp_rank,
    )
    from megatron.training.yaml_arguments import core_transformer_config_from_yaml

if is_coverage_available():
    import coverage


# torch = pytest.importorskip("torch", minversion="2.0.9")
# if torch.version.git_version != "7bcf7da3a268b435777fe87c7794c382f444e86d" or not torch.cuda.is_available():
#     pytest.skip("requires pytorch 2.1 stable release", allow_module_level=True)


def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.

    If you set the use_mcore_models to True, it will return the mcore GPT model and if not the legacy GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"

    print_rank_0("building GPT model ...")
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    if args.use_mcore_models:
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if use_te:
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                    args.num_experts, args.moe_grouped_gemm
                )
            else:
                transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm)

        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
        )
    else:
        assert args.context_parallel_size == 1, "Context parallelism is only supported with Megatron Core!"

        model = megatron.legacy.model.GPTModel(
            config, num_tokentypes=0, parallel_output=True, pre_process=pre_process, post_process=post_process
        )

    return model


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """

    def is_dataset_built_on_rank():
        return (
            mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()
        ) and mpu.get_tensor_model_parallel_rank() == 0

    def core_gpt_dataset_config_from_args(args):
        tokenizer = get_tokenizer()

        return GPTDatasetConfig(
            random_seed=args.seed,
            sequence_length=args.seq_length,
            blend=args.data_path,
            blend_per_split=[args.train_data_path, args.valid_data_path, args.test_data_path],
            split=args.split,
            path_to_cache=args.data_cache_path,
            mock=args.mock_data,
            mmap_bin_files=args.mmap_bin_files,
            tokenizer=tokenizer,
            reset_position_ids=args.reset_position_ids,
            reset_attention_mask=args.reset_attention_mask,
            eod_mask_loss=args.eod_mask_loss,
            create_attention_mask=args.create_attention_mask_in_dataloader,
        )

    args = get_args()
    config = core_gpt_dataset_config_from_args(args)

    if config.mock:
        dataset_type = MockGPTDataset
    else:
        dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type, train_val_test_num_samples, is_dataset_built_on_rank, config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


class GPTTrainStep(MegatronTrainStep):
    """
    GPT train step

    Args:
        args (`argparse.Namespace`): Megatron-LM arguments.
    """

    def __init__(self, args, **kwargs):
        super().__init__()
        if not args.model_return_dict:
            self.model_output_class = None
        else:
            from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

            self.model_output_class = CausalLMOutputWithCrossAttentions

    def get_batch_func(self, **kwargs):
        def get_batch(data_iterator):
            """Generate a batch."""

            # TODO: this is pretty hacky, find a better way
            if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
                return None, None, None, None, None

            # get batches based on the TP rank you are on
            batch = get_batch_on_this_tp_rank(data_iterator)

            # slice batch along sequence dimension for context parallelism
            batch = get_batch_on_this_cp_rank(batch)

            return batch.values()

        return get_batch

    def get_loss_func(self, **kwargs):
        def loss_func(loss_mask, output_tensor):
            """Loss function.

            Args:
                loss_mask (torch.Tensor): Used to mask out some portions of the loss
                output_tensor (torch.Tensor): The tensor with the losses
            """
            args = get_args()

            losses = output_tensor.float()
            loss_mask = loss_mask.view(-1).float()
            if args.context_parallel_size > 1:
                loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), loss_mask.sum().view(1)])
                torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())
                loss = loss[0] / loss[1]
            else:
                loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

            # Check individual rank losses are not NaN prior to DP all-reduce.
            if args.check_for_nan_in_loss_and_grad:
                global_rank = torch.distributed.get_rank()
                assert not loss.isnan(), (
                    f"Rank {global_rank}: found NaN in local forward loss calculation. "
                    f"Device: {torch.cuda.current_device()}, node: {os.uname()[1]}"
                )

            # Reduce loss for logging.
            averaged_loss = average_losses_across_data_parallel_group([loss])

            return loss * args.context_parallel_size, {"lm loss": averaged_loss[0]}

        return loss_func

    def get_forward_step_func(self, **kwargs):
        def forward_step(data_iterator, model: GPTModel):
            """Forward training step.

            Args:
                data_iterator : Input data iterator
                model (GPTModel): The GPT Model
            """
            # Get the batch.
            tokens, labels, loss_mask, attention_mask, position_ids = self.get_batch_func()(data_iterator)
            output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

            return output_tensor, partial(self.get_loss_func(), loss_mask)

        return forward_step


vocab_url = "https://alps-common.oss-cn-hangzhou-zmf.aliyuncs.com/users/jinshi/atorch_unittest_data/vocab.json"
merge_url = "https://alps-common.oss-cn-hangzhou-zmf.aliyuncs.com/users/jinshi/atorch_unittest_data/merges.txt"
vocab_file = "/tmp/gpt2_vocab.json"
merge_file = "/tmp/gpt2_merges.json"


def download_tokenizer_file():
    try:
        if not os.path.exists(vocab_file):
            logger.info(f"Downloading {vocab_url} to {vocab_file}")
            urlretrieve(vocab_url, vocab_file)
        if not os.path.exists(merge_file):
            logger.info(f"Downloading {merge_url} to {merge_file}")
            urlretrieve(merge_url, merge_file)
    except Exception as e:
        logger.exception(f"Download {vocab_url} and {merge_url} failed, please check if the addresses exist. {e}")
        return False
    return True


def run_atorch_trainer_v2():
    output_dir = "/tmp/output_atorch_trainer"
    training_args = AtorchTrainingArgs(
        distributed_type="megatron",
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        do_train=True,
        bf16=True,
        save_strategy="steps",
        save_steps=20,
        save_total_limit=1,
        evaluation_strategy="steps",
        eval_steps=25,
        logging_strategy="steps",
        logging_steps=1,
        logging_nan_inf_filter=False,
        gradient_checkpointing=False,
        tensorboard_dir=os.path.join(output_dir, "runs"),
    )

    train_valid_test_datasets_provider.is_distributed = True

    megatron_args = dict(
        # Custom function
        custom_model_provider_function=model_provider,
        custom_megatron_datasets_provider_function=train_valid_test_datasets_provider,
        custom_train_step_class=GPTTrainStep,
        # model args
        model_type_name="gpt",
        num_layers=12,
        hidden_size=768,
        num_attention_heads=12,
        group_query_attention=True,
        num_query_groups=12,
        max_position_embeddings=512,
        position_embedding_type="rope",
        make_vocab_size_divisible_by=1,
        norm_epsilon=1e-5,
        normalization="RMSNorm",
        untie_embeddings_and_output_weights=True,
        use_flash_attn=True,
        # tokenizer
        tokenizer_type="GPT2BPETokenizer",
        vocab_file=vocab_file,
        merge_file=merge_file,
        # optimizer
        optimizer="adam",
        # Regular args
        attention_dropout=0.0,
        hidden_dropout=0.0,
        weight_decay=1e-1,
        clip_grad=1.0,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        # Megatron training args
        pretraining_flag=True,
        use_mcore_models=True,
        transformer_impl="transformer_engine",
        micro_batch_size=1,
        global_batch_size=2,
        add_bias_linear=False,
        bias_gelu_fusion=False,
        recompute_activations=True,
        recompute_granularity="selective",
        train_iters=25,
        eval_iters=5,
        # Distributed args
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        sequence_parallel=True,
        distributed_backend="nccl",
        use_distributed_optimizer=True,
        # Logging args
        log_timers_to_tensorboard=True,
        log_validation_ppl_to_tensorboard=True,
        log_memory_to_tensorboard=True,
        log_throughput=True,
        log_params_norm=True,
        log_params_std=True,
        tensorboard_dir=training_args.tensorboard_dir,
        # Initialization args
        seed=1403,
        init_method_std=0.02,
        # Learning rate args
        lr=3e-5,
        min_lr=3e-6,
        lr_ecay_style="cosine",
        lr_warmup_fraction=0.1,
        # Data
        data_cache_path=os.path.join(training_args.output_dir, "data_cache"),
        mock_data=True,
        seq_length=512,
        num_workers=0,
        gradient_accumulation_fusion=False,
    )

    if megatron_args["sequence_parallel"]:
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

    training_args.extra_configs = megatron_args

    trainer = AtorchTrainerV2(
        args=training_args,
    )
    train_result = trainer.train()
    print(f"{train_result.metrics}")

    atorch.reset_distributed()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip cpu ut, only run on gpu.")
@pytest.mark.skipif(torch_version() < (2, 0, 0), reason="AtorchTrainer need torch2.0 .")  # type: ignore
@pytest.mark.parametrize("gpu_num", [4])
def test_atorch_trainer(gpu_num):

    if not download_tokenizer_file():
        logger.warning(f"Can't download {vocab_url} and {merge_url}, skip this unit test.")
        return

    # Test for AntMonitor
    if os.environ.get("ANTMONITOR_TFEVENT_PATH") is None:
        os.environ["ANTMONITOR_TFEVENT_PATH"] = "/home/admin/logs/tfevent"

    launch_engine = "coverage run" if is_coverage_available() else "python"
    dist_cmd = (
        f"{launch_engine} -m atorch.distributed.run --nnode=1 --nproc_per_node={gpu_num} "
        f"--node_rank=0 --master_port={find_free_port()} {__file__}"
    )

    subprocess.run(dist_cmd, check=True, shell=True)


if __name__ == "__main__":
    ut_cov = None
    if is_coverage_available():
        ut_cov = coverage.Coverage()
        ut_cov.start()

    run_atorch_trainer_v2()

    if ut_cov is not None:
        ut_cov.stop()
        ut_cov.save()
