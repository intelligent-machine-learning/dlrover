# Switch from DDP to FSDP with NanoGPT

Welcome to this guide on how to transition from DDP (Distributed Data Parallel) to
FSDP (Fully Sharded Data Parallelism) for training the NanoGPT model. This guide assumes
familiarity with the previous DDP guide. If you're new to DDP, we recommend checking out the DDP guide first.

## What is FSDP?

FSDP is an alternative approach to DDP, designed to improve the efficiency of distributed training.
It achieves this by effectively partitioning data and model parameters, reducing communication
overhead, and enabling more efficient training on large-scale models.

## Configure FSDP for NanoGPT

To replace DDP with FSDP in your existing NanoGPT training configuration,
simply make the following changes. Use the `kubectl` command to apply the modified training configuration:

```bash
kubectl -n dlrover apply -f examples/pytorch/nanogpt/fsdp_elastic_job.yaml
```

Upon successful application of the job configuration,
you can monitor the status of the training nodes using the command below:

```bash
kubectl -n dlrover get pods
```

## Comparing DDP and FSDP Results

Let's compare the results obtained using DDP and FSDP with the same parameter settings.
Here are the results for the two approaches:

**DDP:**

```bash
# parameter settings in examples/pytorch/nanogpt/fsdp_elastic_job.yaml
--n_layer 6 \
--n_head 6 \
--n_embd 384
```

**FSDP (Same Parameter Setting as DDP):**

```bash
# parameter settings in examples/pytorch/nanogpt/fsdp_elastic_job.yaml
--n_layer 6 \
--n_head 6 \
--n_embd 384
```

### More detailed description of the pods

Worker-0 Logs

```bash
kubectl logs -n dlrover torch-nanogpt-edljob-worker-0
```

results on DDP:

```text
iter 0: loss 4.2519, time 1295.23ms, mfu -100.00%, cuda memory 0.499G, lr 6.00e-04, total time 1.30s
iter 1: loss 3.5362, time 26.58ms, mfu -100.00%, cuda memory 0.499G, lr 6.00e-04, total time 1.32s
iter 2: loss 4.0429, time 26.42ms, mfu -100.00%, cuda memory 0.499G, lr 6.00e-04, total time 1.35s
iter 3: loss 3.5291, time 26.23ms, mfu -100.00%, cuda memory 0.499G, lr 6.00e-04, total time 1.37s
iter 4: loss 3.3225, time 26.87ms, mfu -100.00%, cuda memory 0.499G, lr 6.00e-04, total time 1.40s
iter 5: loss 5.9597, time 26.80ms, mfu 3.30%, cuda memory 0.499G, lr 6.00e-04, total time 1.43s
iter 6: loss 5.7204, time 27.03ms, mfu 3.30%, cuda memory 0.499G, lr 6.00e-04, total time 1.46s
iter 7: loss 3.3745, time 26.98ms, mfu 3.30%, cuda memory 0.499G, lr 6.00e-04, total time 1.48s
iter 8: loss 3.4374, time 27.36ms, mfu 3.29%, cuda memory 0.499G, lr 6.00e-04, total time 1.51s
iter 9: loss 3.2982, time 27.45ms, mfu 3.29%, cuda memory 0.499G, lr 6.00e-04, total time 1.54s
iter 10: loss 3.2967, time 28.30ms, mfu 3.27%, cuda memory 0.499G, lr 6.00e-04, total time 1.57s
```

results on FSDP:

```text
iter 0: loss 4.2674, time 1967.15ms, mfu -100.00%, cuda memory 0.479G, lr 6.00e-04, total time 1.97s
iter 1: loss 3.4770, time 26.56ms, mfu -100.00%, cuda memory 0.479G, lr 6.00e-04, total time 1.99s
iter 2: loss 4.6944, time 27.10ms, mfu -100.00%, cuda memory 0.479G, lr 6.00e-04, total time 2.02s
iter 3: loss 3.7846, time 28.37ms, mfu -100.00%, cuda memory 0.479G, lr 6.00e-04, total time 2.05s
iter 4: loss 3.4877, time 27.44ms, mfu -100.00%, cuda memory 0.479G, lr 6.00e-04, total time 2.08s
iter 5: loss 3.2793, time 27.75ms, mfu 1.67%, cuda memory 0.479G, lr 6.00e-04, total time 2.10s
iter 6: loss 3.3074, time 29.90ms, mfu 1.66%, cuda memory 0.479G, lr 6.00e-04, total time 2.13s
iter 7: loss 3.3063, time 30.07ms, mfu 1.65%, cuda memory 0.479G, lr 6.00e-04, total time 2.16s
iter 8: loss 3.2537, time 30.04ms, mfu 1.64%, cuda memory 0.479G, lr 6.00e-04, total time 2.19s
iter 9: loss 3.3800, time 29.74ms, mfu 1.63%, cuda memory 0.479G, lr 6.00e-04, total time 2.22s
iter 10: loss 3.2457, time 30.30ms, mfu 1.62%, cuda memory 0.479G, lr 6.00e-04, total time 2.18s
```

Worker-1 Logs

```bash
kubectl logs -n dlrover torch-nanogpt-edljob-worker-1
```

results on DDP:

```text
iter 0: loss 4.2464, time 1295.62ms, mfu -100.00%, cuda memory 0.499G, lr 6.00e-04, total time 1.30s
iter 1: loss 3.4549, time 26.48ms, mfu -100.00%, cuda memory 0.499G, lr 6.00e-04, total time 1.32s
iter 2: loss 4.0122, time 26.27ms, mfu -100.00%, cuda memory 0.499G, lr 6.00e-04, total time 1.35s
iter 3: loss 3.5630, time 26.30ms, mfu -100.00%, cuda memory 0.499G, lr 6.00e-04, total time 1.37s
iter 4: loss 3.2510, time 26.84ms, mfu -100.00%, cuda memory 0.499G, lr 6.00e-04, total time 1.40s
iter 5: loss 6.0906, time 26.68ms, mfu 3.32%, cuda memory 0.499G, lr 6.00e-04, total time 1.43s
iter 6: loss 5.7520, time 27.01ms, mfu 3.31%, cuda memory 0.499G, lr 6.00e-04, total time 1.46s
iter 7: loss 3.3311, time 26.91ms, mfu 3.31%, cuda memory 0.499G, lr 6.00e-04, total time 1.48s
iter 8: loss 3.3454, time 27.30ms, mfu 3.30%, cuda memory 0.499G, lr 6.00e-04, total time 1.51s
iter 9: loss 3.3826, time 27.42ms, mfu 3.30%, cuda memory 0.499G, lr 6.00e-04, total time 1.54s
iter 10: loss 3.3080, time 28.20ms, mfu 3.28%, cuda memory 0.499G, lr 6.00e-04, total time 1.57s
```

results on FSDP:

```text
iter 0: loss 4.2821, time 1893.33ms, mfu -100.00%, cuda memory 0.479G, lr 6.00e-04, total time 1.89s
iter 1: loss 3.5487, time 26.76ms, mfu -100.00%, cuda memory 0.479G, lr 6.00e-04, total time 1.92s
iter 2: loss 4.7303, time 26.95ms, mfu -100.00%, cuda memory 0.479G, lr 6.00e-04, total time 1.95s
iter 3: loss 3.7793, time 28.20ms, mfu -100.00%, cuda memory 0.479G, lr 6.00e-04, total time 1.98s
iter 4: loss 3.5544, time 27.60ms, mfu -100.00%, cuda memory 0.479G, lr 6.00e-04, total time 2.00s
iter 5: loss 3.4145, time 27.70ms, mfu 1.67%, cuda memory 0.479G, lr 6.00e-04, total time 2.03s
iter 6: loss 3.3414, time 29.55ms, mfu 1.66%, cuda memory 0.479G, lr 6.00e-04, total time 2.06s
iter 7: loss 3.3453, time 30.01ms, mfu 1.65%, cuda memory 0.479G, lr 6.00e-04, total time 2.09s
iter 8: loss 3.3394, time 29.89ms, mfu 1.64%, cuda memory 0.479G, lr 6.00e-04, total time 2.12s
iter 9: loss 3.2509, time 29.63ms, mfu 1.63%, cuda memory 0.479G, lr 6.00e-04, total time 2.15s
iter 10: loss 3.2535, time 30.32ms, mfu 1.62%, cuda memory 0.479G, lr 6.00e-04, total time 2.25s
```

## References

This guide is a supplemental resource to [torch_ddp_nanogpt.md](./torch_ddp_nanogpt.md).
For more details about the usage environment, please refer to torch_ddp_nanogpt.md.
