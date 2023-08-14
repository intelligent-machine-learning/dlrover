# Switching from DDP to FSDP with NanoGPT

Welcome to this guide on how to transition from using DDP (DistributedDataParallel) to FSDP (Fully Sharded Data Parallelism) for training the NanoGPT model. This guide assumes familiarity with the previous DDP guide. If you're new to DDP, we recommend checking out the DDP guide first.



## What is FSDP?

FSDP is an alternative approach to DDP, designed to improve the efficiency of distributed training. It achieves this by effectively partitioning data and model parameters, reducing communication overhead, and enabling more efficient training on large-scale models.



## Configuring FSDP for NanoGPT

To replace DDP with FSDP in your existing NanoGPT training configuration, simply make the following changes. Use the `kubectl` command to apply the modified training configuration:

```bash
$ kubectl -n dlrover apply -f dlrover/examples/torch_nanogpt_job_with_fsdp.yaml
```

Upon successful application of the job configuration, you can monitor the status of the training nodes using the command below:

```bash
$ kubectl -n dlrover get pods
```



## Comparing DDP and FSDP Results

Let's compare the results obtained using DDP and FSDP with the same parameter settings. Here are the results for the two approaches:

**DDP:**

```bash
# parameter settings in dlrover/examples/torch_nanogpt_job.yaml
--n_layer 6 \
--n_head 6 \
--n_embd 384
```

**FSDP (Same Parameter Setting as DDP):**

```bash
# parameter settings in dlrover/examples/torch_nanogpt_job_with_fsdp.yaml
--n_layer 6 \
--n_head 6 \
--n_embd 384
```



### More detailed description of the pods:

Worker-0 Logs

```bash

$ kubectl logs -n dlrover torch-nanogpt-edljob-worker-0

```

results on DDP:

```
iter 0: loss 4.2516, time 1259.10ms, mfu -100.00%, lr 6.00e-04, total time 1.26s
iter 1: loss 3.5361, time 26.30ms, mfu -100.00%, lr 6.00e-04, total time 1.29s
iter 2: loss 4.0251, time 27.39ms, mfu -100.00%, lr 6.00e-04, total time 1.31s
iter 3: loss 3.5098, time 24.61ms, mfu -100.00%, lr 6.00e-04, total time 1.34s
iter 4: loss 3.3147, time 25.24ms, mfu -100.00%, lr 6.00e-04, total time 1.36s
iter 5: loss 5.8905, time 25.39ms, mfu 3.49%, lr 6.00e-04, total time 1.39s
iter 6: loss 3.2859, time 26.04ms, mfu 3.48%, lr 6.00e-04, total time 1.41s
iter 7: loss 3.5160, time 27.36ms, mfu 3.45%, lr 6.00e-04, total time 1.44s
iter 8: loss 3.2804, time 26.90ms, mfu 3.44%, lr 6.00e-04, total time 1.47s
iter 9: loss 3.2039, time 26.75ms, mfu 3.42%, lr 6.00e-04, total time 1.50s
iter 10: loss 3.1332, time 27.30ms, mfu 3.41%, lr 6.00e-04, total time 1.52s
```



results on FSDP:

```
iter 0: loss 4.2827, time 2025.59ms, mfu -100.00%, lr 6.00e-04, total time 2.03s
iter 1: loss 3.5478, time 26.23ms, mfu -100.00%, lr 6.00e-04, total time 2.05s
iter 2: loss 4.7255, time 26.75ms, mfu -100.00%, lr 6.00e-04, total time 2.08s
iter 3: loss 3.7794, time 25.40ms, mfu -100.00%, lr 6.00e-04, total time 2.10s
iter 4: loss 3.5554, time 26.42ms, mfu -100.00%, lr 6.00e-04, total time 2.13s
iter 5: loss 3.4140, time 26.93ms, mfu 1.72%, lr 6.00e-04, total time 2.16s
iter 6: loss 3.3416, time 27.06ms, mfu 1.72%, lr 6.00e-04, total time 2.18s
iter 7: loss 3.3455, time 27.24ms, mfu 1.72%, lr 6.00e-04, total time 2.21s
iter 8: loss 3.3400, time 28.50ms, mfu 1.71%, lr 6.00e-04, total time 2.24s
iter 9: loss 3.2522, time 29.86ms, mfu 1.69%, lr 6.00e-04, total time 2.27s
iter 10: loss 3.2481, time 30.10ms, mfu 1.68%, lr 6.00e-04, total time 2.30s
```



Worker-1 Logs

```bash

$ kubectl logs -n dlrover torch-nanogpt-edljob-worker-1

```

results on DDP:

```
iter 0: loss 4.2464, time 1259.01ms, mfu -100.00%, lr 6.00e-04, total time 1.26s
iter 1: loss 3.4552, time 26.49ms, mfu -100.00%, lr 6.00e-04, total time 1.29s
iter 2: loss 3.9973, time 27.44ms, mfu -100.00%, lr 6.00e-04, total time 1.31s
iter 3: loss 3.5437, time 24.62ms, mfu -100.00%, lr 6.00e-04, total time 1.34s
iter 4: loss 3.2443, time 25.21ms, mfu -100.00%, lr 6.00e-04, total time 1.36s
iter 5: loss 6.0296, time 25.49ms, mfu 3.47%, lr 6.00e-04, total time 1.39s
iter 6: loss 3.2579, time 26.01ms, mfu 3.47%, lr 6.00e-04, total time 1.41s
iter 7: loss 3.4510, time 27.37ms, mfu 3.44%, lr 6.00e-04, total time 1.44s
iter 8: loss 3.1951, time 26.71ms, mfu 3.43%, lr 6.00e-04, total time 1.47s
iter 9: loss 3.2957, time 26.76ms, mfu 3.42%, lr 6.00e-04, total time 1.50s
iter 10: loss 3.1399, time 27.30ms, mfu 3.40%, lr 6.00e-04, total time 1.52s
```

results on FSDP:

```
iter 0: loss 4.2673, time 1976.03ms, mfu -100.00%, lr 6.00e-04, total time 1.98s
iter 1: loss 3.4755, time 26.36ms, mfu -100.00%, lr 6.00e-04, total time 2.00s
iter 2: loss 4.6895, time 26.66ms, mfu -100.00%, lr 6.00e-04, total time 2.03s
iter 3: loss 3.7849, time 25.39ms, mfu -100.00%, lr 6.00e-04, total time 2.05s
iter 4: loss 3.4884, time 26.46ms, mfu -100.00%, lr 6.00e-04, total time 2.08s
iter 5: loss 3.2789, time 26.89ms, mfu 1.73%, lr 6.00e-04, total time 2.11s
iter 6: loss 3.3076, time 27.14ms, mfu 1.72%, lr 6.00e-04, total time 2.13s
iter 7: loss 3.3066, time 27.32ms, mfu 1.72%, lr 6.00e-04, total time 2.16s
iter 8: loss 3.2544, time 28.56ms, mfu 1.71%, lr 6.00e-04, total time 2.19s
iter 9: loss 3.3811, time 29.67ms, mfu 1.70%, lr 6.00e-04, total time 2.22s
iter 10: loss 3.2566, time 30.13ms, mfu 1.68%, lr 6.00e-04, total time 2.25s
```




# References

This guide is a supplemental resource to [torch_ddp_nanogpt.md](./torch_ddp_nanogpt.md). For more details about the usage environment, please refer to torch_ddp_nanogpt.md.