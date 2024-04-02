# How to Implement Failure Recovery to Support New Processors in DLRover

If you are using a new process to training your AI models, you can follow
the tutorial to implment some modules to support failure recovery with the new processor.

## Node Detection

Now, DLRover can automatically detect the node by running a GEMM and allgather task on
NVIDIA GPU and Ascend NPU. We can see the [node detection design](../blogs/stabilize_llm_training_cn.md)
for the implementation detail and the [scripts](../../dlrover/trainer/torch/node_check/) of tasks.
If your processor is compatible with CUDA, you can firstly try `dlrover-run --network-check`
with the default NVIDIA GPU detection script.

If you want to implement node detection on your new processor, you can follow the steps:

1. Implement the script to run a PyTorch task in `dlrover/trainer/torch/node_check/`.

2. Set the option in the method `run_network_check` of `dlrover/python/elastic_agent/torch/training.py` like

```Python
def run_network_check(config: ElasticLaunchConfig, entrypoint):
    if config.accelerator == Accelerators.NVIDIA_GPU:
        cmd_args = ["-m", "dlrover.trainer.torch.node_check.nvidia_gpu"]
    elif config.accelerator == Accelerators.ASCEND_NPU:
        cmd_args = ["-m", "dlrover.trainer.torch.node_check.ascend_npu"]
    elif config.accelerator == xxxx:
        cmd_args = ["-m", "dlrover.trainer.torch.node_check.xxxx_xpu"]
```

3. Set the processor type into the arguments `--accelerator` of `dlrover-run`.

```Python
parser.add_argument(
        "--accelerator",
        type=str,
        action=env,
        default=Accelerators.NVIDIA_GPU,
        choices=[Accelerators.NVIDIA_GPU, Accelerators.ASCEND_NPU, xxxx],
        help="The type of accelerator chip of the machine.",
    )
```

Finally, you can use `dlrover-run --network-check --accelerator=xxxx ...` to run your training jobs.

## Mark Fault Node as Unschedulable

The elastic job master of DLRover will mark the fault node as unschedulable if the node
fail to run the detection task or the pod on the node fail with exit code [14, 128, 201, 202].

- 14: infoROM is corrupted at gpu.
- 128: unknown device.
- 201: GPU driver error.
- 202: Residual processes on GPU.

Now, DLRover use k8s Python APIs same as `kubectl cordon` to mark the node unschedulable in `SimpleErrorMonitor` in
`dlrover/python/master/monitor/error_monitor.py`.

```Python
def _handle_node_error(self, node: Node, error_data: str):
    logger.info(
        f"{node.name} on {node.host_name} is down. "
        f"Reason: {error_data}"
    )
    succeed = self._k8s_client.cordon_node(node.host_name)
    if succeed:
        logger.info(f"Node {node.name} is marked unschedulable.")
    return True
```

You can implement a new `ErrorMonitor` to mark the node unschedulable in your cluster.

## Flash Checkpoint

DLRover has implemented [Flash Checkpoint](../blogs/flash_checkpoint.md) to shorten the time to
save checkpoint during training. Now, Flash Checkpoint supports DDP/FSDP/DeepSpeed/Megatron-LM.
You can use Flash Checkpoint if you can use `torch.save` and `torch.load` to save and load checkpoint.
