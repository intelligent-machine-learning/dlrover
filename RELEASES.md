# Release Notes

The DLRover project follows the semantic versioning scheme and maintains a separate branch for each minor version. The main branch always represents the next upcoming minor or major version.

For laset news about DLRover you can check as following link: https://github.com/intelligent-machine-learning/dlrover?tab=readme-ov-file#latest-news=

##  Release 0.3.7 on May 13

Features:
* Flash Checkpoint suppors deleting old checkpoints.

BugFix:
* Save/load the non-params-related variables of dist optimizer in Megatron-LM models.
* The agent waits for async saving checkpoint finishes before exiting.

## Release 0.3.6 on Apr 24

Features:
* Flash checkpoint provides FlashCkptTrainer to support HuggingFace transforemers.Trainer.
* Flash checkpoint supports loading the checkpint of Megatron-LM from the memory.
Flash Checkpoint supports saving and loading FSDP checkpoint with full state dict.
* Job master can sort the node ranks by the access switches of the node.

BugFix:
* Fix the segment fault when restarting the training process.

## Release 0.3.5 on Mar 29

Features:
* Flash checkpoint supports saving and loading Megatron-LM MOE models. #1042
* APIs to extend the module to check the node with different chips. #1023
* Automatically mark the node as unschedulable if the node fails. #1025

BugFix:
* Fix the DDP example of mnist to save and load checkpoint. #1051
* Fix the checkpoint name of DDP. #1034

## Release 0.3.4 on Feb 21

Features:
* Flash checkpoint enables saving and loading Megatron-LM models from multiple ranks in parallel.
* dlrover-run --auto-config Automatically configure the number of nodes and the number of processes per node.
* Users can customize the APIs of storage to save the checkpoint into different file systems.
* Deletion strategy to clean the old checkpoint files.

BugFix:
* The shared memory does not exist if the size of the checkpoint changes.

## Release 0.3.3 on Jan 25

Features:
* Support Python > 3.10.
* Support restarting the training process on Ascend NPU.
* Support asynchronously saving the checkpoint of the distributed optimizer of Megatron-LM to the storage.

BugFix:
* Fix the checkpoint shard inconsistency of all ranks.
* Fix the bug to asynchronously save the Megatron-LM checkpoint of the job with multi-GPUs on multi-nodes.
* Fix the bug to load the Megatron-LM checkpoint.

## Release 0.3.1 on Jan 10

Feature:
* Users can use flash checkpoint using torchrun or python -m torch.distributed.launch.

Bugfix:
* The dlrover master cannot print the error message of the fault node in a kubeflow/PytorchJob.

## Release 0.3.0 on Jan 3

Features:
* Flash Checkpoint to asynchronously persist checkpoint to storage.
* Flash Checkpoint recovers failure in memory.
* Flash Checkpoint supports DDP/FSDP/DeepSpeed/Megatron
* Node detection supports NPU.

Examples
* The example of training nanoGPT using DeepSpeed.
* The example to save/load sharding FSDP checkpoint.


## Release 0.2.2 on Nov 21, 2023

Features:
* dlrover-run can run on any distributed jobs with the NODE_RANK and DLROVER_MASTER_ADDR in the environment.
* DLRover can asynchronously save the checkpoint into the storage which only block the training with a few time.

BugFix:
* Fix the bug to load the FSDP checkpoint.

## Release 0.2.1 on Oct 11, 2023

* Autotuning batch size without restarting the job.
* Automatically detect the straggler (slow worker).
* TFPlus: TFPlus 0.1.0 has been released, see detail in https://github.com/intelligent-machine-learning/dlrover/tree/master/tfplus
