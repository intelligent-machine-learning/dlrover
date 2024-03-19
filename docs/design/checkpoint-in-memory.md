# Flash Checkpoint Loads Checkpoint in Memory for Fault Tolerance

The design describes how to load the checkpoint in the memory
of nodes to support fast fault tolerance. After a failure happens, DLRover
can restart training processes and resume the training from the checkpoint
in the memory of nodes not from the storage.

## Backgroup

Now, the training job loads the checkpoint from the storage after the training
process restarts. The IO speed of storage is a bottleneck to speed up the
model initialization. However, the memory of nodes is usually not fully utilized
and the node usually has a large amount of memory. For example, a machine with
8 A100 GPUs can has up to 1TB memory. The training can utilize the memory of node
to store and load the checkpoint after the training process restarts.

However, the checkpoint in the memory will be lost if the node fails. The training
need to backup copy the checkpoint of a node to other nodes which can reduce
the probability to loss the checkpoint if some nodes fail. So, we need to implement
the following features.

- The training can backup copy the checkpoint shard of a node to other nodes.
- The node can restore the checkpoint from other nodes if a node restarts.
- The training can load checkpoint from the memory of nodes.

## Design

Now, Flash Checkpoint can synchronously copy the model and optimizer states from
device memory to the CPU memory. With different distributed training stratgy,
the layout of the model and optimizer shards is diferent. We need to implement different
backup strategies for different distributed training stratgy.

### Backup and Restore Checkpoint in Memory

#### DDP

In a DDP job, the each rank has a complete model and optimizer replica. Using Flash Checkpoint,
the local rank 0 of each node will copy the checkpoint into the CPU memory. The each node
has a complete checkpoint in the CPU memory. If a node breakdowns and restarts, the job can
select a alive node to broadcat its checkpoint to the new node.

#### FSDP

The each rank has an unique shard of model and optimizer states using FSDP. The ElasticAgent of each node
has its checkpoint shard in the CPU memoy with Flash Checkpoint. The job splits the nodes into groups and
each group has 2 nodes. Each node need to backup the their checkpoint shard to the other node in a group.
We can build a process group with the ElasticAgent using gloo backend to transfer checkpoint shards.
Then, the ElasticAgent uses `torch.distributed.all_gather_object`
to backup checkpoint shards. If a node in a group breakdowns and restarts, the job need to select the other
alive node to broadcast the backup checkpoint to the new node.
If the nodes in a group all fails, the training can only resume from the checkpoint in the storage.

#### Megatron-LM

The megatron-LM uses the 3D parallel to train a LLM. The model and optimizer shards of the node with
the same PP rank in different DP ranks are same. Similar to DDP, the new node can restore
the checkpoint from the alive node with the same PP rank. If the training uses the distributed optimizer
which partition the optimizer states across all rank, we need to backup the checkpoint similar to FSDP.

### Group Nodes to Backup Checkpoint

The job can pair the nodes in groups with two nodes according to their sequence numbers. For example,
the groups are [{0, 1}, {2, 3}, {4, 5}, {6, 7}] if there are 8 nodes. After the training restarts,
each node will report "yes" to the master if the checkpoint in its memory, otherwise report "No" if
the node restarts. The following 3 cases may happen:

- No any node restarts. The job master notifies all nodes to restore the checkpoint from its CPU memory.
- One node in some group has restarted The job master notifies the other alive node to broardcast the
checkpoint in the CPU memory to the restarted node. Then, all nodes rrestore the checkpoint from its CPU memory.
- All the nodes in a group have restarted, which shows some checkpoint shards are lost in the memory.
If no any node restarts,. The job master will notify all nodes to restore checkpoint from the storage.
