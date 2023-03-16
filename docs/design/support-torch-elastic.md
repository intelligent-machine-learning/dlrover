# Design to Support Torch Distributed Elastic

The design is to introduce how to support elastic distributed
training using `torch.distributed.elastic` in DLRover.

## Motivation

With [Torch Distributed Elastic](https://pytorch.org/docs/stable/distributed.elastic.html)
, PyTorch can react to worker membership changes or failures and
restart worker processes. To run a job with elastic PyTorch on
a distributed cluster, an elastic node scheduler is required
to launch or remove nodes. Now, [TorchX-Kubernetes](https://pytorch.org/torchx/latest/)
supports to launch Pods on the Kubernetes with the configuration of min and max number
of workers. However, the batch size of sync-SGD is varying with
the number of workers which may result in inconsistency of model accuracy.
What's more, it is difficult for users to figure out the number of workers
that can maximize the training performance
including throughput and model accuracy.

Besides node scheduling, we need a data partition strategy to support
the change of workers and recover training data of fault workers.
So, the scheduler in DLRover will cooperate with Torch Distributed
Elastic to support the following features:

- Dynamic data sharding service to support elastcity and fault-tolerance.
- Support worker preemption.
- Fixed batch size during elastic training.
- Automatically configure the number of workers to  maximize
the training performance.

## Introduction to Torch Distributed Elastic

There are three main components in PyTorch to support elasticity:
`torch.distributed.elastic.agent.server.ElasticAgent`,
`torch.distributed.elastic.rendezvous.RendezvousHandler`
and `torch.distributed.elastic.rendezvous.dynamic_rendezvous.RendezvousBackend`.

On a machine, there is an `ElasticAgent` to manage one or more training
worker processes. The worker processes are assumed to be regular
distributed PyTorch scripts. If there are N GPUs on a machine,
the `ElasticAgent` will created N worker processes and
provides the necessary information for the worker processes to
properly initialize a torch process group.

`ElasticAgent` will periodically call `num_nodes_waiting` of
`RedezvousHandler` to synchoronize its rendezvous state
to the `RendezvousBackend` and ask the `RendezvousBackend`
whether there are new workers. The `RendezvousBackend`
stores all rendezvous states from all workers and returns
the number of waiting workers to each `ElasticAgent`.
Once there are waiting workers to join training,
the `ElasticAgent` calls `next_rendezvous`
to get the next rendevous and restarts its worker processes
to initialize a new torch process group.

## Design to Support Torch Distributed Elastic in DLRover

DLRover master can improve the efficiency of Torch Distributed
Elastic by the following features:

- DLRover provides the dynamic data sharding to support the change
of workers without repartitioning the whole dataset.
- DLRover master can implements a `RendezvousBackend` with awareness of
the status of all nodes in a job.
- DLRover can scale up/down nodes according to the cluster condition
and adjust the number of worker process to fix the batch
size of asynchronized SGD.

### Dynamic data sharding.

DLRover provides an `ElasticDataset` which can query the index of sample and
read the data of smaple by the index. 

```Python
class ElasticDataset(Dataset):
    def __init__(self, data_shard_service=None):
        """The dataset supports elastic training.

        Args:
            data_shard_service: If we want to use elastic training, we
            need to use the `data_shard_service` of the elastic controller
            in elasticai_api.
        """
        self.data_shard_service = data_shard_service
        self._images = images

    def __len__(self):
        if self.data_shard_service:
            # Set the maxsize because the size of dataset is not fixed
            # when using dynamic sharding
            return sys.maxsize
        else:
            return len(self._images)

    def __getitem__(self, index):
        index = self.data_shard_service.fetch_record_index()
        sample = read_sample(index)
```

### Implement rendezvousservice on the DLRover master.

In native Torch Distributed Elastic, each `ElasticAgent`
on a machine synchronizes its state to a `RendezvousBackend`
after the process `ElasticAgent` starts. The `ElasticAgent` on
check whether the minimum number of workers
is reached. After the minimum number of workers is reached,
the `ElasticAgent` will wait for `last_call_timeout` in case
other `ElasticAgent` arrives. Because the `ElasticAgent`
cannot knows the status of nodes in a job. `ElasticAgent` has to
re-rendezvous if a new worker arrives after a rendezvous is ready.
DLRover master can monitor the status of nodes in a job and
update the rendezvous state in `RendezvousBackend`
until existing nodes are running. The `RendezvousBackend` on DLRover
master can reduce the frequency to build rendezvous and restart
worker processes.

```Python
class TorchRendezvousService(object):
    """TorchRendezvousService runs on the DLRover master.
    The service can update the rendezvous states according to
    the node status. 
    """
    def set_state(self, state: _RendezvousState, token: Optional[Token]):
        """Set the _RendezvousState into the store in the master."""
        pass

    def get_state(self) -> _RendezvousState:
        """Return a new state only if len(_RendezvousState.participants)
        + len(_RendezvousState.wait_list) is base 2. Then, we can
        keep the fixed batch size by setting backward_passes_per_step
        in the worker.
        """
        pass
```


```Python
class DlroverRendezvousBackend(RendezvousBackend):
    """DLRover RendezvousBackend runs on the node of PyTorch.
    It can get rendezvous state from DLRover master.
    """
    def get_state(self) -> Optional[Tuple[bytes, Token]]:
        """See base class."""
        pass
        
    def set_state(self):
        """With DLRover master, the agent need not to
        set its state to the master. Bacause the master
        can set the state of the node according to the node status.
        """
        pass
```

### Keep model accuracy consistency with the fixed batch size.

In an elastic PyTorch job, users need to set the maximum number `N` of nodes.
The batch size `B` of sychronized SGD is `N * mini-B` where `mini-B` is
the mini batch size of each worker processes. To fix the batch size, N
need to be base 2, like 1, 2, 4, 8. In an elastic job. DLRover master
will keep the number n of nodes as base 2. Before synchronizing gradients,
each worker process will execute `N/n` steps to accumulate gradient.
Then, the batch size after synchronizing gradients is always N.


```Python
class _DistributedOptimizer(torch.optim.Optimizer):

    def set_backward_passes_per_step(self):
        max_num = int(os.getenv("MAX_WORKER_NUM"))
        cur_num = int(os.getenv("WORKER_NUM"))
        self.backward_passes_per_step = max_num/cur_num

    def step(self, closure=None):
        self._backward_passes += 1
        if (
            self.fixed_global_batch_size
            and self._backward_passes % self.backward_passes_per_step != 0
        ):
            self.update_gradients = False
        else:
            self.update_gradients = True
            self._backward_passes = 0

        if not self.update_gradients:
            return

        if self._should_synchronize:
            self.synchronize()
        self._synchronized = False
        return super(self.__class__, self).step(closure)


def DistributedOptimizer(
    optimizer,
    fixed_global_batch_size=False,
):
    cls = type(
        optimizer.__class__.__name__,
        (optimizer.__class__,),
        dict(_DistributedOptimizer.__dict__),
    )
    return cls(fixed_global_batch_size)
```

Users only need to wrap their optimizer by the `DistributedOptimizer`
of DLRover to keep the batch size fixed.

```Python
model = Net()
optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
optimizer = DistributedOptimizer(optimizer, fixed_global_batch_size=False)
```
