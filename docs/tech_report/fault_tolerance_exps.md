# Fault-tolerance and Elasticity Experiments of DLRover ElasticJob

The tutorial shows experiments to test the fault-tolerance and elasticity
of DLRover ElasticJob. In the experiments, we use the chaos enginerring toolkit
[chaosblade](https://github.com/chaosblade-io/chaosblade) to simulate fault scenarios.

## Preliminary

- Create a k8s cluster and configure cluster credentials on your local computer.
- Deploy DLRover ElasticJob on the k8s cluster with the [tutorial](../tutorial/torch_elasticjob_on_k8s.md).
- Build the image with chaosblade like the [example](../../examples/pytorch/mnist/mnist_chaos.dockerfile).

## Experiments of PyTorch Distributed Job

We conduct experiments to simulate the following scenarios:

- The Pod is preempted.
- The Pod is a straggler.
- The Pod is placed on a fualt node.
- The Pod network breaks down during training.
- The training process corrupts in the Pod.

### Pod is Preempted

In the experiment, we submit a job with the [example](../../examples/pytorch/mnist/chaos_test_job.yaml)
and the command in the worker spec is

```yaml
  command:
    - /bin/bash
    - -c
    - "dlrover-run --network-check --exclude-straggler --nnodes=3:$NODE_NUM \
        --nproc_per_node=2 --max_restarts=3  --rdzv_conf pend_timeout=600 \
        examples/pytorch/mnist/cnn_train.py --num_epochs 5 \
        --training_data /data/mnist_png/training/ \
        --validation_data /data/mnist_png/testing/"
```

The Pods of the job are:

```text
chaos-test-edljob-worker-0                    1/1     Running   0             85s
chaos-test-edljob-worker-1                    1/1     Running   0             85s
chaos-test-edljob-worker-2                    1/1     Running   0             85s
chaos-test-edljob-worker-3                    1/1     Running   0             85s
elasticjob-chaos-test-dlrover-master          1/1     Running   0             89s
```

We kill the worker-0 to simulate that the Pod is preempted by the command

```bash
kubectl -n dlrover delete pod chaos-test-edljob-worker-0
```

After killing worker-0, job Pods are

```text
chaos-test-edljob-worker-1                    1/1     Running   0             2m3s
chaos-test-edljob-worker-2                    1/1     Running   0             2m3s
chaos-test-edljob-worker-3                    1/1     Running   0             2m3s
chaos-test-edljob-worker-4                    1/1     Running   0             30s
elasticjob-chaos-test-dlrover-master          1/1     Running   0             2m7s
```

Then, we can see the log of worker to check whether the training restores.

```bash
kubectl -n dlrover logs chaos-test-edljob-worker-1
```

```text
loss = 2.298487901687622, step = 0
INFO:torch.nn.parallel.distributed:Reducer buckets have been rebuilt in this iteration.
INFO:torch.nn.parallel.distributed:Reducer buckets have been rebuilt in this iteration.
loss = 2.195965051651001, step = 20
loss = 1.2307546138763428, step = 40
loss = 0.6579511761665344, step = 60
loss = 1.0608341693878174, step = 80
loss = 0.7761049270629883, step = 100
```

### Straggler Pod

In the experiment, we set replicas of worker to 4 in a job and
use chaosblade to perform a CPU full load 90% on the `worker-1` with the command

```bash
chaosblade-1.7.2/blade create cpu load --cpu-percent 90
```

If you use the image `registry.cn-hangzhou.aliyuncs.com/intell-ai/dlrover:pytorch-example`,
you can use chaosblade to create a chaos experiment by

```bash
sh examples/pytorch/mnist/start_chaos.sh cpu-overload 
```

and set the command in the yaml of elasticjob like the [example](../../examples/pytorch/mnist/choas_test_job.yaml).

```yaml
  command:
    - /bin/bash
    - -c
    - "(bash examples/pytorch/mnist/start_chaos.sh cpu-overload &) && \
        dlrover-run --network-check --exclude-straggler --nnodes=3:$NODE_NUM \
        --nproc_per_node=2 --max_restarts=3  --rdzv_conf pend_timeout=600 \
        examples/pytorch/mnist/cnn_train.py --num_epochs 5 \
        --training_data /data/mnist_png/training/ \
        --validation_data /data/mnist_png/testing/"
```

After submitting an ElasticJob to the k8s cluster by
`kubectl -n dlrover apply -f examples/pytorch/mnist/choas_test_job.yaml`,
We can see the `worker-1` exits with errors like

```text
elasticjob-torch-mnist-debug-dlrover-master   0/1     Completed   0             3h17m
torch-mnist-debug-edljob-worker-0             0/1     Completed   0             3h17m
torch-mnist-debug-edljob-worker-1             0/1     Error       0             3h17m
torch-mnist-debug-edljob-worker-2             0/1     Completed   0             3h17m
torch-mnist-debug-edljob-worker-3             0/1     Completed   0             3h17m
torch-mnist-debug-edljob-worker-4             0/1     Completed   0             3h10m
```

From the log of worker-1 by `kubectl -n dlrover logs torch-mnist-debug-edljob-worker-1`,
worker-1 fails because it is a straggler. If we don't want to the worker-1 fails due to
straggler, we can remove the config `dlrover-run --network-check --exclude-straggler`
from the command like `dlrover-run --network-check`.

```text
[2023-09-26 03:52:20,235] [INFO] [training.py:707:run] Fault nodes are: []  and stragglers are: [1].
Traceback (most recent call last):
  File "/usr/local/bin/dlrover-run", line 8, in <module>
    sys.exit(main())
  ...
  File "/usr/local/lib/python3.8/site-packages/dlrover/python/elastic_agent/torch/training.py", line 733, in run
    raise RuntimeError("The node is a straggler and exits.")
RuntimeError: The node is a straggler and exits.

```

We can see the elapsed time of each node to run the task to check straggler in the master log.

```bash
kubectl -n dlrover logs elasticjob-torch-mnist-debug-dlrover-master | grep elapsed
```

```text
Round 0: The node elapsed time are {2: 20.307, 3: 20.265, 0: 206.872, 1: 151.752}
Round 1: The node elapsed time are {2: 20.307, 3: 20.265, 0: 23.174, 1: 135.961}
Round 2: The node elapsed time aree {2: 21.491, 0: 22.685, 3: 20.889, 1: 23.097}
```

From the log, The worker-1 the elapsed time is much bigger than others in the first 2 rounds.
After the worker-1 fails, the ElasticJob relaunch a new Pod worker-4 to restore the failed Pod.
The elapsed times of all nodes have not significant differenct. Note. the index is the
`WOKRER_RANK` of node. The `WORKER_RANK` of worker-4 is the same as worker-1.

### Fault Node

In the experiment, we set replicas of worker to 4 in a job and
use chaosblade to kill the process to run `nvidia_gpu.py`
to simulate the fault node.

and set the command in the yaml of elasticjob like the [example](../../examples/pytorch/mnist/choas_test_job.yaml).

```yaml
command:
    - /bin/bash
    - -c
    - "(bash examples/pytorch/mnist/start_chaos.sh kill-process &) && \
        dlrover-run --network-check --exclude-straggler --nnodes=3:$NODE_NUM \
        --nproc_per_node=2 --max_restarts=3  --rdzv_conf pend_timeout=600 \
        examples/pytorch/mnist/cnn_train.py --num_epochs 5 \
        --training_data /data/mnist_png/training/ \
        --validation_data /data/mnist_png/testing/"
```

```text
chaos-test-edljob-worker-0                    1/1     Running             0             12m
chaos-test-edljob-worker-1                    0/1     Error               0             12m
chaos-test-edljob-worker-2                    1/1     Running             0             12m
chaos-test-edljob-worker-3                    1/1     Running             0             12m
chaos-test-edljob-worker-4                    1/1     Running             0             3m59s
elasticjob-chaos-test-dlrover-master          1/1     Running             0             12m
```

The worker-1 fails with the message

```text
Traceback (most recent call last):
  ....
  File "/usr/local/lib/python3.8/site-packages/dlrover/python/elastic_agent/torch/training.py", line 732, in run
    raise RuntimeError("The node network is breakdown.")
RuntimeError: The node network is breakdown.
```

From the master log by `kubectl -n dlrover logs elasticjob-chaos-test-dlrover-master | grep "The node status"`,
the worker-1 fails in the first 2 round check. Afther worker-4 starts to replace worker-1,
all nodes are noraml.

```text
Round 1: The node status are {1: False, 2: True, 3: True, 0: False}.
Round 2: The node status are {1: False, 2: True, 3: True, 0: True}.
Round 3: The node status are {3: True, 0: True, 1: True, 2: True}.
```

### Network Breakdown

In the experiment, we set replicas of worker to 4 in a job and
use chaosblade to set the network loss rate to 100% to simulate
that the network of the node is breakdown.

We watch the log of worker-1 to check whether the training starts.
The training starts if `loss=..., step=...` in the log.
After the training starts, we perform a network loadd rate 100%
in the worker-1 to simulate the networker of worker-1 is breakdown.

```bash
kubectl -n dlrover exec -it chaos-test-edljob-worker-1  bash
./chaosblade-1.7.2/blade create network loss --percent 100 --interface eth0
```

Then, the worker-1 fails and a new worker-4 starts to replace the worker-1.

```text
chaos-test-edljob-worker-0                    1/1     Running   0             4m39s
chaos-test-edljob-worker-1                    0/1     Error     0             4m39s
chaos-test-edljob-worker-2                    1/1     Running   0             4m39s
chaos-test-edljob-worker-3                    1/1     Running   0             4m39s
chaos-test-edljob-worker-4                    1/1     Running   0             17s
elasticjob-chaos-test-dlrover-master          1/1     Running   0             4m43s
```

The training also restores after the worker-4 starts by the log of worker-0.

```text
loss = 0.24101698398590088, step = 0
INFO:torch.nn.parallel.distributed:Reducer buckets have been rebuilt in this iteration.
INFO:torch.nn.parallel.distributed:Reducer buckets have been rebuilt in this iteration.
loss = 0.4646361768245697, step = 20
```

### Training Process Corruption

In the experiment, we set replicas of worker to 4 in a job and
use chaosblade to kill a training process to simulate the GPU error.

We watch the log of worker-1 to check whether the training starts.
The training starts if `loss=..., step=...` in the log.
After the training starts, we kill a process in the worker-1.

```bash
kubectl -n dlrover exec -it chaos-test-edljob-worker-1  bash
ps -aux | grep cnn_train.py
```

Then, we can kill a training process by `kill -9 ${PID}`. The all workers
are still running and we can see that the training restarts from the log.

```text
chaos-test-edljob-worker-0                    1/1     Running   0             3m4s
chaos-test-edljob-worker-1                    1/1     Running   0             3m4s
chaos-test-edljob-worker-2                    1/1     Running   0             3m4s
chaos-test-edljob-worker-3                    1/1     Running   0             3m4s
elasticjob-chaos-test-dlrover-master          1/1     Running   0             3m9s
```

### Scale Up Nodes

In the experiment, we use the [example](../../examples/pytorch/mnist/elastic_job.yaml)
to submit an elastic training job. In the job, we set the `min_node=3` and
`max_node=$NODE_NUM` as the number of replicas. The ElasticJob will set the replicas
into the environment `NODE_NUM`.

At first, there are 3 running workers and 1 pending worker due to the insufficient resource.

```text
elasticjob-torch-mnist-dxlrover-master           1/1     Running     0             57s
torch-mnist-edljob-worker-0                      1/1     Running     0             47s
torch-mnist-edljob-worker-1                      1/1     Running     0             47s
torch-mnist-edljob-worker-2                      1/1     Running     0             47s
torch-mnist-edljob-worker-3                      0/1     Pending     0             47s
```

After about 2 min, we can see the training starts in 3 running workers with the log.

```text
[2023-09-27 02:23:21,097] [INFO] [training.py:344:_rendezvous] [default] Rendezvous complete for workers. Result:
  restart_count=0
  master_addr=192.168.0.71
  master_port=36725
  group_rank=0
  group_world_size=3
  local_ranks=[0, 1]
  role_ranks=[0, 1]
  global_ranks=[0, 1]
  role_world_sizes=[6, 6]
  global_world_sizes=[6, 6]

rank 1 is initialized local_rank = 1
loss = 2.3198373317718506, step = 0
loss = 2.2946105003356934, step = 0
loss = 1.7543025016784668, step = 20
```

Then, we kill another job to release resource and the worker-3 will start.

```text
elasticjob-torch-mnist-dlrover-master         1/1     Running   0             5m39s
torch-mnist-edljob-worker-0                   1/1     Running   0             5m34s
torch-mnist-edljob-worker-1                   1/1     Running   0             5m34s
torch-mnist-edljob-worker-2                   1/1     Running   0             5m34s
torch-mnist-edljob-worker-3                   1/1     Running   0             5m34s
```

From the log of worker-0, we can see the training starts with `group_world_size=4`.

```text
[2023-09-27 02:25:43,362] [INFO] [training.py:344:_rendezvous] [default] Rendezvous complete for workers. Result:
  restart_count=1
  master_addr=192.168.0.71
  master_port=58241
  group_rank=0
  group_world_size=4
  local_ranks=[0, 1]
  role_ranks=[0, 1]
  global_ranks=[0, 1]
  role_world_sizes=[8, 8]
  global_world_sizes=[8, 8]

rank 1 is initialized local_rank = 1rank 0 is initialized local_rank = 0

loss = 2.2984073162078857, step = 0
loss = 2.1407980918884277, step = 20
loss = 1.1324385404586792, step = 40
loss = 0.4783979058265686, step = 60
loss = 0.5714012384414673, step = 80
loss = 0.6941334009170532, step = 100
```

### Scale Down Nodes

In the experiment, we use the [example](../../examples/pytorch/mnist/elastic_job.yaml)
to submit an elastic training job. In the job, we set the `min_node=3` and
`max_node=$NODE_NUM` as the number of replicas. The ElasticJob will set the replicas
into the environment `NODE_NUM`.

At first, there are 4 running workers.

```text
elasticjob-torch-mnist-dlrover-master            1/1     Running     0             2m43s
torch-mnist-edljob-worker-0                      1/1     Running     0             2m38s
torch-mnist-edljob-worker-1                      1/1     Running     0             2m38s
torch-mnist-edljob-worker-2                      1/1     Running     0             2m38s
torch-mnist-edljob-worker-3                      0/1     Running     0             2m38s
```

Then, we use the chaosblade to make worker-1 failed.

```bash
kubectl -n dlrover exec -it torch-mnist-edljob-worker-1 bash
./chaosblade-1.7.2/blade create process kill --process dlrover-run --signal 1
```

```text
elasticjob-torch-mnist-dlrover-master         1/1     Running   0             4m43s
torch-mnist-edljob-worker-0                   1/1     Running   0             4m38s
torch-mnist-edljob-worker-1                   0/1     Error     0             4m38s
torch-mnist-edljob-worker-2                   1/1     Running   0             4m38s
torch-mnist-edljob-worker-3                   1/1     Running   0             4m38s
```

From the log of worker-0, we can see the training restores the model and data sampler
from the checkpoint and starts with `group_world_size=3`.

```text
[2023-09-27 03:18:00,815] [INFO] [training.py:344:_rendezvous] [default] Rendezvous complete for workers. Result:
  restart_count=1
  master_addr=192.168.0.66
  master_port=39705
  group_rank=0
  group_world_size=3
  local_ranks=[0, 1]
  role_ranks=[0, 1]
  global_ranks=[0, 1]
  role_world_sizes=[6, 6]
  global_world_sizes=[6, 6]

[2023-09-27 03:18:05,957] [INFO] [sampler.py:153:load_state_dict] Load epoch = 0, completed num = 51200, num_samples = 1467
[2023-09-27 03:18:05,958] [INFO] [sampler.py:153:load_state_dict] Load epoch = 0, completed num = 51200, num_samples = 1467
loss = 0.2617453336715698, step = 0
loss = 0.2548859417438507, step = 20
```

## Experiments of TensorFlow PS Distributed Job

We conduct experiments with the TF distributed job using PS to
test the fault-tolerance of worker and PS.

### Fault-tolerance of Worker

We can sumit a TensorFlow PS job using the [example](../../examples/tensorflow/criteo_deeprec/manual_job.yaml).
The job will launch 1 chief, 1 worker and 1 PS.

```bash
kubectl -n dlrover apply -f examples/tensorflow/criteo_deeprec/manual_job.yaml
```

```text
deepctr-manual-scale-edljob-chief-0              1/1     Running   0             88s
deepctr-manual-scale-edljob-ps-0                 1/1     Running   0             88s
deepctr-manual-scale-edljob-worker-0             1/1     Running   0             88s
elasticjob-deepctr-manual-scale-dlrover-master   1/1     Running   0             99s
```

We use `kubectl` to kill a worker.

```bash
kubectl -n dlrover delete pod deepctr-manual-scale-edljob-worker-0
```

After the worker-0 is killed, the job relaunch the worker-1 to restore the failed node.

```text
NAME                                                 READY   STATUS    RESTARTS   AGE
deepctr-manual-scale-edljob-chief-0              1/1     Running   0             2m57s
deepctr-manual-scale-edljob-ps-0                 1/1     Running   0             2m57s
deepctr-manual-scale-edljob-worker-1             1/1     Running   0             60s
elasticjob-deepctr-manual-scale-dlrover-master   1/1     Running   0             3m8s
```

After the job runs about 4min, the chief-0 fails with OOM due to the insufficient memory
configuration. The job relaunches the chief-1 with more memory to restore it.

```text
deepctr-manual-scale-edljob-chief-0              0/1     OOMKilled   0             4m53s
deepctr-manual-scale-edljob-chief-1              1/1     Running     0             64s
deepctr-manual-scale-edljob-ps-0                 1/1     Running     0             4m53s
deepctr-manual-scale-edljob-worker-1             1/1     Running     0             2m56s
```

We can view the memory of chief-0 and chief-1 by

```bash
kubectl -n dlrover get pod deepctr-manual-scale-edljob-chief-0 -o yaml | grep memory

>>>
        memory: 4Gi
        memory: 4Gi
```

```bash
kubectl -n dlrover get pod deepctr-manual-scale-edljob-chief-1 -o yaml | grep memory

>>>
        memory: 8Gi
        memory: 8Gi
```

We can view the log of chief-1 to check whether the training restores.

```shell
[2023-03-20 11:51:10,774] [INFO][session_manager.py:511:_try_run_local_init_op] Running local_init_op.
[2023-03-20 11:51:11,302] [INFO][session_manager.py:513:_try_run_local_init_op] Done running local_init_op.
[2023-03-20 11:51:14,279] [INFO][global_step_hook.py:39:before_run] global_step: 126
```

### Fault-tolerance of PS

We kill the ps-0 by `kubectl -n dlrover delete pod deepctr-manual-scale-edljob-ps-0`.
The job relaunches the ps-1 to restore the killed ps-0>

```text
deepctr-manual-scale-edljob-chief-0              0/1     OOMKilled   0             10m
deepctr-manual-scale-edljob-chief-1              1/1     Running     0             7m1s
deepctr-manual-scale-edljob-ps-1                 1/1     Running     0             109s
deepctr-manual-scale-edljob-worker-1             0/1     OOMKilled   0             8m53s
deepctr-manual-scale-edljob-worker-2             1/1     Running     0             4m13s
elasticjob-deepctr-manual-scale-dlrover-master   1/1     Running     0             11m
```

From the log of chief, the training job restore the model from the latest checkpoint
and contiune training the model.

```text
[2023-09-26 19:24:00,861] [INFO][saver.py:1531:restore] Restoring parameters from /nas/deepctr/model.ckpt-126
[2023-09-26 19:24:03,473] [INFO][session_manager.py:511:_try_run_local_init_op] Running local_init_op.
[2023-09-26 19:24:03,580] [INFO] [resource.py:164:report_resource] Report Resource CPU : 0.98, Memory 7146, GPU []
[2023-09-26 19:24:03,670] [INFO][session_manager.py:513:_try_run_local_init_op] Done running local_init_op.
[2023-09-26 19:24:07,665] [INFO][basic_session_run_hooks.py:627:_save] Saving checkpoints for 126 into /nas/deepctr/model.ckpt.
```
