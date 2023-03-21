# worker和ps容错样例
## worker容错示例
在任务运行过程中删除worker-i对应的pod，之后dlrover master会重新拉起一个pod。work-i对应的pod的名称会发生变化，新创建的pod的启动命令和被kill掉的pod的启动命令相同，启动后参与组网，并进行训练。期间其他worker不受影响。
### 启动作业
首先，启动作业。为了避免自动扩容缩容的影响，选择人工配置扩容缩容策略。
```shell
kubectl apply -f deepctr_manual_scale_job.yaml -n dlrover
```
当前有1个ps和3个worker。
```shell
NAME                                                 READY   STATUS    RESTARTS   AGE
deepctr-auto-scaling-job-edljob-chief-0              1/1     Running   0          117s
deepctr-auto-scaling-job-edljob-ps-0                 1/1     Running   0          117s
deepctr-auto-scaling-job-edljob-worker-0             1/1     Running   0          65s
deepctr-auto-scaling-job-edljob-worker-1             1/1     Running   0          65s
```
查看worker-0对应的pod的信息
```shell
Name:             deepctr-auto-scaling-job-edljob-worker-0
Namespace:        dlrover
Priority:         0
Service Account:  default
Node:             cn-beijing.192.168.0.13/192.168.0.13
Start Time:       Mon, 20 Mar 2023 10:17:10 +0800
Labels:           app=dlrover
                  elasticjob-name=deepctr-auto-scaling-job
                  rank-index=0
                  replica-index=0
                  replica-type=worker
                  restart-count=0
Annotations:      k8s.aliyun.com/pod-ips: 192.168.0.65
                  kubernetes.io/psp: ack.privileged
Status:           Running
IP:               192.168.0.65
IPs:
  IP:           192.168.0.65
Controlled By:  ElasticJob/deepctr-auto-scaling-job
Containers:
  main:
    Container ID:  containerd://b1ad0d4b08efa07ea79bc5af0ea2eca67f2d9e91ad0023ed57e89a933b122ee4
    Image:         registry.cn-hangzhou.aliyuncs.com/dlrover_deeprec/deeprec:v11
    Image ID:      registry.cn-hangzhou.aliyuncs.com/dlrover_deeprec/deeprec@sha256:d0159b59af3dfb9e9ab4384945ef2b3b2a9cf3250dbe0a1bc06c06421ef8c780
    Port:          <none>
    Host Port:     <none>
    Command:
      /bin/bash
      -c
      pip install pyhocon && cd /usr/local/lib/python3.8/dist-packages/dlrover/trainer/examples/deepfm_deeprec && python -m dlrover.trainer.entry.local_entry --platform=Kubernetes --conf=deepfm_deeprec_conf.TrainConf --enable_auto_scaling=True
    State:          Running
      Started:      Mon, 20 Mar 2023 10:17:11 +0800
    Ready:          True
    Restart Count:  0
    Limits:
      cpu:     500m
      memory:  4Gi
    Requests:
      cpu:     500m
      memory:  4Gi
    Environment:
      DLROVER_MASTER_ADDR:  elasticjob-deepctr-auto-scaling-job-dlrover-master:50001
      WORKER_TYPE:          worker
      WORKER_ID:            0
      WORKER_RANK:          0
      WORKER_NUM:           1
      TF_CONFIG:            {"cluster":{"worker":["deepctr-auto-scaling-job-edljob-worker-0:3333"],"ps":["deepctr-auto-scaling-job-edljob-ps-0.dlrover.svc:2222"],"chief":["deepctr-auto-scaling-job-edljob-chief-0:3333"]},"task":{"type":"worker","index":0}}
    Mounts:
      /nas from pvc-nas (rw)
      /var/run/secrets/kubernetes.io/serviceaccount from kube-api-access-jtpfw (ro)
Conditions:
  Type              Status
  Initialized       True 
  Ready             True 
  ContainersReady   True 
  PodScheduled      True 
Volumes:
  pvc-nas:
    Type:       PersistentVolumeClaim (a reference to a PersistentVolumeClaim in the same namespace)
    ClaimName:  pvc-nas
    ReadOnly:   false
  kube-api-access-jtpfw:
    Type:                    Projected (a volume that contains injected data from multiple sources)
    TokenExpirationSeconds:  3607
    ConfigMapName:           kube-root-ca.crt
    ConfigMapOptional:       <nil>
    DownwardAPI:             true
QoS Class:                   Guaranteed
Node-Selectors:              <none>
Tolerations:                 node.kubernetes.io/not-ready:NoExecute op=Exists for 300s
                             node.kubernetes.io/unreachable:NoExecute op=Exists for 300s
Events:
  Type    Reason          Age    From               Message
  ----    ------          ----   ----               -------
  Normal  Scheduled       2m14s  default-scheduler  Successfully assigned dlrover/deepctr-auto-scaling-job-edljob-worker-0 to cn-beijing.192.168.0.13
  Normal  AllocIPSucceed  2m14s  terway-daemon      Alloc IP 192.168.0.65/24
  Normal  Pulling         2m14s  kubelet            Pulling image "registry.cn-hangzhou.aliyuncs.com/dlrover_deeprec/deeprec:v11"
  Normal  Pulled          2m14s  kubelet            Successfully pulled image "registry.cn-hangzhou.aliyuncs.com/dlrover_deeprec/deeprec:v11" in 282.030396ms (282.03863ms including waiting)
  Normal  Created         2m13s  kubelet            Created container main
  Normal  Started         2m13s  kubelet            Started container main
```
### 容错模拟
为了模拟容错，需要主动删除worker-0对应的pod
```shell
kubectl delete pods -n dlrover deepctr-auto-scaling-job-edljob-worker-0
pod "deepctr-auto-scaling-job-edljob-worker-0" deleted
```
worker-0对应的新pod启动，完成准备工作后开始消费数据，进行训练。
```shell
deepctr-auto-scaling-job-edljob-chief-0              1/1     Running             0          4m24s
deepctr-auto-scaling-job-edljob-ps-0                 1/1     Running             0          4m24s
deepctr-auto-scaling-job-edljob-worker-1             1/1     Running             0          3m32s
deepctr-auto-scaling-job-edljob-worker-2             0/1     ContainerCreating   0          2s
```
查看worker-0对应的pod的信息
```shell
Name:             deepctr-auto-scaling-job-edljob-worker-2
Namespace:        dlrover
Priority:         0
Service Account:  default
Node:             cn-beijing.192.168.0.13/192.168.0.13
Start Time:       Mon, 20 Mar 2023 11:50:34 +0800
Labels:           app=dlrover
                  elasticjob-name=deepctr-auto-scaling-job
                  rank-index=0
                  replica-index=2
                  replica-type=worker
                  restart-count=0
Annotations:      k8s.aliyun.com/pod-ips: 192.168.0.63
                  kubernetes.io/psp: ack.privileged
Status:           Running
IP:               192.168.0.63
IPs:
  IP:           192.168.0.63
Controlled By:  ElasticJob/deepctr-auto-scaling-job
Containers:
  main:
    Container ID:  containerd://31b97063042b4f5569be958b79fe28c555ed2802a9fdd3fcbd79b6a1a779fdb0
    Image:         registry.cn-hangzhou.aliyuncs.com/dlrover_deeprec/deeprec:v11
    Image ID:      registry.cn-hangzhou.aliyuncs.com/dlrover_deeprec/deeprec@sha256:d0159b59af3dfb9e9ab4384945ef2b3b2a9cf3250dbe0a1bc06c06421ef8c780
    Port:          <none>
    Host Port:     <none>
    Command:
      /bin/bash
      -c
      pip install pyhocon && cd /usr/local/lib/python3.8/dist-packages/dlrover/trainer/examples/deepfm_deeprec && python -m dlrover.trainer.entry.local_entry --platform=Kubernetes --conf=deepfm_deeprec_conf.TrainConf --enable_auto_scaling=True
    State:          Running
      Started:      Mon, 20 Mar 2023 11:50:36 +0800
    Ready:          True
    Restart Count:  0
    Limits:
      cpu:     500m
      memory:  4Gi
    Requests:
      cpu:     500m
      memory:  4Gi
    Environment:
      DLROVER_MASTER_ADDR:  elasticjob-deepctr-auto-scaling-job-dlrover-master:50001
      WORKER_TYPE:          worker
      WORKER_ID:            2
      WORKER_RANK:          0
      WORKER_NUM:           1
      TF_CONFIG:            {"cluster":{"worker":["deepctr-auto-scaling-job-edljob-worker-0:3333"],"ps":["deepctr-auto-scaling-job-edljob-ps-0.dlrover.svc:2222"],"chief":["deepctr-auto-scaling-job-edljob-chief-0:3333"]},"task":{"type":"worker","index":0}}
    Mounts:
      /nas from pvc-nas (rw)
      /var/run/secrets/kubernetes.io/serviceaccount from kube-api-access-n4lq9 (ro)
Conditions:
  Type              Status
  Initialized       True 
  Ready             True 
  ContainersReady   True 
  PodScheduled      True 
Volumes:
  pvc-nas:
    Type:       PersistentVolumeClaim (a reference to a PersistentVolumeClaim in the same namespace)
    ClaimName:  pvc-nas
    ReadOnly:   false
  kube-api-access-n4lq9:
    Type:                    Projected (a volume that contains injected data from multiple sources)
    TokenExpirationSeconds:  3607
    ConfigMapName:           kube-root-ca.crt
    ConfigMapOptional:       <nil>
    DownwardAPI:             true
QoS Class:                   Guaranteed
Node-Selectors:              <none>
Tolerations:                 node.kubernetes.io/not-ready:NoExecute op=Exists for 300s
                             node.kubernetes.io/unreachable:NoExecute op=Exists for 300s
Events:
  Type    Reason          Age   From               Message
  ----    ------          ----  ----               -------
  Normal  Scheduled       93s   default-scheduler  Successfully assigned dlrover/deepctr-auto-scaling-job-edljob-worker-2 to cn-beijing.192.168.0.13
  Normal  AllocIPSucceed  92s   terway-daemon      Alloc IP 192.168.0.63/24
  Normal  Pulling         92s   kubelet            Pulling image "registry.cn-hangzhou.aliyuncs.com/dlrover_deeprec/deeprec:v11"
  Normal  Pulled          92s   kubelet            Successfully pulled image "registry.cn-hangzhou.aliyuncs.com/dlrover_deeprec/deeprec:v11" in 314.52769ms (314.541567ms including waiting)
  Normal  Created         92s   kubelet            Created container main
  Normal  Started         92s   kubelet            Started container main
```
worker-0 对应pod的日志
```shell
[2023-03-20 11:51:10,774] [INFO][session_manager.py:511:_try_run_local_init_op] Running local_init_op.
[2023-03-20 11:51:11,302] [INFO][session_manager.py:513:_try_run_local_init_op] Done running local_init_op.
[2023-03-20 11:51:14,279] [INFO][global_step_hook.py:39:before_run] global_step: 10488361
```
## ps容错示例
运行过程中删除一个ps-i对应的pod，之后dlrover master会重新拉起一个pod，ps-i对应的pod的名称会发生变化，但是新创建的pod的启动命令和被kill掉的pod的启动命令相同。在pod被kill到新的pod启动ps创建server之前，worker训练会中断。
### 启动作业
启动作业之后，可以查看当前运行的worker和ps。

```shell
NAME                                                 READY   STATUS    RESTARTS   AGE
deepctr-auto-scaling-job-edljob-chief-0              1/1     Running   0          4m3s
deepctr-auto-scaling-job-edljob-ps-0                 1/1     Running   0          4m3s
deepctr-auto-scaling-job-edljob-ps-1                 1/1     Running   0          106s
deepctr-auto-scaling-job-edljob-worker-0             1/1     Running   0          2m30s
deepctr-auto-scaling-job-edljob-worker-1             1/1     Running   0          2m30s
dlrover-controller-manager-7dccdf6c4d-jp4wb          2/2     Running   0          3h26m
elasticjob-deepctr-auto-scaling-job-dlrover-master   1/1     Running   0          4m9s
mysql-7d757854f-8l5k4                                1/1     Running   0          4d4h
```
### 容错模拟
为了模拟容错，需要主动删除ps-0对应的pod，删除后worker的日志
```shell
[2023-03-20 15:04:34,350] [INFO][monitored_session.py:1336:run] An error was raised. This may be due to a preemption in a connected worker or parameter server. The current session will be closed and a new session will be created. This error may also occur due to a gRPC failure caused by high memory or network bandwidth usage in the parameter servers. If this error occurs repeatedly, try increasing the number of parameter servers assigned to the job. Error: From /job:ps/replica:0/task:1:
RecvTensor expects a different device incarnation: 11288349594494262162 vs. 11542130100054943552. Your worker job ("/job:localhost/replica:0/task:0") was probably restarted. Check your worker job for the reason why it was restarted.
```

当ps pod重新创建，ps server启动
```shell
NAME                                                 READY   STATUS    RESTARTS   AGE
deepctr-auto-scaling-job-edljob-chief-0              1/1     Running   0          11m
deepctr-auto-scaling-job-edljob-ps-1                 1/1     Running   0          8m55s
deepctr-auto-scaling-job-edljob-ps-2                 1/1     Running   0          6m13s
deepctr-auto-scaling-job-edljob-worker-0             1/1     Running   0          9m39s
deepctr-auto-scaling-job-edljob-worker-1             1/1     Running   0          9m39s
```
worker会加载最近一次的checkpoint，并继续训练
```shell
[2023-03-20 15:04:34,100] [INFO][monitored_session.py:1336:run] An error was raised. This may be due to a preemption in a connected worker or parameter server. The current session will be closed and a new session will be created. This error may also occur due to a gRPC failure caused by high memory or network bandwidth usage in the parameter servers. If this error occurs repeatedly, try increasing the number of parameter servers assigned to the job. Error: 
=====================
Aborted: From /job:chief/replica:0/task:0:
RecvTensor expects a different device incarnation: 11288349594494262162 vs. 11542130100054943552. Your worker job ("/job:localhost/replica:0/task:0") was probably restarted. Check your worker job for the reason why it was restarted.
Additional GRPC error information:
{"created":"@1679295874.088182934","description":"Error received from peer","file":"external/grpc/src/core/lib/surface/call.cc","file_line":1039,"grpc_message":"RecvTensor expects a different device incarnation: 11288349594494262162 vs. 11542130100054943552. Your worker job ("/job:localhost/replica:0/task:0") was probably restarted. Check your worker job for the reason why it was restarted.","grpc_status":10}
	 [[node global_step (defined at /local/lib/python3.8/dist-packages/tensorflow_core/python/framework/ops.py:1748) ]]
Aborted: From /job:ps/replica:0/task:0:
Session handle is not found: f8368e3b7d417955. Possibly this worker ("/job:localhost/replica:0/task:0") just restarted.
=====================


Original stack trace for 'global_step':
  File "/lib/python3.8/threading.py", line 890, in _bootstrap
    self._bootstrap_inner()
  File "/lib/python3.8/threading.py", line 932, in _bootstrap_inner
    self.run()
  File "/lib/python3.8/threading.py", line 870, in run
    self._target(*self._args, **self._kwargs)
  File "/local/lib/python3.8/dist-packages/dlrover/trainer/worker/tf_kubernetes_worker.py", line 56, in run_worker
    self.estimator.train_and_evaluate()
  File "/local/lib/python3.8/dist-packages/dlrover/trainer/tensorflow/executor/estimator_executor.py", line 273, in train_and_evaluate
    tf.estimator.train_and_evaluate(
  File "/local/lib/python3.8/dist-packages/tensorflow_estimator/python/estimator/training.py", line 473, in train_and_evaluate
    return executor.run()
  File "/local/lib/python3.8/dist-packages/tensorflow_estimator/python/estimator/training.py", line 640, in run
    getattr(self, task_to_run)()
  File "/local/lib/python3.8/dist-packages/tensorflow_estimator/python/estimator/training.py", line 645, in run_chief
    return self._start_distributed_training()
  File "/local/lib/python3.8/dist-packages/tensorflow_estimator/python/estimator/training.py", line 790, in _start_distributed_training
    self._estimator.train(
  File "/local/lib/python3.8/dist-packages/tensorflow_estimator/python/estimator/estimator.py", line 370, in train
    loss = self._train_model(input_fn, hooks, saving_listeners)
  File "/local/lib/python3.8/dist-packages/tensorflow_estimator/python/estimator/estimator.py", line 1166, in _train_model
    return self._train_model_default(input_fn, hooks, saving_listeners)
  File "/local/lib/python3.8/dist-packages/tensorflow_estimator/python/estimator/estimator.py", line 1184, in _train_model_default
    global_step_tensor = self._create_and_assert_global_step(g)
  File "/local/lib/python3.8/dist-packages/tensorflow_estimator/python/estimator/estimator.py", line 1082, in _create_and_assert_global_step
    step = self._create_global_step(graph)
  File "/local/lib/python3.8/dist-packages/tensorflow_estimator/python/estimator/estimator.py", line 1071, in _create_global_step
    return training.create_global_step(graph)
  File "/local/lib/python3.8/dist-packages/tensorflow_core/python/training/training_util.py", line 137, in create_global_step
    return variable_scope.get_variable(
  File "/local/lib/python3.8/dist-packages/tensorflow_core/python/ops/variable_scope.py", line 1951, in get_variable
    return get_variable_scope().get_variable(
  File "/local/lib/python3.8/dist-packages/tensorflow_core/python/ops/variable_scope.py", line 1509, in get_variable
    return var_store.get_variable(
  File "/local/lib/python3.8/dist-packages/tensorflow_core/python/ops/variable_scope.py", line 786, in get_variable
    return _true_getter(
  File "/local/lib/python3.8/dist-packages/tensorflow_core/python/ops/variable_scope.py", line 731, in _true_getter
    return self._get_single_variable(
  File "/local/lib/python3.8/dist-packages/tensorflow_core/python/ops/variable_scope.py", line 1199, in _get_single_variable
    v = variables.VariableV1(
  File "/local/lib/python3.8/dist-packages/tensorflow_core/python/ops/variables.py", line 460, in __call__
    return cls._variable_v1_call(*args, **kwargs)
  File "/local/lib/python3.8/dist-packages/tensorflow_core/python/ops/variables.py", line 401, in _variable_v1_call
    return previous_getter(
  File "/local/lib/python3.8/dist-packages/tensorflow_core/python/ops/variables.py", line 394, in <lambda>
    previous_getter = lambda **kwargs: default_variable_creator(None, **kwargs)
  File "/local/lib/python3.8/dist-packages/tensorflow_core/python/ops/variable_scope.py", line 3389, in default_variable_creator
    return variables.RefVariable(
  File "/local/lib/python3.8/dist-packages/tensorflow_core/python/ops/variables.py", line 464, in __call__
    return super(VariableMetaclass, cls).__call__(*args, **kwargs)
  File "/local/lib/python3.8/dist-packages/tensorflow_core/python/ops/variables.py", line 1883, in __init__
    self._init_from_args(
  File "/local/lib/python3.8/dist-packages/tensorflow_core/python/ops/variables.py", line 2030, in _init_from_args
    self._variable = state_ops.variable_op_v2(
  File "/local/lib/python3.8/dist-packages/tensorflow_core/python/ops/state_ops.py", line 76, in variable_op_v2
    return gen_state_ops.variable_v2(
  File "/local/lib/python3.8/dist-packages/tensorflow_core/python/ops/gen_state_ops.py", line 1619, in variable_v2
    _, _, _op = _op_def_lib._apply_op_helper(
  File "/local/lib/python3.8/dist-packages/tensorflow_core/python/framework/op_def_library.py", line 792, in _apply_op_helper
    op = g.create_op(op_type_name, inputs, dtypes=None, name=scope,
  File "/local/lib/python3.8/dist-packages/tensorflow_core/python/util/deprecation.py", line 507, in new_func
    return func(*args, **kwargs)
  File "/local/lib/python3.8/dist-packages/tensorflow_core/python/framework/ops.py", line 3360, in create_op
    return self._create_op_internal(op_type, inputs, dtypes, input_types, name,
  File "/local/lib/python3.8/dist-packages/tensorflow_core/python/framework/ops.py", line 3422, in _create_op_internal
    ret = Operation(
  File "/local/lib/python3.8/dist-packages/tensorflow_core/python/framework/ops.py", line 1748, in __init__
    self._traceback = tf_stack.extract_stack()

[2023-03-20 15:04:34,499] [INFO][monitored_session.py:256:finalize] Graph was finalized.
[2023-03-20 15:04:34,511] [INFO][session_manager.py:220:_restore_checkpoint] run with loading checkpoint
[2023-03-20 15:04:34,724] [INFO][saver.py:1531:restore] Restoring parameters from /nas/model.ckpt-10701903
```

 
