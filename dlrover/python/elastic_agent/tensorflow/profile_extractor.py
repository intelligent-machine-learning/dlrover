# Copyright 2022 The DLRover Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json


class OperationStats(object):
    """Collect the stats of TensorFlow operations.
    op_count: the total count of operations in a graph.
    recv_op_count: the total count of RecvTensor on the chief in a graph.
    update_op_count: the total count of the updation ops in optimizer to
        update weights with gradients.
    input_fetch_dur: the time (microsecond) of the IterationGetNext to get
        the batch input data.
    flops: the flops in the graph.
    """

    def __init__(self):
        self.op_count = 0
        self.recv_op_count = 0
        self.update_op_count = 0
        self.read_op_count = 0
        self.input_fetch_dur = 0
        self.flops = 0  # static flops from initilizing graph
        self.runtime_flops = 0

    def to_json(self):
        return json.dump(self.__dict__)


class TensorStats(object):
    """Collect the stats of TensorFlow tensors.
    tensor_alloc_bytes: a dict where the key is the name of device and the
        value is the totabl bytes of tensors on the device.
    variabel_count: the total count of variabels in a graph.
    total_variable_size: the total size of variables in a graph.
    max_variable_size: the maximum size of variables in a graph.
    kv_embedding_dims: the dimesions of KvEmbedding variables.
    """

    def __init__(self):
        self.tensor_alloc_bytes = {}
        self.variable_count = 0
        self.total_variable_size = 0
        self.max_variable_size = 0
        self.kv_embedding_dims = []

    def update_varible_stats(self, variables):
        self.variable_count = len(variables)
        for var in variables:
            shape = var.get_shape().as_list()
            if hasattr(var, "key_dtype"):  # The unique attr for KV embedding
                self.kv_embedding_dims.append(int(shape[-1]))
            else:
                var_size = 1
                for dimesion in shape:
                    var_size *= dimesion
                self.total_variable_size += var_size
                self.max_variable_size = max(self.max_variable_size, var_size)


class ProfileExtracter(object):
    def __init__(self, trace):
        self._trace = trace
        self._step_stats = trace.analyze_step_stats(
            show_dataflow=True, show_memory=True
        )

        self.pid_devices = self.get_pid_process()

    def get_pid_process(self):
        """Get the device name of pid. The metadata in chrome_trace is like:
        [{'name': 'process_name',
        'ph': 'M',
        'pid': 2,
        'args': {'name': '/job:chief/replica:0/task:0/device:CPU:0 Tensors'}}]
        """
        pid_device = {}
        for event in self._step_stats.chrome_trace._metadata:
            if event["name"] == "process_name":
                device_name = event["args"]["name"]
                pid_device[event["pid"]] = device_name
            else:
                break
        return pid_device

    def get_tensor_alloc_bytes(self):
        tensor_alloc_bytes = {}
        for _, tensor in self._trace._tensors.items():
            device = self.pid_devices[tensor.pid]
            tensor_alloc_bytes.setdefault(device, 0)
            tensor_alloc_bytes[device] += tensor._num_bytes
        return tensor_alloc_bytes

    def get_chief_recv_op_count(self):
        recv_op_count = 0
        for event in self._step_stats.chrome_trace._events:
            device = self.pid_devices[event["pid"]]
            if "/job:chief" not in device:
                continue
            if "dur" in event:
                if "RecvTensor" in event["name"]:
                    recv_op_count += 1
        return recv_op_count

    def get_input_fetch_dur(self):
        for event in self._step_stats.chrome_trace._events:
            device = self.pid_devices[event["pid"]]
            if "/job:chief" not in device:
                continue
            if "dur" in event and "IteratorGetNext" in event["name"]:
                return event["dur"]
        return 0
