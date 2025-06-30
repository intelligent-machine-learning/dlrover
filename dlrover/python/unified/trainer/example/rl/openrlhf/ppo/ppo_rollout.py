# Copyright 2025 The DLRover Authors. All rights reserved.
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
#
# This package includes code from [https://github.com/OpenRLHF/OpenRLHF]
# licensed under the Apache License 2.0. See [https://github.com/OpenRLHF/
# OpenRLHF] for details.

import os
import queue
from collections import defaultdict

import ray
from dlrover.python.unified.trainer.default.openrlhf.ppo.ppo_base import (
    BasePPORole,
)
from vllm import LLM


@ray.remote
class RolloutRayActor(BasePPORole):
    def __init__(self, master_handle, config):
        super().__init__(master_handle, config)

        self.num_actors = 0
        self.actor_counter = 0
        self.requests = {}
        self.response_queues = defaultdict(queue.Queue)
        self.llm = None

    def init(self, *args, **kwargs):
        os.environ["VLLM_RAY_BUNDLE_INDICES"] = str(self.local_rank)
        os.environ["VLLM_RAY_PER_WORKER_GPUS"] = str(kwargs.pop("num_gpus"))

        # Number of actors that will send prompt to this engine
        self.num_actors = kwargs.pop("num_actors")
        self.actor_counter = 0
        self.requests = {}
        self.response_queues = defaultdict(queue.Queue)
        self.llm = LLM(*args, **kwargs)

        if kwargs.pop("enable_sleep_mode"):
            self.sleep()

    def init_process_group(
        self,
        master_address,
        master_port,
        rank_offset,
        world_size,
        group_name,
        backend,
        use_ray,
    ):
        return self.llm.collective_rpc(
            "init_process_group",
            args=(
                master_address,
                master_port,
                rank_offset,
                world_size,
                group_name,
                backend,
                use_ray,
            ),
        )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        return self.llm.collective_rpc(
            "update_weight", args=(name, dtype, shape, empty_cache)
        )

    def update_weight_cuda_ipc(
        self, name, dtype, shape, ipc_handles, empty_cache=False
    ):
        return self.llm.collective_rpc(
            "update_weight_cuda_ipc",
            args=(name, dtype, shape, ipc_handles, empty_cache),
        )

    def reset_prefix_cache(self):
        self.llm.llm_engine.reset_prefix_cache()

    def sleep(self, level=1):
        self.llm.sleep(level=level)

    def wake_up(self):
        self.llm.wake_up()

    def add_requests(self, actor_rank, *, sampling_params, prompt_token_ids):
        """
        Save the requests from actors and generate responses when all actors
        have sent their requests
        """
        self.requests[actor_rank] = prompt_token_ids
        self.actor_counter += 1
        if self.actor_counter == self.num_actors:
            assert len(self.requests) == self.num_actors
            num_requests = []
            requests = []
            for actor_rank, request in self.requests.items():
                num_requests.append((actor_rank, len(request)))
                requests.extend(request)

            if len(requests) > 0:
                # For now we assume that all requests have the same sampling
                # params
                responses = self.llm.generate(
                    sampling_params=sampling_params, prompt_token_ids=requests
                )
            else:
                responses = []

            offset = 0
            self.responses = {}
            for actor_rank, num in num_requests:
                self.response_queues[actor_rank].put(
                    responses[offset : offset + num]
                )
                offset += num

            self.actor_counter = 0
            self.requests = {}

    def get_responses(self, actor_rank):
        """
        Return the responses for the actor with the given rank
        """
        return self.response_queues[actor_rank].get()
