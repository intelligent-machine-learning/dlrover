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


from dlrover.python.unified.api.builder.rl import RLJobBuilder

classes = f"{__package__}.classes"


def test_api(tmp_ray):
    builder = RLJobBuilder().device_per_node(8).device_type("CPU")
    builder.actor(f"{classes}.Actor").total(2).per_group(2).resource(
        accelerator=0.5
    )
    builder.critic(f"{classes}.Critic")
    builder.rollout(f"{classes}.Rollout").total(2).resource(accelerator=0.5)
    builder.reference(f"{classes}.Reference")
    builder.reward(f"{classes}.Reward")
    builder.trainer(f"{classes}.Trainer").resource(cpu=1)

    rl_job = builder.build()
    ret = rl_job.submit("openrlhf_demo", master_cpu=1, master_memory=128)
    assert ret == 0
