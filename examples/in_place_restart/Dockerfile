# Copyright 2026 The DLRover Authors. All rights reserved.
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

# NGC 26.09 contains PyTorch 2.14.0a0 and NCCL 2.31.  That NCCL release is
# sufficient for the nccl2 reconfigure API, which requires NCCL 2.28 or newer.
#
# To test with NCCL 2.32 instead, rebuild with:
#   --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:26.10-py3
# NGC 26.10 uses PyTorch 2.15.0a0, so rerun the recovery test matrix first.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.09-py3
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_CONSTRAINT=/etc/pip/constraint.txt \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/dlrover \
    TORCH_NCCL_ASYNC_ERROR_HANDLING=0 \
    TORCH_NCCL_BLOCKING_WAIT=0

WORKDIR /workspace/dlrover

# Do not install examples/in_place_restart/requirements.txt here.  The NGC
# image contains a purpose-built PyTorch build which must not be overwritten
# by a generic PyPI wheel.
COPY LICENSE README.md setup.py ./
COPY dlrover ./dlrover
COPY examples/in_place_restart ./examples/in_place_restart

RUN python -m pip install ".[k8s]" \
    && python -c "import torch, torch.distributed as dist; \
assert (int(torch.__version__.split('.')[0]), \
int(torch.__version__.split('.')[1])) >= (2, 14); \
assert dist.is_backend_available('nccl2'); \
assert hasattr(dist, '_reconfigure'); \
assert hasattr(dist, '_get_reconfigure_handle'); \
assert torch.cuda.nccl.version() >= (2, 28, 0); \
print('PyTorch', torch.__version__); \
print('NCCL', torch.cuda.nccl.version())"

CMD ["python", "examples/in_place_restart/dp4/run_in_place_demo.py"]
