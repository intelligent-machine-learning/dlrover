FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

RUN apt-get update && apt-get install -y git wget gdb curl vim && \
    apt-get clean


RUN mkdir -p /tmp && cd /tmp && git clone https://github.com/NVIDIA/nccl.git -b v2.21.5-1 --depth=1 && \
    cd nccl && CUDARTLIB=cudart BUILDDIR=build-shared make -j && \
    NCCL_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/nccl/lib/libnccl.so.2 && \
    if [[ -f ${NCCL_PATH} ]]; then rm -rf ${NCCL_PATH}; fi && \
    cp build-shared/lib/libnccl.so.2.21.5 ${NCCL_PATH} && \
    cd /tmp && rm -rf /tmp/nccl
    

RUN wget -q https://github.com/bazelbuild/bazelisk/releases/download/v1.19.0/bazelisk-linux-amd64 -O /bin/bazelisk && \
    chmod +x /bin/bazelisk && \
    pip install protobuf

WORKDIR /build

COPY . /build

RUN cd /build && bash build.sh nvidia && pip install /build/dist_bin/py_xpu_timer-1.1+cu124-cp311-cp311-linux_x86_64.whl && \
    pip install -r /build/demo/train_fsdp/requirements.txt

