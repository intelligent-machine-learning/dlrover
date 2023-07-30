FROM easydl/easydl:ci as builder

WORKDIR /dlrover
COPY ./ .
RUN sh scripts/build_wheel.sh

FROM nvidia/cuda:10.2-runtime-ubuntu18.04 as base

WORKDIR /dlrover

COPY --from=builder /dlrover/dist/dlrover-*.whl /

RUN sed -i '1s/deb /deb [trusted=yes] /' /etc/apt/sources.list.d/cuda.list && \
    sed -i '1s/deb /deb [trusted=yes] /' /etc/apt/sources.list.d/nvidia-ml.list

# Install Python 3.8 and pip
RUN apt-get update && apt-get install -y \
    python3.8 python3-pip python3.8-venv python3.8-dev wget vim curl && \
    ln -s /usr/bin/python3.8 /usr/bin/python && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py && rm get-pip.py && \
    pip install --upgrade pip

RUN pip install urllib3==1.21.1 grpcio==1.34.1 grpcio-tools==1.34.1 protobuf==3.20.3

RUN pip install torch==2.0.1+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 \
    -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install /*.whl --extra-index-url=https://pypi.org/simple --no-deps && \
    rm -f /*.whl