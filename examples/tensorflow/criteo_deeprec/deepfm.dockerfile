FROM registry.cn-hangzhou.aliyuncs.com/dlrover_deeprec/deeprec:v11 as base

WORKDIR /dlrover
RUN apt-get update && apt-get install -y sudo
RUN pip install pyhocon pynvml

RUN pip install dlrover[tensorflow]==0.2.1 -U

COPY examples/tensorflow examples/tensorflow