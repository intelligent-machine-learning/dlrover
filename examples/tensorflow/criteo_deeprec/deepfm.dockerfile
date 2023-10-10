FROM easydl/easydl:ci as builder

WORKDIR /dlrover
COPY ./ .
RUN sh scripts/build_wheel.sh

FROM registry.cn-hangzhou.aliyuncs.com/dlrover_deeprec/deeprec:v11 as base

WORKDIR /dlrover
RUN apt-get update && apt-get install -y sudo
RUN pip install pyhocon pynvml

COPY --from=builder /dlrover/dist/dlrover-*.whl /
RUN pip install /*.whl --extra-index-url=https://pypi.org/simple && rm -f /*.whl

COPY examples/tensorflow examples/tensorflow