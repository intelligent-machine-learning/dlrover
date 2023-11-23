FROM easydl/dlrover:ci as builder

WORKDIR /dlrover
COPY ./ .
RUN sh scripts/build_wheel.sh

FROM registry.cn-hangzhou.aliyuncs.com/easydl/dlrover-train:torch201-py38  as base

WORKDIR /dlrover

RUN apt-get update && apt-get install -y sudo
RUN pip install -r ./examples/pytorch/llama2/requirements.txt

COPY --from=builder /dlrover/dist/dlrover-*.whl /
RUN pip install /*.whl --extra-index-url=https://pypi.org/simple && rm -f /*.whl

COPY ./examples/pytorch/llama2 ./examples/pytorch/llama2
