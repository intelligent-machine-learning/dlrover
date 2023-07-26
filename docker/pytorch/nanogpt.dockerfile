FROM easydl/easydl:ci as builder

WORKDIR /dlrover
COPY ./ .
RUN sh scripts/build_wheel.sh

FROM registry.cn-hangzhou.aliyuncs.com/easydl/dlrover-train:torch201-py38  as base

WORKDIR /dlrover

RUN apt-get install sudo
COPY ./data/nanogpt /data/nanogpt

COPY --from=builder /dlrover/dist/dlrover-*.whl /
RUN pip install /*.whl --extra-index-url=https://pypi.org/simple --no-deps && rm -f /*.whl

COPY ./model_zoo/pytorch/nanogpt ./model_zoo/pytorch/nanogpt
