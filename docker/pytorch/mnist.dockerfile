FROM easydl/easydl:ci as builder

WORKDIR /dlrover
COPY ./ .
RUN sh scripts/build_wheel.sh

FROM easydl/dlrover-train:torch201-cpu-py38  as base

WORKDIR /dlrover

RUN apt-get install sudo
COPY ./data /data

COPY --from=builder /dlrover/dist/dlrover-*.whl /
RUN pip install /*.whl --extra-index-url=https://pypi.org/simple --no-deps && rm -f /*.whl

COPY ./model_zoo ./model_zoo
