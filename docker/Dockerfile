FROM easydl/easydl:ci as builder

WORKDIR /dlrover
COPY ./ .
RUN sh scripts/build_wheel.sh

FROM python:3.8.14 as base
RUN pip install pyparsing kubernetes ray -i https://pypi.org/simple

COPY --from=builder /dlrover/dist/dlrover-*.whl /
RUN pip install /*.whl --extra-index-url=https://pypi.org/simple && rm -f /*.whl