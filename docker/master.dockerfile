FROM easydl/dlrover:ci as builder

WORKDIR /dlrover
COPY ./ .
RUN sh scripts/build_wheel.sh

FROM python:3.8.14 as base
RUN pip install pyparsing -i https://pypi.org/simple

RUN apt-get -qq update && apt-get install -y iputils-ping vim gdb

ENV VERSION="0.3.6rc0"
COPY --from=builder /dlrover/dist/dlrover-${VERSION}-py3-none-any.whl /
RUN pip install /dlrover-${VERSION}-py3-none-any.whl[k8s] --extra-index-url=https://pypi.org/simple && rm -f /*.whl
RUN unset VERSION