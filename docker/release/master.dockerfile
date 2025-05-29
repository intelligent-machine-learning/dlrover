FROM easydl/dlrover:ci AS builder

ARG PY_VERSION=3.8.14
ARG VERSION
ENV VERSION=${VERSION}

WORKDIR /dlrover
COPY ./ .
RUN sh scripts/build_wheel.sh

FROM python:${PY_VERSION} AS base

ARG VERSION
RUN pip install pyparsing -i https://pypi.org/simple
RUN apt-get -qq update && apt-get install -y iputils-ping vim gdb

COPY --from=builder /dlrover/dist/dlrover-${VERSION}-py3-none-any.whl /
RUN pip install /dlrover-${VERSION}-py3-none-any.whl[k8s,ray] --extra-index-url=https://pypi.org/simple && rm -f /*.whl
RUN unset VERSION
