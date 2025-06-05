FROM reg.docker.alibaba-inc.com/antcnstack/dlrover:ci as builder

WORKDIR /dlrover
COPY ./ .
RUN pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
RUN sh scripts/build_wheel.sh

FROM reg.docker.alibaba-inc.com/antcnstack/python:3.8.14 as base

RUN pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

RUN pip install pyparsing

RUN apt-get -qq update && apt-get install -y iputils-ping vim gdb

ENV VERSION="0.5.0rc0"
COPY --from=builder /dlrover/dist/dlrover-${VERSION}-py3-none-any.whl /
RUN pip install /dlrover-${VERSION}-py3-none-any.whl[k8s,master] && rm -f /*.whl
RUN unset VERSION