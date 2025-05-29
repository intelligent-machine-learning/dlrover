FROM easydl/dlrover:ci as builder

WORKDIR /dlrover
COPY ./ .
RUN sh scripts/build_wheel.sh

FROM python:3.8.14 as base
RUN pip install pyparsing -i https://pypi.org/simple

RUN apt-get -qq update && apt-get install -y iputils-ping vim gdb

# must equal with
#ENV VERSION="0.5.0.dev"
RUN VERSION=$(grep "version=" setup.py | sed -E "s/.*version=['\"]([^'\"]+)['\"].*/\1/") && \
    echo "Version found: ${VERSION}"

COPY --from=builder /dlrover/dist/dlrover-${VERSION}-py3-none-any.whl /
RUN pip install /dlrover-${VERSION}-py3-none-any.whl[k8s,ray] --extra-index-url=https://pypi.org/simple && rm -f /*.whl
RUN unset VERSION
