FROM easydl/dlrover:ci as builder

WORKDIR /dlrover
COPY ./ .
RUN sh scripts/build_wheel.sh

FROM registry.cn-hangzhou.aliyuncs.com/intell-ai/dlrover:pytorch-example  as base

WORKDIR /dlrover

RUN apt-get update && apt-get install -y sudo iproute2

RUN pip install py-spy

# Install chaosblade to test the fualt-node and straggler.
# https://github.com/chaosblade-io/chaosblade/releases/download/v1.7.2/chaosblade-1.7.2-linux-amd64.tar.gz
# resource can be replaced for faster downlading by: http://rayoltest.oss-cn-hangzhou-zmf.aliyuncs.com/tianyi/software/chaosblade/chaosblade-1.7.2-linux-amd64.tar.gz
RUN wget http://rayoltest.oss-cn-hangzhou-zmf.aliyuncs.com/tianyi/software/chaosblade/chaosblade-1.7.2-linux-amd64.tar.gz && \
    tar -zxvf chaosblade-1.7.2-linux-amd64.tar.gz && \
    rm chaosblade-1.7.2-linux-amd64.tar.gz && \
    chmod +x chaosblade-1.7.2/blade

COPY --from=builder /dlrover/dist/dlrover-*.whl /
RUN pip install /*.whl --extra-index-url=https://pypi.org/simple && rm -f /*.whl

COPY ./examples ./examples
