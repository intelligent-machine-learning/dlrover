FROM easydl/dlrover:ci as builder

WORKDIR /dlrover
COPY ./ .
RUN sh scripts/build_wheel.sh

FROM easydl/dlrover-train:torch201-cpu-py38  as base

WORKDIR /dlrover

RUN apt-get update && apt-get install -y sudo iproute2

RUN pip install py-spy

# Install chaosblade to test the fualt-node and straggler.
# Download `wget https://github.com/chaosblade-io/chaosblade/releases/download/v1.7.2/chaosblade-1.7.2-linux-amd64.tar.gz`
COPY chaosblade-1.7.2-linux-amd64.tar.gz ./chaosblade-1.7.2-linux-amd64.tar.gz
RUN tar -zxvf chaosblade-1.7.2-linux-amd64.tar.gz 
RUN rm chaosblade-1.7.2-linux-amd64.tar.gz 
RUN chmod +x chaosblade-1.7.2/blade

COPY --from=builder /dlrover/dist/dlrover-*.whl /
RUN pip install /*.whl --extra-index-url=https://pypi.org/simple && rm -f /*.whl

COPY ./examples ./examples
