#!/bin/bash

set -e
set -x

PROTOBUF_VERSION_SRC=$(pip show protobuf | grep Version | cut -d' ' -f2)
GRPCIO_VERSION_SRC=$(pip show grpcio | grep Version | cut -d' ' -f2)
GRPCIO_TOOLS_VERSION_SRC=$(pip show grpcio-tools | grep Version | cut -d' ' -f2)
PROTOS_DIR="atorch/protos"

PROTOBUF_VERSION_2="3.20.3"
GRPCIO_VERSION_2="1.34.1"
GRPCIO_TOOLS_VERSION_2="1.34.1"
pip install protobuf==$PROTOBUF_VERSION_2 grpcio==$GRPCIO_VERSION_2 grpcio-tools==$GRPCIO_TOOLS_VERSION_2
cp $PROTOS_DIR/*.proto $PROTOS_DIR/protobuf_3_20_3/
pushd .
cd $PROTOS_DIR/protobuf_3_20_3
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. ./acceleration.proto
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. ./coworker.proto
sed -i 's/import acceleration_pb2/from \. import acceleration_pb2/g' acceleration_pb2_grpc.py
sed -i 's/import coworker_pb2/from \. import coworker_pb2/g' coworker_pb2_grpc.py
rm *.proto
popd

PROTOBUF_VERSION_3="4.25.3"
GRPCIO_VERSION_3="1.62.1"
GRPCIO_TOOLS_VERSION_3="1.58.0"
pip install protobuf==$PROTOBUF_VERSION_3 grpcio==$GRPCIO_VERSION_3 grpcio-tools==$GRPCIO_TOOLS_VERSION_3
cp $PROTOS_DIR/*.proto $PROTOS_DIR/protobuf_4_25_3/
pushd .
cd $PROTOS_DIR/protobuf_4_25_3
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. ./acceleration.proto
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. ./coworker.proto
sed -i 's/import acceleration_pb2/from \. import acceleration_pb2/g' acceleration_pb2_grpc.py
sed -i 's/import coworker_pb2/from \. import coworker_pb2/g' coworker_pb2_grpc.py
rm *.proto
popd


pip install protobuf==$PROTOBUF_VERSION_SRC grpcio==$GRPCIO_VERSION_SRC grpcio-tools==$GRPCIO_TOOLS_VERSION_SRC
