#!/bin/bash
# Copyright 2024 The EasyDL Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -e
set -x
PROTOBUF_VERSION_SRC=$(pip show protobuf | grep Version | cut -d' ' -f2)
GRPCIO_VERSION_SRC=$(pip show grpcio | grep Version | cut -d' ' -f2)
GRPCIO_TOOLS_VERSION_SRC=$(pip show grpcio-tools | grep Version | cut -d' ' -f2)
PROTOS_DIR=dlrover/proto
generate_proto_files() {
  base_dir=$PWD
  local protodir="$1"

  dir="${PROTOS_DIR}"
  cd $dir
  mkdir -p "$protodir"
  proto_files=$(find . -maxdepth 1 -type f -name "*.proto")
  cp ./*.proto "$protodir"

  cd "$protodir"
  for fn in $proto_files; do
    filename=$(basename "$fn" .proto)
    python -m grpc_tools.protoc -I. -I"$base_dir" --python_out=. --grpc_python_out=. "$filename".proto
    sed -i "s/import ${filename}_pb2/from \. import ${filename}_pb2/g" "$filename"_pb2_grpc.py
  done
  rm -rf ./*.proto
  cd ${base_dir}
}
CUR_PYTHON_VERSION=$(python3 --version | awk -F " " '{print $NF}'| awk -F. '{print $1 $2}')

if [ "${CUR_PYTHON_VERSION}" = "38" ]; then
  PROTOBUF_VERSION_2="3.20.3"
  GRPCIO_VERSION_2="1.34.1"
  GRPCIO_TOOLS_VERSION_2="1.34.1"
  pip install protobuf==$PROTOBUF_VERSION_2 grpcio==$GRPCIO_VERSION_2 grpcio-tools==$GRPCIO_TOOLS_VERSION_2
  generate_proto_files protobuf_3_20_3
fi

PROTOBUF_VERSION_3="4.25.3"
GRPCIO_VERSION_3="1.62.1"
GRPCIO_TOOLS_VERSION_3="1.58.0"
pip install protobuf==$PROTOBUF_VERSION_3 grpcio==$GRPCIO_VERSION_3 grpcio-tools==$GRPCIO_TOOLS_VERSION_3
generate_proto_files protobuf_4_25_3
pip install protobuf=="$PROTOBUF_VERSION_SRC" grpcio=="$GRPCIO_VERSION_SRC" grpcio-tools=="$GRPCIO_TOOLS_VERSION_SRC"
