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
PROTOBUF_VERSION_SRC=$(pip3 show protobuf | grep Version | cut -d' ' -f2)
GRPCIO_VERSION_SRC=$(pip3 show grpcio | grep Version | cut -d' ' -f2)
GRPCIO_TOOLS_VERSION_SRC=$(pip3 show grpcio-tools | grep Version | cut -d' ' -f2)
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
    if command -v protoc-gen-pyi
    then
        python3 -m grpc_tools.protoc -I. -I"$base_dir" --python_out=. --pyi_out=. --grpc_python_out=. "$filename".proto
    else
        python3 -m grpc_tools.protoc -I. -I"$base_dir" --python_out=. --grpc_python_out=. "$filename".proto
    fi
    sed -i "s/import ${filename}_pb2/from \. import ${filename}_pb2/g" "$filename"_pb2_grpc.py
  done
  rm -rf ./*.proto
  cd "$base_dir"
}
CUR_PYTHON_VERSION=$(python3 --version | awk -F " " '{print $NF}'| awk -F. '{print $1 $2}')

if [ "$(printf '%02d\n' "${CUR_PYTHON_VERSION}")" -le 36 ]; then
  PROTOBUF_VERSION="3.13.0"
  GRPCIO_VERSION="1.29.0"
  GRPCIO_TOOLS_VERSION="1.29.0"
  pip3 install protobuf==$PROTOBUF_VERSION grpcio==$GRPCIO_VERSION grpcio-tools==$GRPCIO_TOOLS_VERSION
  generate_proto_files protobuf_3_20_3
elif [ "$(printf '%02d\n' "${CUR_PYTHON_VERSION}")" -ge 38 ]; then
  PROTOBUF_VERSION="4.25.3"
  GRPCIO_VERSION="1.62.1"
  GRPCIO_TOOLS_VERSION="1.58.0"
  pip3 install protobuf==$PROTOBUF_VERSION grpcio==$GRPCIO_VERSION grpcio-tools==$GRPCIO_TOOLS_VERSION
  generate_proto_files protobuf_4_25_3
  pip3 install protobuf=="$PROTOBUF_VERSION_SRC" grpcio=="$GRPCIO_VERSION_SRC" grpcio-tools=="$GRPCIO_TOOLS_VERSION_SRC"
fi
