#!/bin/bash

set -e
set -x

python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. atorch/protos/acceleration.proto
