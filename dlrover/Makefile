PYTHON = python
CXX = g++

all: python_pb go_pb

python_pb: dlrover/proto/elastic_training.proto dlrover/proto/brain.proto
	/bin/sh scripts/build_proto.sh

go_pb: dlrover/proto/brain.proto
	protoc -I. dlrover/proto/brain.proto --go_out=plugins=grpc:.
