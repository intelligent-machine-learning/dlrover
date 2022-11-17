PYTHON = python
CXX = g++

all: python_pb go_pb

python_pb: dlrover/proto/elastic_training.proto
	$(PYTHON) -m grpc_tools.protoc -I. \
	    dlrover/proto/elastic_training.proto --python_out=. --grpc_python_out=.
	$(PYTHON) -m grpc_tools.protoc -I. \
	    dlrover/proto/brain.proto --python_out=. --grpc_python_out=.

go_pb: dlrover/proto/brain.proto
	protoc -I. dlrover/proto/brain.proto --go_out=plugins=grpc:.