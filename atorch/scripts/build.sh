set -e
python tools/scripts/render_setup.py --version 0.0.1
# python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. atorch/protos/coworker.proto

python setup.py bdist_wheel

rm -f atorch/protos/coworker_pb2*