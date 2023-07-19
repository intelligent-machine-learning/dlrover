FROM easydl/easydl:ci as builder

WORKDIR /dlrover
COPY ./ .
RUN sh scripts/build_wheel.sh

FROM python:3.8.14 as base
RUN apt update
RUN apt install -y libgl1-mesa-glx libglib2.0-dev vim
RUN pip install deprecated pyparsing -i https://pypi.org/simple
RUN pip install torch opencv-python torchvision

COPY --from=builder /dlrover/dist/dlrover-*.whl /
RUN pip install /*.whl --extra-index-url=https://pypi.org/simple && rm -f /*.whl
