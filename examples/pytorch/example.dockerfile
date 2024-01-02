# Build the image of examples using the dockerfile.
# We need to make the data directory data/ with the dataset of
# nanogpt, mnist and llama2 before building the image.
# The details to prepare data are in the `README` of examples.

FROM easydl/dlrover:ci as builder

WORKDIR /dlrover
COPY ./ .
RUN sh scripts/build_wheel.sh

FROM python:3.8.14 as base

WORKDIR /dlrover
RUN apt-get update && apt-get install -y sudo vim libgl1-mesa-glx libglib2.0-dev
RUN pip install deprecated pyparsing torch==2.0.1 opencv-python==4.7.0.72 \
torchvision==0.15.2 transformers deepspeed

COPY ./data /data
COPY ./examples ./examples

COPY --from=builder /dlrover/dist/dlrover-*.whl /
RUN pip install /*.whl --extra-index-url=https://pypi.org/simple && rm -f /*.whl
