# atorch image

To build docker images
```bash
# Build atorch image
sudo docker build -f dev/docker/Dockerfile-dev-pt2 --net host -t "easydl/atorch:iml" .
# To build base image, usually not needed, base Dockerfile is copied from pytorch repo for reference.
make -f docker.Makefile
```

We use Docker container for development. The Dockerfile can be found at dlrover/atorch/dev/docker/

```bash
# Pull Docker image based on pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel
sudo docker pull easydl/atorch:iml
# Run Docker container and mount source directory to /v
sudo docker run -it --rm --net=host -v ${PWD}:/v -w /v easydl/atorch:iml /bin/bash
```

For development, refer to following steps:
```bash
# run pre-commit
sh dev/scripts/pre-commit.sh

# run unittest
PYTHONPATH=#ATORCH_ROOT# pytest atorch/tests

# build atorch wheel
sh dev/scripts/build.sh
```
