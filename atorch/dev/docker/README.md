# atorch image

To build docker images
```bash
# Build pytorch base image under dev/docker/base folder.
sudo docker build -f Dockerfile --net host -t "easydl/pytorch_gpu_base:2.0.1-cuda12.1-cudnn8-devel" .
sudo docker build -f Dockerfile-pt21 --net host -t "easydl/pytorch_gpu_base:2.1.0-cuda12.1-cudnn8-devel" .

# Build atorch image
sudo docker build -f dev/docker/Dockerfile-ubuntu2004-pt210 --net host -t "reg.docker.alibaba-inc.com/atorch/atorch-open:pt210" .
# To build base image, usually not needed, base Dockerfile is copied from pytorch repo for reference.
make -f docker.Makefile
```

We use Docker container for development. The Dockerfile can be found at dlrover/atorch/dev/docker/

```bash
# Pull Docker image based on pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel
sudo docker pull "registry.cn-hangzhou.aliyuncs.com/atorch/atorch-open-20240430:pt210"
# Run Docker container and mount source directory to /v
sudo docker run -it --rm --net=host --shm-size=1G -v ${PWD}:/v -w /v "registry.cn-hangzhou.aliyuncs.com/atorch/atorch-open-20240430:pt210" /bin/bash
```

For development, refer to following steps:
```bash
# build proto
sh dev/scripts/build_proto.sh

# run pre-commit
sh dev/scripts/pre-commit.sh

# run unittest
PYTHONPATH=#ATORCH_ROOT# pytest atorch/tests

# build atorch wheel
sh dev/scripts/build.sh
```
