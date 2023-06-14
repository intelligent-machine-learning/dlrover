## Developing

We use Docker container for development. The Dockerfile can be found [here](dev/Dockerfile):

Pull docker
```bash
# Log in docker
sudo docker login --username=xxx
# Pull from remote
sudo docker pull easydl/tfplus:[tag]
# Run Docker container and mount source directory to /v
docker run -it --rm --net=host -v ${PWD}:/v -w /v easydl/tfplus:[tag] /bin/bash
# Build and run unit tests and run c++, python code style check
sh build.sh
```

To build docker image locally and upload to remote
```bash
# Build Docker image
# i.e. sudo docker build -f dev/docker/Dockerfile.cpu -t easydl/tfplus:tf212_dev .
sudo docker build -f dev/docker/Dockerfile.cpu -t easydl/tfplus:[tag] .
# push to remote
sudo docker login --username=xxx
sudo docker tag [ImageId] easydl/tfplus:[tag]
sudo docker push easydl/tfplus:[tag]
# Run Docker container and mount source directory to /v
docker run -it --rm --net=host -v ${PWD}:/v -w /v nash635/tfplus:[tag]

For development, refer to following steps:
# Configure will install TensoFlow or use existing one. (open here when tfplus added.)
# ./scripts/configure.sh
# Build tfplus
bazel build -s --verbose_failures //tfplus/...
# Build and run tfplus C++ tests
bazel test //tfplus/...
# Build and run tfplus C++ tests using sanitizers (optional)
# bazel test -c dbg --config=asan //tfplus/...
# Run python style checks
find . -name '*.py' | xargs pylint --rcfile=.pylint
# Run c++ style checks
cpplint --recursive tfplus
# Build tfplus package
python setup.py bdist_wheel
# Run python tests, must install tfplus first
pip install -U dist/*.whl
pytest tests

# check pre-commit
sh dev/scripts/pre-commit.sh
```

A package file `dist/tfplus-*.whl` will be generated. It can be installed in local docker and uploaded to oss.

To install the package file in docker, please use command:
```bash
pip install -U dist/tfplus-*.whl
```

Additionally, users can specify the download link of Bazel by passing argument `--build-arg BAZEL_LINK_BASE=my_bazel_link_base` to docker build command. For example, assuming that `my_bazel_link_base` is `http://xxx/common/bazel`, we can enable to use this custom link as follows:
```bash
# Build Docker image using custom Bazel download link
docker build -f dev/docker/Dockerfile.cpu -t easydl/tfplus[tag] --build-arg BAZEL_LINK_BASE=*** .
```
