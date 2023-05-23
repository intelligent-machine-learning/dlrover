#!/bin/sh
tag=$1
if [ -z $tag ]
then
   read -p "image tag is NULL, please input: " tag
fi
sudo docker build -f tools/docker/Dockerfile-aci --net host -t "reg.docker.alibaba-inc.com/atorch/aci-base:$tag" ./
