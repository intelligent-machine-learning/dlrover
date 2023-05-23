#!/bin/sh
tag=$1
dockerfile=$2

if [ -z $tag ]
then
   read -p "image tag is NULL, please input: " tag
fi

if [ -z $dockerfile ]
then
   read -p "dockerfile path is empty, please input: " dockerfile
fi

sudo docker build -f $dockerfile --net host -t "reg.docker.alibaba-inc.com/atorch/atorch-dev:$tag" ./
