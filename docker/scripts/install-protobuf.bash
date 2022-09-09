#!/bin/bash

set -e

wget -q https://github.com/protocolbuffers/protobuf/releases/download/v3.7.1/protoc-3.7.1-linux-x86_64.zip
unzip -qq protoc-3.7.1-linux-x86_64.zip -d /usr/local
rm protoc-3.7.1-linux-x86_64.zip
