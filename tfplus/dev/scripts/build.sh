#!/bin/bash
# Copyright 2023 The addon Authors. All Rights Reserved.
set -e
# Configure will install TensoFlow or use existing one

# configure.sh will be open when tensorflow plus is introduced.
# bash dev/scrits/configure.sh

bazel build -s --verbose_failures //tfplus/...
bazel test //tfplus/...
rm -fr dist/
python setup.py bdist_wheel
pip install -U dist/*.whl