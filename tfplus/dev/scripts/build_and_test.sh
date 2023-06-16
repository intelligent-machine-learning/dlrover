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
pytest tests

STATUS=$?

if [ ${STATUS} -ne 0 ]
then
    echo "============================== Hello  World ================================="
    echo "|                                                                            |"
    echo "| Please check above error message.                                          |"
    echo "| You can run sh dev/scripts/pre-commit.sh locally                               |"
    echo "|                                                                            |"
    echo "============================== Hello  World ================================="
    exit ${STATUS}
fi