#!/bin/sh
# Copyright 2023 The TFPlus Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


TYPE=$1
[ -z ${TYPE} ] && TYPE=release

pip install pre-commit==2.21.0

if [ ${TYPE} != "precommit" ]
then

    sh build.sh ${TYPE}
    pushd dist
    file=`ls -al | grep ".whl" | awk '{print $NF}'`
    pip install "${file}[tfplus]" --upgrade
    popd

    \rm -rf build dist
    pip install pytest -I
    pip install pytest-xdist -I
    pip install coverage -I
    install sklearn
    # install_fix keras==2.0.6
    pushd ../../third_party
    pip install --force-reinstall -U tensorflow
    popd
fi
