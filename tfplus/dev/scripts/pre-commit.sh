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



DIR=`dirname $0`
sh ${DIR}/prepare.sh precommit
git config --global --add safe.directory '*'

[ -e build ] && rm -rf build

Config=.pre-commit-config.yaml$1

echo "Precommit run without deploy folder"

pre-commit run -v --files $(find . -path ./deploy -prune -o -name "*.py" -print0 | tr '\0' ' ') -c ${Config}

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