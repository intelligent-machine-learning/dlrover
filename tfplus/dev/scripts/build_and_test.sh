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

set -e
BASE_DIR=$(cd "$(dirname "$0")"; cd ../..; pwd)

# Configure will install TensorFlow or use existing one
"$BASE_DIR/configure.sh" "$@"

# Build TFPlus
bazel build -s --verbose_failures -- //tfplus/...
# Let all tests and checks run
set +e
# Build and run TFPlus C++ tests
bazel test //tfplus/...
# Build TFPlus package
cd "$BASE_DIR"
python setup.py bdist_wheel $1
pip install -U dist/*.whl --force-reinstall --no-deps

pytest --import-mode=importlib "$BASE_DIR/py_ut"

STATUS=$?

if [ ${STATUS} -ne 0 ]
then
    echo "============================== Hello  World ================================="
    echo "|                                                                            |"
    echo "| Please check the error message above.                                      |"
    echo "| You can re-run this script by 'sh tfplus/dev/scripts/build_and_test.sh'.  |"
    echo "|                                                                            |"
    echo "============================== Hello  World ================================="
    exit ${STATUS}
fi