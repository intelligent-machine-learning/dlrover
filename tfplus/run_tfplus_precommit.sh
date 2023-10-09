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

pre-commit run -v --files $(find . -path ./build -prune -o \( -name "*.py" -not -name "*pb2.py" -not -name "copyright.py" -o -name "*.cc" -o -name "*.h" -o -name "*.hpp" \) -type f -print) -c .pre-commit-config.yaml