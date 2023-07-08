# Copyright 2023 The DLRover Authors. All rights reserved.
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

from setuptools import find_packages, setup

install_requires = [
    "grpcio-tools==1.34.1",
    "protobuf>=3.15.3,<4.0dev",
    "psutil",
    "urllib3<1.27,>=1.21.1",
]


extra_require = {
    "master": ["kubernetes", "ray"],
    "tensorflow": ["tensorflow"],
    "torch": ["torch"],
}


setup(
    name="dlrover",
    version="0.3.0rc1",
    description="An Automatic Distributed Deep Learning Framework",
    long_description="DLRover helps model developers focus on model algorithm"
    " itself, without taking care of any engineering stuff,"
    " say, hardware acceleration, distribute running, etc."
    " It provides static and dynamic nodes' configuration automatically,"
    ", before and during a model training job running on k8s",
    long_description_content_type="text/markdown",
    author="Ant Group",
    url="https://github.com/intelligent-machine-learning/dlrover",
    install_requires=install_requires,
    extras_require=extra_require,
    python_requires=">=3.5",
    packages=find_packages(
        exclude=[
            "model_zoo*",
        ]
    ),
    package_data={
        "": [
            "proto/*",
            "docker/*",
            "Makefile",
        ]
    },
    entry_points={
        "console_scripts": ["dlrover-run=dlrover.trainer.torch.main:main"]
    },
)
