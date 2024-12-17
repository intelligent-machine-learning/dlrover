# Copyright 2024 The DLRover Authors. All rights reserved.
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

import argparse
import os
import shutil
import subprocess
import sys
import sysconfig
import textwrap
from abc import ABC, abstractmethod
from pathlib import Path

IS_ONLINE = os.environ.get("BUILD_ENV_ONLINE", "0") == "1"
IGNORE_GIT = os.environ.get("BUILD_IGNORE_GIT", "0") == "1"


def check_git_modifications():
    if IGNORE_GIT:
        return
    # Check for unstaged changes
    result = subprocess.run(["git", "diff", "--quiet"], stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise Exception("When build with release, git tree must be clean")

    # Check for staged but not committed changes
    result = subprocess.run(["git", "diff", "--cached", "--quiet"], stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise Exception("When build with release, git tree must be clean")


def pipe_commands(commands):
    if len(commands) == 1:
        return subprocess.Popen(commands[0], stdout=subprocess.PIPE).communicate()

    cur = None
    for command in commands:
        cur = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stdin=cur.stdout if cur is not None else None,
        )
    return [i.decode().strip() if i is not None else i for i in cur.communicate()]


def find_python_releated():
    from torch.utils import cpp_extension

    return {
        "python_lib": f'{sysconfig.get_config_var("LIBDIR")}/libpython{sysconfig.get_python_version()}.so',
        "python_include": sysconfig.get_config_var("INCLUDEPY"),
        "torch_path": str(Path(cpp_extension.TORCH_LIB_PATH).parent),
    }


class BaseBuildRender(ABC):
    def __init__(self, args):
        self.args = args
        if self.args.ssl_path is None:
            self.args.ssl_path = "/opt/conda"

        self.python_info = find_python_releated()

        base = textwrap.dedent(
            """
            build --cxxopt=-std=c++17
            build --cxxopt=-fvisibility=hidden
            build --copt=-fvisibility=hidden
            build --linkopt=-lstdc++fs
            build --jobs={parallel}
            """
        ).format(parallel=self.args.build_parallel)

        self.bazelrc_config = [base, f"build --repo_env=OPENSSL_HOME={self.args.ssl_path}"]

        release = textwrap.dedent(
            """
            build:release --compilation_mode=opt
            build:release --copt=-O2
            build:release --copt=-DNDEBUG
            build:release --cxxopt=-O2
            build:release --cxxopt=-DNDEBUG
            build:release --strip=always
            build --config=release
            build --workspace_status_command "bash version.sh release"
            """
        )
        debug = textwrap.dedent(
            """
            build:debug --compilation_mode=dbg
            build:debug --copt=-g
            build:debug --cxxopt=-g
            build:debug --copt=-DXPU_DEBUG
            build:debug --cxxopt=-DXPU_DEBUG
            build:debug --strip=never
            build --config=debug
            build --workspace_status_command "bash version.sh debug"
            """
        )
        if self.args.build_type == "release":
            check_git_modifications()
            self.bazelrc_config.append(release)
        elif self.args.build_type == "debug":
            self.bazelrc_config.append(debug)
        elif self.args.build_type == "test":
            self.bazelrc_config.append(release)

        self.xpu_timer_config = [
            textwrap.dedent(
                """
            TORCH_PATH = "{torch_path}"
            PYTHON_LIB = "{python_lib}"
            PYTHON_INCLUDE = "{python_include}"
        """
            ).format(**self.python_info)
        ]

    @abstractmethod
    def rend_config_bzl(self) -> None:
        ...

    @abstractmethod
    def rend_bazelrc(self) -> None:
        ...

    @abstractmethod
    def setup_files(self) -> None:
        ...

    @abstractmethod
    def setup_platform_version(self) -> None:
        ...

    @abstractmethod
    def __post_init__(self) -> None:
        ...

    def run(self):
        self.__post_init__()

        for k, v in vars(self.args).items():
            setattr(self, k, v)

        # platform_version
        platform_version, platform_type = self.setup_platform_version()
        with open(".platform_version", "w") as f:
            f.write(platform_version)
        with open(".build_platform", "w") as f:
            f.write(platform_type)

        # xpu_config.bzl
        path = Path("xpu_config/xpu_config.bzl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            f.write(self.rend_config_bzl())

        # bazelrc
        with open(".bazelrc", "w") as f:
            f.write(self.rend_bazelrc())

        # setup file
        with open("WORKSPACE", "w") as f:
            f.write(self.setup_files())

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument("--sdk-path", help="Path to the SDK", default=None)
        parser.add_argument("--ssl-path", help="Path to ssl", default=None)
        parser.add_argument("--build-parallel", type=str, default="16")
        parser.add_argument("--build-type", type=str, default="release", choices=["test", "release", "debug"])
        parser.add_argument("--cache", action="store_true")
