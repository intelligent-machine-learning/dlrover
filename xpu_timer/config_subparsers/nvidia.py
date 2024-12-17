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

import textwrap

from . import BaseBuildRender


class BuildRender(BaseBuildRender):
    @classmethod
    def add_arguments(cls, parser):
        BaseBuildRender.add_arguments(parser)

    def __post_init__(self):
        if self.args.sdk_path is None:
            self.args.sdk_path = "/usr/local/cuda"

        delete_packate = ",".join(f"//xpu_timer/{non_target}/..." for non_target in self.args.non_target)
        self.bazelrc_config.append(f"build --deleted_packages={delete_packate}")

    def rend_config_bzl(self):
        nvidia_config = textwrap.dedent(
            """
        XPU_TIMER_CONFIG = struct(
            linkopt = [
                "-Wl,--version-script=$(location //xpu_timer/nvidia:only_keep_nv.lds)",
                "-L{cuda_path}/lib64",
                "-lcudart",
            ],
            copt = [
                "-DXPU_NVIDIA",
            ],
            deps = ["@cuda//:cuda_headers", "@nccl_multi_version//:nccl_h", "//xpu_timer/nvidia:only_keep_nv.lds"],
            py_bin = [
                "//third_party/matmul_benchmark:nv_cublaslt_gemm",
                "//third_party/matmul_benchmark:nv_cublas_benchmark",
                "//xpu_timer/nvidia:intercepted.sym.default",
                "//xpu_timer/nvidia:libparse_params.so",
            ],
            gen_symbol = ["//xpu_timer/nvidia:gen_nvidia_symbols.py"],
            timer_deps = ["//xpu_timer/nvidia:nvidia_timer"],
            hook_deps = ["//xpu_timer/nvidia:nvidia_hook"],
        )

        """
        )

        self.xpu_timer_config.append(nvidia_config.format(cuda_path=self.sdk_path))
        return "\n".join(self.xpu_timer_config)

    def rend_bazelrc(self):
        self.bazelrc_config.append(f"build --repo_env=CUDA_HOME={self.sdk_path}")
        return "\n".join(self.bazelrc_config)

    def setup_files(self):
        with open("WORKSPACE.template") as f:
            workspace = f.read()

        deps = textwrap.dedent(
            """
            load("//third_party/cuda:cuda_workspace.bzl", "cuda_workspace")
            cuda_workspace()

            load("//third_party/nccl_multi_version:nccl_multi_version_workspace.bzl", "nccl_multi_version_workspace")
            nccl_multi_version_workspace()

            load("//third_party/matmul_benchmark:matmul_benchmark_workspace.bzl", "matmul_benchmark_workspace")
            matmul_benchmark_workspace()
            """
        )
        return workspace + deps

    def setup_platform_version(self):
        # /usr/local/cuda/include/cuda_runtime_api.h:139:#define CUDART_VERSION  12010
        version = None
        path = f"{self.sdk_path}/include/cuda_runtime_api.h"
        pattern = "#define CUDART_VERSION"
        with open(path) as f:
            for line in f:
                if line.startswith(pattern):
                    version = line.split(pattern)[-1]
                    break
        if version is None:
            raise ValueError("Cannot found version")

        version = int(version)
        major = version // 1000
        minor = (version % 1000) // 10
        return f"cu{major}{minor}", "NVIDIA"
