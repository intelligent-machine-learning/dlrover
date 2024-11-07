import os
from typing import List, Union

import torch_npu
from torch.library import Library

import atorch
from atorch.ops.op_builder.builder import OpBuilder

ATORCH_NPU_LIBRARY = Library("atorch_npu", "DEF")


class NPUOpBuilder(OpBuilder):
    def absolute_name(self):
        return f"atorch.ops.npu.{self.name}_op"

    def register_op_proto(self, op_proto: Union[str, List[str]]):
        if isinstance(op_proto, str):
            op_proto = [op_proto]
        for proto in op_proto:
            ATORCH_NPU_LIBRARY.define(proto)

    def atorch_src_path(self, code_path):
        if os.path.isabs(code_path):
            return code_path
        else:
            atorch_path = os.path.abspath(os.path.dirname(atorch.__file__))
            return os.path.join(atorch_path, code_path)

    @property
    def cann_path(self):
        ASCEND_HOME_PATH = "ASCEND_HOME_PATH"
        if ASCEND_HOME_PATH in os.environ and os.path.exists(os.environ[ASCEND_HOME_PATH]):
            return os.environ[ASCEND_HOME_PATH]
        return None

    @property
    def torch_npu_path(self):
        return os.path.dirname(os.path.abspath(torch_npu.__file__))

    def include_paths(self):
        paths = [
            os.path.join(self.cann_path, "include"),
            os.path.join(self.torch_npu_path, "include"),
            os.path.join(self.torch_npu_path, "include/third_party/acl/inc"),
        ]
        return paths

    def extra_ldflags(self):
        flags = [
            "-L" + os.path.join(self.cann_path, "lib64"),
            "-lascendcl",
            "-L" + os.path.join(self.torch_npu_path, "lib"),
            "-ltorch_npu",
        ]
        return flags

    def cxx_args(self):
        args = [
            "-fstack-protector-all",
            "-Wl,-z,relro,-z,now,-z,noexecstack",
            "-fPIC",
            "-pie",
            "-Wl,--disable-new-dtags,--rpath",
            "-s",
        ]
        return args
