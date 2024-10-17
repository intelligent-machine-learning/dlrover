import torch

from .npu_builder import NPUOpBuilder


class GMMOpBuilder(NPUOpBuilder):
    OP_NAME = "grouped_matmul"
    OP_PROTO = (
        "npu_gmm.List(Tensor x, Tensor weight, *, Tensor? bias, int[]? group_list, int? group_type) -> Tensor",
        "npu_gmm.Tensor(Tensor x, Tensor weight, *, Tensor? bias, Tensor? group_list, int? group_type) -> Tensor",
    )
    TORCH_MAJOR, TORCH_MINOR = map(int, torch.__version__.split(".")[:2])

    def __init__(self):
        super(GMMOpBuilder, self).__init__(self.OP_NAME)
        self.register_op_proto(self.OP_PROTO)

    def sources(self):
        return ["npu/csrc/cann/gmm.cpp"]

    def include_paths(self):
        paths = super().include_paths()
        paths += ["npu/csrc/inc"]
        return paths

    def cxx_args(self):
        args = super().cxx_args()
        args += [
            "-Wno-sign-compare",
            "-Wno-deprecated-declarations",
            "-Wno-return-type",
            "-D__FILENAME__='\"$$(notdir $$(abspath $$<))\"'",
        ]
        if self.TORCH_MAJOR >= 2 and self.TORCH_MINOR >= 1:
            cpp_std = " -std=c++17"
        else:
            cpp_std = " -std=c++14"
        args.append(cpp_std)
        return args
