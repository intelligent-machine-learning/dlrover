import fnmatch
import glob
import os
import subprocess
import sys

import setuptools.command.build_py
from setuptools import find_packages, setup

from atorch.ops.op_builder import get_default_compute_capabilities
from atorch.ops.op_builder.all_ops import ALL_OPS

torch_available = True
try:
    import torch
except ImportError:
    torch_available = False
    print(
        "[WARNING] Unable to import torch, pre-compiling ops will be disabled. "
        "Please visit https://pytorch.org/ to see how to properly install torch on your system."
    )


RED_START = "\033[31m"
RED_END = "\033[0m"
ERROR = f"{RED_START} [ERROR] {RED_END}"


class build_proto(setuptools.command.build_py.build_py):
    def run(self):
        try:
            self.spawn(["sh", "dev/scripts/build_proto.sh"])
        except RuntimeError as e:
            self.warn(f"build proto error:{e}")
        super().run()


def abort(msg):
    print(f"{ERROR} {msg}")
    assert False, msg


def fetch_requirements(path):
    with open(path, "r") as fd:
        return [r.strip() for r in fd.readlines()]


required_deps = fetch_requirements("atorch/requirements.txt")
in_req_path = "atorch/internal_requirements.txt"
internal_required_deps = [] if not os.path.exists(in_req_path) else fetch_requirements(in_req_path)
required_deps += internal_required_deps

cmdclass = {
    "build_py": build_proto,
}

# For any pre-installed ops force disable ninja.
if torch_available:
    from atorch.ops.accelerator import get_accelerator

    cmdclass["build_ext"] = get_accelerator().build_extension().with_options(use_ninja=False)

if torch_available:
    TORCH_MAJOR = torch.__version__.split(".")[0]
    TORCH_MINOR = torch.__version__.split(".")[1]
else:
    TORCH_MAJOR = "0"
    TORCH_MINOR = "0"

if torch_available and not torch.cuda.is_available():
    # Fix to allow docker builds, similar to https://github.com/NVIDIA/apex/issues/486.
    print(
        "[WARNING] Torch did not find cuda available, if cross-compiling or running with cpu only "
        "you can ignore this message. Adding compute capability for Pascal, Volta, and Turing "
        "(compute capabilities 6.0, 6.1, 6.2)"
    )
    if os.environ.get("TORCH_CUDA_ARCH_LIST", None) is None:
        os.environ["TORCH_CUDA_ARCH_LIST"] = get_default_compute_capabilities()

ext_modules = []

# Default to pre-install kernels to false so we rely on JIT on Linux, opposite on Windows.
BUILD_OP_PLATFORM = 1 if sys.platform == "win32" else 0
BUILD_OP_DEFAULT = int(os.environ.get("ATORCH_BUILD_OPS", BUILD_OP_PLATFORM))
print(f"ATORCH_BUILD_OPS={BUILD_OP_DEFAULT}")

if BUILD_OP_DEFAULT:
    assert (
        torch_available
    ), "Unable to pre-compile ops without torch installed. Please install torch before attempting to pre-compile ops."


def command_exists(cmd):
    if sys.platform == "win32":
        result = subprocess.Popen(f"{cmd}", stdout=subprocess.PIPE, shell=True)
        return result.wait() == 1
    else:
        result = subprocess.Popen(f"type {cmd}", stdout=subprocess.PIPE, shell=True)
        return result.wait() == 0


def op_envvar(op_name):
    assert hasattr(ALL_OPS[op_name], "BUILD_VAR"), f"{op_name} is missing BUILD_VAR field"
    return ALL_OPS[op_name].BUILD_VAR


def op_enabled(op_name):
    env_var = op_envvar(op_name)
    return int(os.environ.get(env_var, BUILD_OP_DEFAULT))


compatible_ops = dict.fromkeys(ALL_OPS.keys(), False)
install_ops = dict.fromkeys(ALL_OPS.keys(), False)
for op_name, builder in ALL_OPS.items():
    op_compatible = builder.is_compatible()
    compatible_ops[op_name] = op_compatible
    compatible_ops["atorch_not_implemented"] = False

    # If op is requested but not available, throw an error.
    if op_enabled(op_name) and not op_compatible:
        env_var = op_envvar(op_name)
        if env_var not in os.environ:
            builder.warning(f"One can disable {op_name} with {env_var}=0")
        abort(f"Unable to pre-compile {op_name}")

    # If op install enabled, add builder to extensions.
    if op_enabled(op_name) and op_compatible:
        assert torch_available, f"Unable to pre-compile {op_name}, please first install torch"
        install_ops[op_name] = op_enabled(op_name)
        ext_modules.append(builder.builder())

print(f"Install Ops={install_ops}")

# Write out version/git info.
git_hash_cmd = "git rev-parse --short HEAD"
git_branch_cmd = "git rev-parse --abbrev-ref HEAD"
if command_exists("git"):
    try:
        result = subprocess.check_output(git_hash_cmd, shell=True)
        git_hash = result.decode("utf-8").strip()
        result = subprocess.check_output(git_branch_cmd, shell=True)
        git_branch = result.decode("utf-8").strip()
    except subprocess.CalledProcessError:
        git_hash = "unknown"
        git_branch = "unknown"
else:
    git_hash = "unknown"
    git_branch = "unknown"

# Parse the  version string from version.txt.
version_str = "$version"

# Build specifiers like .devX can be added at install time. Otherwise, add the git hash.
# Building wheel for distribution, update version file.
# None of the above, probably installing from source.
version_str += f"+{git_hash}"

torch_version = ".".join([TORCH_MAJOR, TORCH_MINOR])
bf16_support = False
# Set cuda_version to 0.0 if cpu-only.
cuda_version = "0.0"
nccl_version = "0.0"

if torch_available and torch.version.cuda is not None:
    cuda_version = ".".join(torch.version.cuda.split(".")[:2])
    if sys.platform != "win32":
        if isinstance(torch.cuda.nccl.version(), int):
            # This will break if minor version > 9.
            nccl_version = ".".join(str(torch.cuda.nccl.version())[:2])
        else:
            nccl_version = ".".join(map(str, torch.cuda.nccl.version()[:2]))
    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_available():
        bf16_support = torch.cuda.is_bf16_supported()
torch_info = {
    "version": torch_version,
    "bf16_support": bf16_support,
    "cuda_version": cuda_version,
    "nccl_version": nccl_version,
}

print(f"version={version_str}, git_hash={git_hash}, git_branch={git_branch}")
with open("atorch/ops/git_version_info_installed.py", "w") as fd:
    fd.write(f"version = '{version_str}'\n")
    fd.write(f"git_hash = '{git_hash}'\n")
    fd.write(f"git_branch = '{git_branch}'\n")
    fd.write(f"installed_ops = {install_ops}\n")
    fd.write(f"compatible_ops = {compatible_ops}\n")
    fd.write(f"torch_info = {torch_info}\n")

print(f"install_requires={required_deps}")
print(f"compatible_ops={compatible_ops}")
print(f"ext_modules={ext_modules}")

# package dependencies of source files and binaries
package_data = []
filename_suffixes = ["*.cu", "*.cuh", "*.cc", "*.cpp", "*.h", ".so"]
data_path = os.path.abspath("./atorch/ops/csrc")
for rootname, _, filenames in os.walk(data_path):
    for filename in filenames:
        src = os.path.join(rootname, filename)
        for suffix in filename_suffixes:
            if fnmatch.filter([src], suffix):
                package_data.append(src)

proto_files = glob.glob("atorch/protos/*.proto")
setup(
    name="atorch",
    version="$version",  # render by script,do not modify
    description="A pytorch extension for efficient deep learning.",
    long_description="ATorch supports efficient and easy-to-use model training experience."
    " ATorch provides performance optimizations in aspects such as I/O, preprocessing,"
    " computation, and communication (including automatic optimization), and has supported"
    " large-scale pretraining and finetuning of LLMs with over 100 billion parameters and"
    " thousands of advanced GPUs.",
    author="Ant Group",
    url="https://github.com/intelligent-machine-learning/dlrover/tree/master/atorch",
    python_requires=">=3.8",
    packages=find_packages(exclude=["*test*", "benchmarks*"]),
    install_requires=required_deps,
    package_data={"": ["*.so", "_fa_api_compat_patch"], "atorch": package_data},
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    data_files=["atorch/requirements.txt", "dev/scripts/build_proto.sh"] + proto_files,
)
