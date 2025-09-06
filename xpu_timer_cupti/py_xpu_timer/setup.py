import os
import shutil
from glob import glob

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install

LITE_MODE = os.environ.get("LITE_MODE", "0") == "1"

# TODO(zhangji.zhang) compatible with perfetto sdk
# https://commondatastorage.googleapis.com/perfetto-luci-artifacts/v47.0/linux-amd64/trace_processor_shell
# /root/.local/share/perfetto/prebuilts/trace_processor_shell


class InstallAfterCompile(install):
    def run(self):
        # build before install, copy to bin
        if not LITE_MODE:
            self.run_command("build_ext")
        install.run(self)


class BuildAndMoveToBin(build_ext):
    def build_extension(self, ext):
        # the library is used for dlopen, move to bin
        super().build_extension(ext)
        if not LITE_MODE:
            build_lib_path = self.get_ext_fullpath(ext.name)
            shutil.move(build_lib_path, "bin/libpy_xpu_timer_callstack.so")


ext = (
    []
    if LITE_MODE
    else [
        Extension(
            "py_xpu_callstack",
            sources=["src/py_xpu_callstack.cc"],
        )
    ]
)

lite_mode_files = [i for i in glob("bin/*") if not i.endswith(".so")]
full_mode_files = [i for i in glob("bin/*")]

version = "1.1+lite" if LITE_MODE else f"1.1+{os.environ['XPU_PLATFORM'].lower()}"

setup(
    name="py_xpu_timer",
    version=version,
    packages=find_packages(),
    data_files=[("bin", lite_mode_files if LITE_MODE else full_mode_files)],
    ext_modules=ext,
    cmdclass={"build_ext": BuildAndMoveToBin, "install": InstallAfterCompile},
    install_requires=["aiohttp", "py-spy", "tqdm", "pandas", "matplotlib"],
    python_requires=">=3.7",
)
