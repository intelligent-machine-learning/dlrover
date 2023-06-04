from typing import Dict

from setuptools import find_packages, setup

cmdclass: Dict[type, type] = {}


def fetch_requirements(path):
    with open(path, "r") as fd:
        return [r.strip() for r in fd.readlines()]


required_deps = fetch_requirements("atorch/requirements.txt")

setup(
    name="atorch",
    version="$version",  # render by script,do not modify
    description="A pytorch extension for efficient deep learning.",
    author="Ant Group",
    python_requires=">=3.5",
    packages=find_packages(exclude=["*test*"]),
    # 0.4.2,require python3.7
    install_requires=required_deps,
    package_data={"": ["*.so"]},
    cmdclass=cmdclass,
    # data_files=["atorch/requirements.txt", "bin/build_proto.sh", "atorch/protos/coworker.proto"],
    data_files=["atorch/requirements.txt"],
)
