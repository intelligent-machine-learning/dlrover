import setuptools.command.build_py
from setuptools import find_packages, setup


class build_proto(setuptools.command.build_py.build_py):
    def run(self):
        try:
            self.spawn(["sh", "bin/build_proto.sh"])
        except RuntimeError as e:
            self.warn(f"build proto error:{e}")
        super().run()


def fetch_requirements(path):
    with open(path, "r") as fd:
        return [r.strip() for r in fd.readlines()]


required_deps = fetch_requirements("atorch/requirements.txt")

cmdclass = {}
cmdclass["build_py"] = build_proto

setup(
    name="atorch",
    version="0.0.1",  # render by script,do not modify
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
