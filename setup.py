from setuptools import find_packages, setup

with open("dlrover/requirements.txt") as f:
    required_deps = f.read().splitlines()

extras = {}

setup(
    name="dlrover",
    version="0.1.0rc0.dev0",
    description="An Automatic Distributed Deep Learning Framework",
    long_description="DLRover helps model developers focus on model algorithm"
    " itself, without taking care of any engineering stuff,"
    " say, hardware acceleration, distribute running, etc."
    " It provides static and dynamic nodes' configuration automatically,"
    ", before and during a model training job running on k8s",
    long_description_content_type="text/markdown",
    author="Ant Group",
    url="https://github.com/intelligent-machine-learning/dlrover",
    install_requires=required_deps,
    extras_require=extras,
    python_requires=">=3.5",
    packages=find_packages(
        exclude=[
            "*test*",
            "model_zoo*",
        ]
    ),
    package_data={
        "": [
            "proto/*",
            "docker/*",
            "Makefile",
            "requirements.txt",
        ]
    },
)
