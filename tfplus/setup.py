# Copyright 2023 The TFPlus Authors. All rights reserved.
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
"""Setup for pip package."""
from __future__ import absolute_import, division, print_function

import fnmatch
import os
import shutil

from setuptools import find_packages, setup
from setuptools.command.install import install
from setuptools.dist import Distribution

USING_SO_FILES = (
    "_kv_variable_ops.so",
    "_flash_attention.so",
    "libtfplus.so",
    "libtfplus_opdef.so",
    "libkv_variable_opdef.so",
    "libstorage_config_proto_cc.so",
)

__version__ = "0.1.0"
REQUIRED_PACKAGES = [
  "tensorflow-cpu==2.13.0"
]
project_name = "tfplus"
project = "tfplus"

datapath = "bazel-bin"
generated_files = []
package_data = {}

if datapath is not None:
  for rootname, _, filenames in os.walk(os.path.join(datapath, "tfplus")):
    if not fnmatch.fnmatch(rootname, "*test*") and not fnmatch.fnmatch(
        rootname, "*runfiles*"):
      for filename in fnmatch.filter(filenames, "*.so"):
        if filename not in USING_SO_FILES:
          print("The tfplus doesn't use {}. Skipped.".format(filename))
          continue
        src = os.path.join(rootname, filename)
        dst = src[len(datapath) + 1:]
        print("Copy src: ", src, " dst: ", dst)
        shutil.copyfile(src, dst)
        generated_files.append(dst)
        package_data.setdefault("tfplus", [])
        package_data["tfplus"].append(dst[len("tfplus") + 1:] + "*")

genfile_path = "bazel-out/k8-opt/bin"
for rootname, _, filenames in os.walk(os.path.join(genfile_path, "tfplus")):
  if not fnmatch.fnmatch(rootname, "*test*") and not fnmatch.fnmatch(
      rootname, "*runfiles*"):
    for filename in fnmatch.filter(filenames, "*.py"):
      src = os.path.join(rootname, filename)
      dst = src[len(genfile_path) + 1:]
      key = "tfplus"
      shutil.copyfile(src, dst)
      generated_files.append(dst)
      package_data.setdefault(key, [])
      package_data[key].append(dst[len(key) + 1:])
      print("---cccccing:" + dst[len(key) + 1:])


class InstallPlatlib(install):

  def finalize_options(self):
    install.finalize_options(self)
    self.install_lib = self.install_platlib


class BinaryDistribution(Distribution):
  """This class is needed in order to create OS specific wheels."""

  def has_ext_modules(self):
    return True

  def is_pure(self):
    return False


libdfs_rel_path = os.path.join("dfs", "python", "ops", "libzdfs.so")
package_data["tfplus"].append(libdfs_rel_path + "*")
package_data["tfplus"] = list(set(package_data["tfplus"]))

print(find_packages())
setup(
    name=project_name,
    version=__version__,
    description=("TFPlus is a high-performance TensorFlow extension"
    " library developed in-house by Ant Group"),
    long_description="""
TFPlus is a high-performance TensorFlow extension library developed in-house
by Ant Group and encapsulates the Ant Group's core capabilities for large-scale
sparse training. TFPlus has accumulated essential functionalities and performance
optimizations for core sparse scenarios. It has deeply optimized the performance
in terms of IO, operators, graph optimization, distribution, and collective
communication for sparse models. Also, it provides special optimizers, fault
tolerance, elasticity, incremental updates, etc., unique to sparse scenarios.
Its main features are:
-  Provide highly efficient TF operator extensions in a plug-in manner
-  Support high-performance sparse Embedding training in recommendation scenarios: Kv Variable
-  Offer high-performance, self-developed deep learning optimizers
    """,
    long_description_content_type='text/markdown',
    author="Ant Group",
    url="https://github.com/intelligent-machine-learning/dlrover",
    python_requires=">=3.8",
    # Contained modules and scripts.
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    # Add in any packaged data.
    # include_package_data=True,
    zip_safe=False,
    distclass=BinaryDistribution,
    cmdclass={"install": InstallPlatlib},
    package_data=package_data,
    # PyPI package information.
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Libraries",
    ],
    license="Apache License 2.0",
    keywords="tfplus",
)
