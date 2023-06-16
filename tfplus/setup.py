"""Setup for pip package."""
from __future__ import absolute_import, division, print_function

import fnmatch
import os
import shutil
import sys
import tempfile

from setuptools import sandbox

content = """
# Copyright 2023 The tfplus Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
\"\"\"Setup for pip package.\"\"\"
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from setuptools import find_packages
from setuptools import setup
from setuptools.dist import Distribution

REQUIRED_PACKAGES = [
    # "wheel",
    # "aliyun-log-python-sdk",
    # "sklearn",
    # "pyodps",
    # "requests",
    # "six",
    # "objgraph",
    "numpy",
    "pandas",
    # "scipy",
]
package_name = '{}'
version = '{}'
author_email = '{}'
description = '{}'
url = '{}'

class BinaryDistribution(Distribution):
  \"\"\"This class is needed in order to create OS specific wheels.\"\"\"

  def has_ext_modules(self):
    return True

setup(
    name=package_name,
    version=version,
    description=description,
    author='xxx.',
    author_email=author_email,
    url=url,
    # Contained modules and scripts.
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    # extras_require=,
    # Add in any packaged data.
    include_package_data=True,
    zip_safe=False,
    distclass=BinaryDistribution,
    # PyPI package information.
    classifiers=[],
    license='Apache License Version 2.0, January 2004',
    keywords='tfplus',
)
"""

package_info = {}
with open("tfplus/version.py", encoding="utf-8") as fd:
    exec(compile(fd.read(), "<string>", "exec"))  # pylint: disable=exec-used

extras_require = {
    # "xx": ["xx==0.5.6"],
}

rootpath = tempfile.mkdtemp()
print(f"setup.py - create {rootpath} and copy tfplus")
shutil.copytree("tfplus", os.path.join(rootpath, "tfplus"))

print(f"setup.py - create {rootpath}/MANIFEST.in")
with open(os.path.join(rootpath, "MANIFEST.in"), "w", encoding="utf-8") as f:
    f.write("recursive-include tfplus *.so")

package_name = package_info["package_name"]
package_version = package_info["version"]
print(
    f"setup.py - create {rootpath}/setup.py, project_name = '{package_name}' and __version__ = {package_version}"
)  # pylint: disable=line-too-long
with open(os.path.join(rootpath, "setup.py"), "w", encoding="utf-8") as f:
    f.write(
        content.format(
            package_info["package_name"],
            package_info["version"],
            package_info["author"],
            package_info["description"],
            package_info["url"],
        )
    )

datapath = os.environ.get("TFPLUS_DATAPATH") or "bazel-bin"

if datapath is not None:
    for rootname, _, filenames in os.walk(os.path.join(datapath, "tfplus")):
        if not fnmatch.fnmatch(rootname, "*test*") and not fnmatch.fnmatch(rootname, "*runfiles*"):
            for filename in fnmatch.filter(filenames, "*.so"):
                src = os.path.join(rootname, filename)
                dst = os.path.join(
                    rootpath,
                    os.path.relpath(os.path.join(rootname, filename), datapath),
                )
                print(f"setup.py - copy {src} to {dst}")
                shutil.copyfile(src, dst)

for rootname, _, filenames in os.walk(os.path.join(datapath, "tfplus")):
    if not fnmatch.fnmatch(rootname, "*test*") and not fnmatch.fnmatch(rootname, "*runfiles*"):
        for filename in fnmatch.filter(filenames, "*.py"):
            src = os.path.join(rootname, filename)
            dst = os.path.join(
                rootpath,
                os.path.relpath(os.path.join(rootname, filename), datapath),
            )
            print(f"setup.py - copy {src} to {dst}")
            shutil.copyfile(src, dst)

setup_path = os.path.join(rootpath, "setup.py")
dest_path = sys.argv[1:]
print(f"setup.py - run sandbox.run_setup {setup_path} {dest_path}")
sandbox.run_setup(os.path.join(rootpath, "setup.py"), sys.argv[1:])

if not os.path.exists("dist"):
    os.makedirs("dist")
for f in os.listdir(os.path.join(rootpath, "dist")):
    src_path = os.path.join(rootpath, "dist", f)
    dest_path = os.path.join("dist", f)
    print(f"setup.py - copy {src_path} to {dest_path}")
    shutil.copyfile(os.path.join(rootpath, "dist", f), os.path.join("dist", f))
print(f"setup.py - remove {rootpath}")
shutil.rmtree(rootpath)
print("setup.py - complete")
