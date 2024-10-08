import logging
import re
from importlib import metadata
from typing import List, Tuple, Union

import torch
from packaging.version import Version

__all__: List[str] = ["torch_version"]


# Adopted from Fairscale
def torch_version(
    version: str = torch.__version__, return_dev: bool = False
) -> Union[Tuple[int, ...], Tuple[Tuple[int, ...], str]]:
    numbering = re.search(r"^(\d+).(\d+).(\d+)([^\+]*)(\+\S*)?$", version)
    if not numbering:
        return tuple()
    # Catch torch version if run against internal pre-releases, like `1.8.0a0fb`,
    if numbering.group(4):
        # Two options here:
        # - either skip this version (minor number check is not relevant)
        # - or check that our codebase is not broken by this ongoing development.

        # Assuming that we're interested in the second use-case more than the first,
        # return the pre-release or dev numbering
        logging.warning(f"Pytorch pre-release version {version} - assuming intent to test it")

    ver = tuple(int(numbering.group(n)) for n in range(1, 4))
    if return_dev:
        return ver, numbering.group(4)
    else:
        return ver


def get_digit_part(string):
    match = re.search(r"^\d+", string)
    if match:
        return match.group()
    else:
        return ""


def get_version(package):
    version = package.__version__
    numbering = version.split(".")
    vs = [get_digit_part(x) for x in numbering]
    digits = [int(x) if x != "" else "" for x in vs]
    return tuple(digits)


def package_version_smaller_than(pkg_name, version):
    pkg_v = metadata.version(pkg_name)
    return Version(pkg_v) < Version(version)


def package_version_bigger_than(pkg_name, version):
    pkg_v = metadata.version(pkg_name)
    return Version(pkg_v) > Version(version)


def package_version_equal_to(pkg_name, version):
    pkg_v = metadata.version(pkg_name)
    return Version(pkg_v) == Version(version)
