"""fp common"""
from __future__ import absolute_import, division, print_function

import ctypes
import inspect
import os
import sys
from distutils.version import LooseVersion


def _load_library(filename):
    """_load_library"""
    f = inspect.getfile(sys._getframe(1))  # pylint: disable=protected-access
    # Construct filename

    f = os.path.join(os.path.dirname(f), filename)
    filenames = [f]

    # Add datapath to load if en var is set, used for running tests where shared
    # libraries are built in a different path
    datapath = os.environ.get("FPLIB_DATAPATH")
    if datapath is not None:
        # Build filename from `datapath` + `package_name` + `relpath_to_library`
        f = os.path.join(datapath, os.path.relpath(f, os.path.dirname(filename)))
        filenames.append(f)

    # Function to load the library, return True if file system library is loaded
    load_fn = ctypes.cdll.LoadLibrary

    # Try to load all paths for file, fail if none succeed
    errs = []
    for f in filenames:
        try:
            l = load_fn(f)
            if l is not None:
                return l
        except Exception as err:  # pylint: disable=broad-except
            print("try load " + f + " failed.")
            errs.append(err)
    raise NotImplementedError(
        f"unable to open file: {filename}, from paths: {filenames}\ncaused by: {errs}"
    )  # pylint: disable=line-too-long
