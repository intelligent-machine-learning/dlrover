# Copyright 2023 tfplus.
"""
  Demo ops.
"""

from __future__ import absolute_import, division, print_function

from tfplus.python.common import _load_library

demo_ops = _load_library("_demo.so")


def print_localtime():
    """
    print_localtime
    Args:
    """
    return demo_ops.print_localtime()


if __name__ == "__main__":
    print_localtime()
