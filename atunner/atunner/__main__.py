"""
ATunner main entry point.

This module provides the main entry point for the ATunner CLI.
"""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
