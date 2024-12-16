# version is genreated by bazel
import io
import sys

from .version import *

if isinstance(sys.stdout, io.TextIOWrapper) and sys.version_info >= (3, 7):
    sys.stdout.reconfigure(encoding="utf-8")  # type ignore[attr-defined]
print(f"git commit is {__version__}")  # type: ignore[name-defined]
print(f"build time is {__build_time__}")  # type: ignore[name-defined]
print(f"build type is {__build_type__}")  # type: ignore[name-defined]
print(f"build platform is {__build_platform__}")  # type: ignore[name-defined]
print(f"build platform version is {__build_platform_version__}")  # type: ignore[name-defined]
