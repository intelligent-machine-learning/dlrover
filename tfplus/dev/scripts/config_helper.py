"""Config Utility to write .bazelrc based on tensorflow."""
from __future__ import print_function

import re
import sys

import tensorflow as tf


def write_config():
    """Retrive compile and link information from tensorflow and write to .bazelrc."""

    cflags = tf.sysconfig.get_compile_flags()

    inc_regex = re.compile("^-I")
    opt_regex = re.compile("^-D")

    include_list = []
    opt_list = []

    for arg in cflags:
        if inc_regex.match(arg):
            include_list.append(arg)
        elif opt_regex.match(arg):
            opt_list.append(arg)
        else:
            print(f"WARNING: Unexpected cflag item {arg}")

    if len(include_list) != 1:
        print("ERROR: Expected a single include directory in " + "tf.sysconfig.get_compile_flags()")
        sys.exit(1)

    library_regex = re.compile("^-l")
    libdir_regex = re.compile("^-L")

    library_list = []
    libdir_list = []

    lib = tf.sysconfig.get_link_flags()

    for arg in lib:
        if library_regex.match(arg):
            library_list.append(arg)
        elif libdir_regex.match(arg):
            libdir_list.append(arg)
        else:
            print(f"WARNING: Unexpected link flag item {arg}")

    if len(library_list) != 1 or len(libdir_list) != 1:
        print("ERROR: Expected exactly one lib and one libdir in" + "tf.sysconfig.get_link_flags()")
        sys.exit(1)

    try:
        with open(".bazelrc", "w", encoding="utf-8") as bazel_rc:
            for opt in opt_list:
                bazel_rc.write(f'build --copt="{opt}"\n')

        bazel_rc.write(f'build --action_env TF_HEADER_DIR="{include_list[0][2:]}"\n')

        bazel_rc.write("test --cache_test_results=no\n")
        bazel_rc.write("test --test_output all\n")
        bazel_rc.write(
            f'build --action_env TF_SHARED_LIBRARY_DIR="{libdir_list[0][2:]}"\n'
        )  # pylint: disable=line-too-long
        library_name = library_list[0][2:]
        if library_name.startswith(":"):
            library_name = library_name[1:]
        else:
            library_name = "lib" + library_name + ".so"
        bazel_rc.write(f'build --action_env TF_SHARED_LIBRARY_NAME="{library_name}"\n')
        bazel_rc.close()
    except OSError:
        print("ERROR: Writing .bazelrc")
        sys.exit(1)


def write_sanitizer():
    """Append asan config for sanitizers"""
    asan_options = "handle_abort=1:allow_addr2line=true:check_initialization_order=true:strict_init_order=true:detect_odr_violation=1"  # noqa : E501

    ubsan_options = "halt_on_error=true:print_stacktrace=1"
    try:
        with open(".bazelrc", "a", encoding="utf-8") as bazel_rc:
            bazel_rc.write("\n\n# Basic ASAN/UBSAN that works for gcc\n")
            bazel_rc.write("build:asan --define ENVOY_CONFIG_ASAN=1\n")
            bazel_rc.write("build:asan --copt -fsanitize=address\n")
            bazel_rc.write("build:asan --linkopt -lasan\n")
            bazel_rc.write("build:asan --define tcmalloc=disabled\n")
            bazel_rc.write("build:asan --build_tag_filters=-no_asan\n")
            bazel_rc.write("build:asan --test_tag_filters=-no_asan\n")
            bazel_rc.write("build:asan --define signal_trace=disabled\n")
            bazel_rc.write("build:asan --copt -D__SANITIZE_ADDRESS__\n")
            bazel_rc.write(f'build:asan --test_env=ASAN_OPTIONS="{asan_options}"\n')
            bazel_rc.write(f'build:asan --test_env=UBSAN_OPTIONS="{ubsan_options}"\n')
            bazel_rc.write("build:asan --test_env=ASAN_SYMBOLIZER_PATH\n")
            bazel_rc.close()
    except OSError:
        print("ERROR: Writing .bazelrc")
        sys.exit(1)


write_config()
write_sanitizer()
