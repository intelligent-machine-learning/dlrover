load("//:workspace.bzl", "dynamic_local_repository")
load("@xpu_timer_cfg//:xpu_config.bzl", "PYTHON_INCLUDE", "PYTHON_LIB")

def python_workspace():
    dynamic_local_repository(
        name = "python_lib",
        include = "PYTHON_INCLUDE",
        include_default_path = PYTHON_INCLUDE,
        lib = "PYTHON_LIB",
        lib_default_path = PYTHON_LIB,
        build_file = "//third_party/python:python.BUILD",
        template = True,
    )
