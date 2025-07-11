import textwrap

from . import BaseBuildRender


class BuildRender(BaseBuildRender):
    @classmethod
    def add_arguments(cls, parser):
        BaseBuildRender.add_arguments(parser)

    def __post_init__(self):
        if self.args.sdk_path is None:
            self.args.sdk_path = "/usr/local/Ascend/ascend-toolkit/latest/"

    def rend_config_bzl(self):
        hpu_config = textwrap.dedent(
            """
            XPU_TIMER_CONFIG = struct(
                linkopt = [
                    "-Wl,--version-script=$(location //xpu_timer/hpu:only_keep_hpu.lds)",
                    "-L{hpu_path}/lib64",
                ],
                copt = [
                    "-DXPU_HPU",
                ],
                deps = ["@hpu//:hpu_headers", "//xpu_timer/hpu:only_keep_hpu.lds"],
                timer_deps = ["//xpu_timer/hpu:hpu_timer"],
                hook_deps = ["//xpu_timer/hpu:hpu_hook"],
            )"""
        )

        self.xpu_timer_config.append(hpu_config.format(hpu_path=self.sdk_path))
        return "\n".join(self.xpu_timer_config)

    def rend_bazelrc(self):
        self.bazelrc_config.append(f"build --repo_env=HPU_HOME={self.sdk_path}")
        return "\n".join(self.bazelrc_config)

    def setup_files(self):
        with open("WORKSPACE.template") as f:
            workspace = f.read()

        deps = textwrap.dedent(
            """
            load("//third_party/hpu:hpu_workspace.bzl", "hpu_workspace")
            hpu_workspace()
            """
        )
        return workspace + deps

    def setup_platform_version(self):
        # /usr/local/Ascend/ascend-toolkit/latest/version.cfg

        version = None
        path = f"{self.sdk_path}/version.cfg"
        pattern = "toolkit_installed_version="
        with open(path) as f:
            for line in f:
                if line.startswith(pattern):
                    version = line.split(pattern)[-1]
                    break
        if version is None:
            raise ValueError("Cannot found version")

        major = version.split(":")[-1].replace("]", "").replace(".", "")
        minor = ""
        return f"hu{major}{minor}", "HPU"
