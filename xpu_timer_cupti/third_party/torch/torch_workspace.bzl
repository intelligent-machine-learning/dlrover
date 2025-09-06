load("//:workspace.bzl", "dynamic_local_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@xpu_timer_cfg//:xpu_config.bzl", "TORCH_PATH")

def torch_workspace():
    dynamic_local_repository(
        name = "torch",
        include_default_path = TORCH_PATH,
        build_file = "//third_party/torch:torch.BUILD",
        include = "TORCH_HOME",
    )

    http_archive(
        name = "flash_attn",
        build_file = "//third_party/torch:flash_attn.BUILD",
        strip_prefix = "flash-attention-fa2_pack_glm_mask",
        urls = ["https://github.com/intelligent-machine-learning/flash-attention/archive/fa2_pack_glm_mask.tar.gz"],
        sha256 = "1e2ab9fb7198c57f7f332e0b30f988bb47fb66220af56de6411d8744805e2e2b",
    )
