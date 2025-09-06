load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_file")

def matmul_benchmark_workspace():

    http_file(
      name = "cublaslt_gemm_bin",
      sha256 = "3f81a13c63724ab32dbefa4a723f4bdeeaf11da9bb19be7b28b2116d96e7f221",
      url = "http://alps-common.oss-cn-hangzhou-zmf.aliyuncs.com/users/lizhi/cublaslt_gemm",
      executable = True,
    )
    
    http_file(
      name = "cublas_benchmark_bin",
      sha256 = "a64baee971a41fd3cd2900305af4fad33ba50e4e9719aa86e59670e8c7ccd002",
      url = "http://alps-common.oss-cn-hangzhou-zmf.aliyuncs.com/users/lizhi/cublas_benchmark",
      executable = True,
    )
