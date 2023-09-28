package(default_visibility = ["//visibility:public"])

licenses(["notice"]) # # BSD 3-Clause

genrule(
  name = "build_murmurhash",
  srcs = glob(["**"]) + [
    "@local_config_cc//:toolchain",
  ],
  outs = [
    "libmurmurhashh.a",
  ],
  cmd = """
    set -e
    WORK_DIR=$$PWD
    DEST_DIR=$$PWD/$(@D)
    export CXXFLAGS="-msse4.2 -msse4.1 -mavx -mavx2 -mfma -mfpmath=both -frecord-gcc-switches -D_GLIBCXX_USE_CXX11_ABI=0  -std=c++17 -fPIC -DNDEBUG -D_GLIBCXX_USE_CXX11_ABI=0 -D__STDC_FORMAT_MACROS -fno-canonical-system-headers -Wno-builtin-macro-redefined -D__DATE__=redacted -D__TIMESTAMP__=redacted -D__TIME__=redacted"
    pushd external/murmurhash
    g++ $$CXXFLAGS -c src/MurmurHash1.h src/MurmurHash2.h src/MurmurHash3.h src/MurmurHash1.cpp src/MurmurHash2.cpp src/MurmurHash3.cpp
    ar -crv libmurmurhash.a MurmurHash1.o MurmurHash2.o MurmurHash3.o
    cp libmurmurhash.a $$DEST_DIR/libmurmurhashh.a
    popd
    """,
)

cc_library(
  name = "murmurhash",
  hdrs = [
    "src/MurmurHash1.h",
    "src/MurmurHash2.h",
    "src/MurmurHash3.h"
  ],
  srcs = [
    "libmurmurhashh.a",
  ],
  visibility = ["//visibility:public"],
  linkstatic = 1,
)