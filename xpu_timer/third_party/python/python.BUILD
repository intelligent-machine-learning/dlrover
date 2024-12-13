package(
    default_visibility=["//visibility:public"]
)

cc_library(
    name = "python",
    hdrs = glob(["*.h", "**/*.h"]),
    includes = ["./"],
)

cc_import(
    name = "python_import",
    hdrs = glob(["*.h", "**/*.h"]),
    includes = ["./"],
    shared_library = "%{SHARED_LIB}", 
)
