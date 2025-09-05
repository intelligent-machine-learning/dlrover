
from enum import Enum, auto, unique

ANT_PATCH_ENV_NOT_SET = "ANT_PATCH_ENV_NOT_SET"

@unique
class PatchStatus(Enum):
    PATCH_OK_VERSION_CHECK = auto()
    PATCH_OK_ENV_CHECK = auto()
    PATCH_FAIL_VERSION_CHECK = auto()
    PATCH_FAIL_ENV_CHECK = auto()
    UNKNOWN = auto()

    def __str__(self):
        return PATCH_STATUS_MSG[self]

PATCH_STATUS_MSG = {
   PatchStatus.PATCH_OK_VERSION_CHECK: "Y,Patch OK(VERSION)",
   PatchStatus.PATCH_OK_ENV_CHECK: "Y,Patch OK(ENV)",
   PatchStatus.PATCH_FAIL_VERSION_CHECK: "N,Patch Fail(VERSION)",
   PatchStatus.PATCH_FAIL_ENV_CHECK: "N,Patch Fail(ENV)",
   PatchStatus.UNKNOWN: "unknown",
}

a = PatchStatus.PATCH_OK_VERSION_CHECK
print(a)
