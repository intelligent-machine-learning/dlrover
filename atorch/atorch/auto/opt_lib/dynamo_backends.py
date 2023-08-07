# listing all backends supported by torch dynamo
import enum


class DynamoBackends(enum.Enum):
    NO = "NO"
    EAGER = "EAGER"
    AOT_EAGER = "AOT_EAGER"
    INDUCTOR = "INDUCTOR"
    NVFUSER = "NVFUSER"
    AOT_NVFUSER = "AOT_NVFUSER"
    AOT_CUDAGRAPHS = "AOT_CUDAGRAPHS"
    OFI = "OFI"
    FX2TRT = "FX2TRT"
    ONNXRT = "ONNXRT"
    IPEX = "IPEX"
