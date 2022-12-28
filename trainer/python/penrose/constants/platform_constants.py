from penrose.constants.constants import Constant


class PlatformConstants(object):
    """Platform related constants"""

    PlatformName = Constant("Platform")
    ExecutePlaform = Constant("ExecutePlaform")
    Local = Constant("LOCAL", "LOCAL")
    Kubernetes = Constant("KUBERNETES", "KUBERNETES")
    WorkerActionRun = Constant("run","run")
    