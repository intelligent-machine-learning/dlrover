class InferenceConfigKey(object):
    LOG_FILE = "log_file"
    ERRORS = "errors"


class DiagnoseAction(object):
    RESTART_WORKER = "restart_worker"
    RELAUNCH_WORKER = "relaunch_worker"
