import os 
import json 
from dlrover.trainer.util.log_util import default_logger as logger

def get_tf_config():
    tf_config = json.loads(os.environ.get("TF_CONFIG") or "{}")
    if not tf_config:
        logger.error(
            "TF_CONFIG should not be empty in distributed environment."
        )
        raise Exception(
            "TF_CONFIG should not be empty in distributed environment."
        )
    return tf_config


def get_tf_config_task_type_and_index():
    tf_config = get_tf_config()
    task = tf_config.get("task", None)
    if task is None:
        raise Exception(
            "TF_CONFIG task should not be empty in distributed environment."
        )
    task_type = task.get("type", None)
    task_index = task.get("index", None)
    if task_type is None or task_index is None:
        raise Exception(
            "TF_CONFIG task type or index should not be empty in distributed environment."
        )
    return task_type, task_index 