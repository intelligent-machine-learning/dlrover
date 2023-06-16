class Task(object):
    def __init__(self, id, task_type, process_mode="ONE_PROCESS", task_info=None):
        self.id = id
        self.type = task_type
        self.process_mode = process_mode
        if task_type in ["TUNE", "DRYRUN", "FINISH"]:
            self.strategy = task_info
        elif task_type == "SETUP_PARALLEL_GROUP":
            self.parallel_group_info = task_info
        elif task_type == "ANALYSE":
            self.analysis_method = task_info
