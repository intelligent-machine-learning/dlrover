import time
from dlrover.trainer.util.log_util import default_logger as logger
from dlrover.python.master.elastic_training.elastic_ps import ElasticPsService


class FailoverClient:
    """
        FailoverClient interacts with dlrover master. 
        It get's ps address, ps/worker global/local version.
    """

    def __init__(self, role=None):
        logger.info("initiating FailoverClient")
        self.role = role
        task_type, task_id = role.split(":")
        task_id = int(task_id)
        self.task_type = task_type
        self.task_id = task_id
        if self.task_type == "chief":
            self.task_type = "worker"
        if self.task_type == "worker":
            self.task_id =+ 1 
        self._client = ElasticPsService(self.task_type, self.task_id)
        logger.info("ElasticPsService is created, task_type: {} and task_id {}.".format(task_type, task_id))
        self.ps_client = ElasticPsService("ps", 0)


    def get_local_version(self):
        local_version = self._client.get_local_cluster_version()
        logger.info("get local version : %s.", local_version)
        return local_version


    def get_global_version(self):
        global_version = self.ps_client.get_global_cluster_version()
        logger.info("get ps global version : %s.".format(global_version))
        return global_version
     
    def ready_for_ps_relaunch(self):
        logger.info("Noticing dlrover master that it's ready for ps relaunch")
        self._client.ready_for_ps_relaunch()

    def set_global_version(self, version=0):
        self.ps_client.update_global_cluster_version(version)
        logger.info(
            "successfully set ps global version: %s.", version
        )
    

    def set_local_version(self, version=0):
        self._client.update_local_cluster_version(version)
        logger.info(
            "successfully set local version: {}.", version
        )


    def get_training_ps_addr(self):
        logger.info("get training ps addresses")
        ps_nodes, new_ps_ready = self._client.get_all_ps_nodes()
        logger.info("ps_nodes is %s",ps_nodes)
        logger.info("new_ps_ready is %s",new_ps_ready)
        return [n.addr for n in ps_nodes]


    def init_version(self, version=0):
        logger.info("initiating local and global version")
        local_version = self.get_local_version()
        global_version = self.get_global_version()
        if local_version == 0 and self.task_type=="ps":
            version = local_version + 1 
            self.set_local_version(version)
            if self.task_id==0:
                # ps:0 updates global version while 
                # other ps waiting for global version to be updated
                self.set_global_version(version)
            else:
                while (global_version==0):
                    global_version = self.get_global_version()
                    time.sleep(3)
                    logger.info("Waiting for ps:0 updating global version from 0 to 1.")
            
        if self.task_type in ["worker","chief"] and local_version==0:
            self.set_local_version(1)
            while (global_version==0):
                # workers waits for global version to be updated to 1
                global_version = self.get_global_version()
                time.sleep(3)
                logger.info("Waiting for ps-0 updating global version from 0 to 1.")
            version = self.get_local_version()
            logger.info("{}:{} local version is {}".format(self.task_type, self.task_id, version))
