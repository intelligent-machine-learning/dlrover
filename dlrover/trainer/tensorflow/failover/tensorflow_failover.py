#!/usr/bin/env python
# coding: utf-8

import os
import json 
import time 
import threading
from dlrover.trainer.tensorflow.util import common_util
from dlrover.trainer.util.log_util import default_logger as logger
from dlrover.trainer.tensorflow.failover.failover_client import FailoverClient
  

class TensorflowFailover:
    def __init__(
        self,
        failover_client=FailoverClient,
        failover_level=1
    ):
        """
        Args:
            role: "ps:0", "worker:1"
            uuid_key: unique key that marks this cluster
            failover_level: switch for dynet
        """
        logger.info("initiating tensorflow_failover and failover level is {}".format(failover_level))
        self._failover_level = failover_level
        if common_util.should_failover(self._failover_level):
            self.init_for_dynet()
            self._failover_client = failover_client(role=self._role)
            self._failover_client.init_version()


    def init_for_dynet(self):
        TF_CONFIG = os.getenv("TF_CONFIG")
        if isinstance(TF_CONFIG,str):
            TF_CONFIG = json.loads(TF_CONFIG)
            task_type = TF_CONFIG["task"]["type"]  
            task_id =  TF_CONFIG["task"]["index"]
            self._role = task_type + ":" + str(task_id)
            # address 有可能是两种，一种是稠密的，一种是sparse
            # to adaption
            self._address = TF_CONFIG["cluster"][task_type][task_id]
        if self._role is None:
            return 
        self.task_type, self.task_id = self._role.split(":")
        self.task_id = int(self.task_id)
        self._is_chief = (
            self.task_type == "chief" and self.task_id == 0
        )
        self.prev_address = TF_CONFIG["cluster"]["ps"]

    def start_failover_monitor(self):
        if self._role and self.task_type not in ["evaluator","ps"]: 
            self._start_failover_monitor()
    

    def _start_failover_monitor(self):
        def monitor_fun():
            logger.info("Successfully to start failover monitor!")
            while True:
                ps_address_changed, _ = self.ps_addresses_changed()
                if ps_address_changed:
                    self.refresh_env()
                time.sleep(10)
        self.runner = threading.Thread(target=monitor_fun)
        self.runner.setDaemon(True)
        self.runner.start()

    def ps_addresses_changed(self):
        '''
            Check whether ps addresses changed.
            There are at least two kinds: 1) the num of ps 
            addresses changed, 2) single ps address varies.
        '''
        changed = False
        changed_type = None
        curr_address = self._failover_client.get_training_ps_addr()
        logger.info("prev address is {} and current address is {}".format(self.prev_address, curr_address))
        if "".join(curr_address)!="".join(self.prev_address):
            if len(curr_address) != len(self.prev_address):
                changed_type = "scaling"
            else:
                changed_type = "migrating"
            self.prev_address = curr_address
            changed = True
        return changed, changed_type

    def refresh_env(self):
        """Refresh tf env
        update TF_CONFIG, when the training thread restarts 
        estimator will use the new TF_CONFIG
        """
        ps_address = self.prev_address
        cluster = {"cluster":
                        {"worker":{
                                    str(self.task_id):
                                            self._address},
                         "ps": 
                              ps_address
                        }, 
                         "task":{"type": self.task_type, "index":self.task_id}}
        os.environ["TF_CONFIG"] = json.dumps(cluster)
 
 

     
