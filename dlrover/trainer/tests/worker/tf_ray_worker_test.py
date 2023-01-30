import unittest
import threading 
import time 
from dlrover.trainer.worker.tf_ray_worker import TFRayWorker
from dlrover.trainer.util.log_util import default_logger as logger

class TFRayWorkerTest(unittest.TestCase):
    
    def test_ray_worker_ps_init(self):
        class Conf:
            task_id = 0
            task_type = "ps"
        
        class args:
            conf = Conf
        tf_ray_worker = TFRayWorker(args)
        tf_ray_worker.health_check()
 

    def test_ray_worker_worker_init(self):
        class Conf:
            task_id = 0
            task_type = "worker"
        
        class args:
            conf = Conf

        tf_ray_worker = TFRayWorker(args)
        tf_ray_worker.health_check()
        import pdb 
        pdb.set_trace()
        tf_ray_worker.run()