from penrose.util.log_util import default_logger as logger 
from penrose.util.conf_util import get_conf

 

class TFKubernetesWorker:
    """KubemakerWorker"""

    def __init__(self, args):
        """
        Argument:
            args: result of parsed command line arguments
        """
        self._args = args
        task_conf = get_conf(py_conf=args.conf)
        self.tensorflow_failover = TensorflowFailover()
        self.init_executor(task_conf)

    def init_executor(self, task_conf):
        pass 
    
    def start_failover_monitor(self):
        pass

    def run(self):
        pass