import threading
import time

from dlrover.python.common.diagnosis import K8sPodData, DiagnosisDataType
from dlrover.python.common.log import default_logger as logger
from dlrover.python.master.diagnosis.diagnosis import DiagnosisManager
from dlrover.python.master.watcher.k8s_watcher import K8sPodWatcher
from dlrover.python.scheduler.job import JobArgs


class PodMonitor(object):

    def __init__(self, job_args: JobArgs):
        self._stopped = False
        self._k8s_pod_watcher = K8sPodWatcher(job_args.job_name, job_args.namespace)

    def start(self):
        """Start Detecting. The method should be called only once."""
        threading.Thread(
            target=self._monitor_pod,
            name="pod-monitor",
            daemon=True
        ).start()

    def stop(self):
        self._stopped = True

    def _monitor_pod(self):
        logger.info("Start monitoring pod events.")
        while True:
            logger.info("PodMonitor: monitoring pods")
            if self._stopped:
                logger.info("Stop monitoring pods.")
                break
            try:
                pods = self._k8s_pod_watcher.list()
                logger.info(f"PodMonitor: get pods {len(pods)}")
                data = K8sPodData(0, pods)
                DiagnosisManager.singleton_instance().collect_diagnosis_data(DiagnosisDataType.K8SPODDATA, data)
            except Exception as e:
                logger.warning(e)
                time.sleep(30)
            time.sleep(5)

