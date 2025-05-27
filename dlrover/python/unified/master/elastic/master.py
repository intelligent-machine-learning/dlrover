#  Copyright 2025 The DLRover Authors. All rights reserved.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import threading
from typing import Dict

from dlrover.python.common.constants import RendezvousName
from dlrover.python.common.event.reporter import get_event_reporter
from dlrover.python.master.elastic_training.rdzv_manager import \
    RendezvousManager, ElasticTrainingRendezvousManager, \
    NetworkCheckRendezvousManager, _master_evt
from dlrover.python.master.elastic_training.sync_service import SyncService
from dlrover.python.master.master import ElasticMaster
from dlrover.python.master.monitor.perf_monitor import PerfMonitor
from dlrover.python.master.servicer import create_master_service
from dlrover.python.scheduler.job import JobArgs
from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.master.elastic.job_manager import \
    create_job_manager


class RayElasticMaster(ElasticMaster):
    """
    RayElasticMaster is the ray based implementation of
    'dlrover.python.master.master::ElasticMaster', for elastic-agent training
    management.
    """

    def __init__(self, args: JobArgs):
        self.perf_monitor = PerfMonitor()
        self.job_manager = create_job_manager(args, self.perf_monitor)
        self.rdzv_managers: Dict[str, RendezvousManager] = {
            RendezvousName.ELASTIC_TRAINING: ElasticTrainingRendezvousManager(),
            RendezvousName.NETWORK_CHECK: NetworkCheckRendezvousManager(),
        }
        # self.diagnosis_manager = DiagnosisMaster(args)
        self._event_reporter = get_event_reporter()
        # self.job_metric_collector = self._create_metric_collector_if_needed(
        #     args
        # )
        self.sync_service = SyncService(self.job_manager)
        self._master_server = self._create_master_service(port, args)
        self._job_args = args
        self._exit_code = 0
        self._exit_reason = None
        self._job_evt = _master_evt.train_job(
            job_name=args.job_name, args=vars(args)
        )

    @property
    def master_server(self):
        return self._master_server

    def _create_master_service(self, port, params: JobArgs):
        # no need to start server for ray actor
        return create_master_service(
            port,
            None,
            self.job_manager,
            self.perf_monitor,
            self.rdzv_managers,
            self.diagnosis_manager,
            self.job_metric_collector,
            None,
            self.sync_service,
        )

    def prepare(self):
        if self.job_manager:
            self._add_node_event_callback()
            self.job_manager.start()

        threading.Thread(
            target=self._diagnose_job,
            name="job_diagnosing",
            daemon=True,
        ).start()

    def pre_check(self):
        pass

    def run(self):
        pass

    def stop(self):
        pass

    def request_stop(self, success, reason, msg=""):
        pass
