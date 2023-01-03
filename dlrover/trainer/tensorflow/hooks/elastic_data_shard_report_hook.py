from tensorflow.python.training.session_run_hook import SessionRunHook
from dlrover.trainer.util.log_util import default_logger as logger

class ElasticDataShardReportHook(SessionRunHook):
    def __init__(self, sharding_client):
        self._sharding_client = sharding_client

    def after_run(self, run_context, run_values):
        try:
            self._sharding_client.report_batch_done()
            logger.info("report_batch_done")
        except Exception as ex:
            logger.error(
                "DLrover agent: report batch done failed: %s", ex
            )