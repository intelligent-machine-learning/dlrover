from atorch.common.log_utils import default_logger as logger

try:
    from elasticai_api.pytorch.DDP_controller import DDPController
except ImportError:
    logger.warning("Please install elasticai_api >= 1.4.2 .")


class ElasticController(DDPController):
    def __init__(self, data_shard_service):
        super(ElasticController, self).__init__(data_shard_service)


def elastic_controller(data_shard_service):
    return ElasticController(data_shard_service)
