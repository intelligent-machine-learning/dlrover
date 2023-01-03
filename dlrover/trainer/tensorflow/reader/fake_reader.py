
from dlrover.trainer.util.log_util import default_logger as logger

try:
    from dlrover.python.elastic_agent.sharding.client import ShardingClient
except Exception as e:
    from elasticai_api.common.data_shard_service import DataShardService as ShardingClient
    logger.info("import DataShardService as ShardingClient")


def build_data_shard_service(batch_size=1,
                             num_epochs=1,
                             dataset_size=1,
                             num_minibatches_per_shard=1,
                             dataset_name="iris_training_data"):
    sharding_client = ShardingClient(
        dataset_name=dataset_name,
        batch_size=batch_size,
        num_epochs=num_epochs,
        dataset_size=dataset_size,
        num_minibatches_per_shard=num_minibatches_per_shard,
    )
    return sharding_client



class FakeReader:
    def __init__(self, num_epochs=1, batch_size=64, enable_easydl=False):

        self._num_epochs = num_epochs
        self._batch_size = batch_size
        self.enable_easydl = enable_easydl
        self.data_shard_service = None
        self.count_data()
        self.data_shard_client = None
        self.build_data_shard_client()

    def build_data_shard_client(self):
        #if self.enable_easydl is True:
        self.data_shard_client = build_data_shard_service(
            batch_size=self._batch_size,
            num_epochs=1,
            dataset_size=self._data_nums,
            num_minibatches_per_shard=1,
            dataset_name="iris_training_data"
        )

    def count_data(self):
        self._data_nums = 2000

    def get_default_shard(self):
        return 1

    def _read_data(self):
        if self.data_shard_client is not None:
            shard = self.data_shard_client.fetch_shard()
            logger.info("getting data shard from easydl {}".format(shard))
        else:
            logger.info("getting data shard by default")
            shard = self.get_default_shard()
        data = self.get_data_by_shard(shard)
        yield data

    def get_data_by_shard(self, shard):
        return "1,1"

    def iterator(self):
        while True:
            for data in self._read_data():
                yield data