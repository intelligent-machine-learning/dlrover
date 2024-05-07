import torch

from atorch.common.log_utils import default_logger as logger

try:
    from elasticai_api.common.data_shard_service import RecordIndexService
    from elasticai_api.pytorch.DDP_controller import DDPController
except ImportError:
    logger.warning("Please install elasticai_api >= 1.4.2 .")

from atorch.common.log_utils import default_logger as logger
from atorch.data.elastic_dataset import ElasticDataset


class ElasticTrainer(object):
    def __init__(
        self,
        dataset_size,
        batch_size,
        num_workers,
        data_process_fn,
        shuffle,
        collate_fn,
        start_epoch,
        end_epoch,
        model,
        create_optim_from_ddp_model_fn,
        train_one_batch_fn,
    ):
        data_shard_service = RecordIndexService(
            batch_size=batch_size,
            dataset_size=dataset_size,
            num_epochs=(end_epoch - start_epoch + 1),
            shuffle=shuffle,
            dataset_name="",
        )
        dataset = ElasticDataset(data_shard_service, data_process_fn)
        self._dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

        self._start_epoch = start_epoch
        self._end_epoch = end_epoch
        self._model = model
        self._create_optim_from_ddp_model_fn = create_optim_from_ddp_model_fn
        self._train_one_batch_fn = train_one_batch_fn
        self._num_epochs = end_epoch - start_epoch + 1
        self._DDP_controller = DDPController(data_shard_service)

        if start_epoch < 0 or start_epoch > end_epoch:
            raise RuntimeError("start/end epoch set error.")

        self.batch_num_before_start_epoch = self._DDP_controller.batch_count_per_epoch * start_epoch

    def train(self):
        with self._DDP_controller.scope():
            last_epoch = -1
            for _, input_data in enumerate(self._dataloader):

                self._DDP_controller.elastic_DDP(
                    self._model,
                    func_def_optimizer=self._create_optim_from_ddp_model_fn,
                )

                epoch_index = self._DDP_controller.get_current_epoch() + self._start_epoch
                is_new_epoch = last_epoch != self._DDP_controller.get_current_epoch()

                self._DDP_controller.train_one_batch(
                    self._train_one_batch_fn,
                    ddp_model=self._DDP_controller.model,
                    optimizer=self._DDP_controller.optimizer,
                    data=input_data,
                    global_batch_idx=(
                        self._DDP_controller.global_completed_batch_num
                        + self._DDP_controller.get_local_rank()
                        + self.batch_num_before_start_epoch
                    ),
                    epoch_idx=epoch_index,
                    new_epoch=is_new_epoch,
                    new_ddp=self._DDP_controller.is_process_group_reinit(),
                )

                last_epoch = epoch_index if epoch_index > last_epoch else last_epoch

                if last_epoch >= self._end_epoch:
                    logger.info(
                        "ElasticTrainer runs {} epochs from epoch_{}"
                        "to epoch_{}, training exits now.".format(
                            self._num_epochs,
                            self._start_epoch,
                            self._end_epoch,
                        )
                    )
                    exit()
