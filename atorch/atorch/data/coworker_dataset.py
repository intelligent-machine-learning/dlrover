import multiprocessing as mp
import random
import threading

import torch
from torch.utils.data._utils.collate import default_collate

from atorch.common.log_utils import default_logger as logger
from atorch.data.unordered_dataloader import UnorderedDataLoader
from atorch.distributed.distributed import coworker_size, rank, world_size


class CoworkerDataset(torch.utils.data.Dataset):
    """
    A Dataset that dispatch some preprocessing task to
    coworkers(CPU Pod in cluster).
    """

    def __init__(
        self,
        process_fn,
        dataset_size,
        pre_process_fn=None,
        num_workers=0,
        num_rpc_workers=0,
        coworker_process_percentage=0,
        custom_collate_fn=None,
        num_rpc_fuse=4,
        need_padding_for_none=False,
    ):
        """
        Args:
            process_fn: Transformation function, executes either locally or in
                coworker.
            dataset_size: The number of data that dataset has.
            pre_process_fn: Optional process before process_fn, executed
                locally.
            batch_size: Training batch size.
            num_workers: Number of dataloader workers.
            num_rpc_workers: Number of dataloader workers calling CoWorkers by
                RPC for preprocessing.
            coworker_process_percentage: percentage of data to be
                preprocessed by coworkers.
            shuffle: Set to True to have the data reshuffled at every epoch.
                If num_rpc_workers > 0, shuffle must be True.
            custom_collate_fn: User-defined collate_fn.
            num_rpc_fuse: Call RPC to process num_rpc_fuse data at a time to
                reduce RPC overhead and increase CoWorkers' CPU utilization.
            need_padding_for_none: During preprocessing, some data may fail to
                be preprocessed. As a result, the number of preprocessed data
                is less than batch_size. For data that fails to preprocess,
                if process_fn doesnot return any objects or returns None,
                `build_coworker_dataloader` will randomly select and copy
                the processed data from this batch, and put it into the return
                value so that the returned data is equal to batch_size.
        """
        super(CoworkerDataset, self).__init__()
        self.process_fn = process_fn
        self.dataset_size = dataset_size
        self.pre_process_fn = pre_process_fn
        self.num_workers = num_workers
        self.num_rpc_workers = num_rpc_workers
        self.num_local_workers = 0
        self.coworker_process_percentage = coworker_process_percentage
        self.coworkers = self.init_coworkers_for_dataloader()
        self.num_coworkers = len(self.coworkers)
        self.num_workers_needs_rpc = 0
        self.coworkers_distribution = self.init_coworkers_for_workers()
        self.custom_collate_fn = custom_collate_fn
        self.num_rpc_fuse = num_rpc_fuse
        self.need_padding_for_none = need_padding_for_none
        self.indices_queues, self.results_queues = None, None
        self.init_cowokers_utils()

    def __getitem__(self, index):
        return index

    def __len__(self):
        return self.dataset_size

    def get_collate_fn(self):
        if self.custom_collate_fn is not None:
            return self.custom_collate_fn
        else:
            return default_collate

    def init_coworkers_for_dataloader(self):
        logger.info("{} coworkers will be used to preprocess.".format(coworker_size()))
        if coworker_size() == 0:
            logger.warning(
                "Not found any coworkers. Set `num_rpc_workers` and "
                "`coworker_process_percentage` to 0. Only use local"
                " CPUs to preprocess data."
            )
            self.coworker_process_percentage = 0
            self.num_rpc_workers = 0
            self.num_local_workers = self.num_workers
            return []
        if self.num_workers < self.num_rpc_workers:
            logger.error(
                "num_workers({}) smaller than num_rpc_workers({}).".format(self.num_workers, self.num_rpc_workers)
            )
        elif self.num_workers == self.num_rpc_workers:
            logger.warning(
                "num_workers({}) equals num_rpc_workers({}). All the workers"
                " process data by CoWorkers.".format(self.num_workers, self.num_rpc_workers)
            )
            self.num_local_workers = 0
        else:
            self.num_local_workers = self.num_workers - self.num_rpc_workers
        num_training_processes = world_size() - coworker_size()
        if coworker_size() < num_training_processes:
            if num_training_processes % coworker_size() != 0:
                logger.warning(
                    "There are {} training processes and {} coworkers. The"
                    " number of training processes is preferably divisible"
                    " by the number of coworkers.".format(num_training_processes, coworker_size())
                )
            coworker = [str(rank() % coworker_size() + num_training_processes)]
            return coworker
        elif coworker_size() >= num_training_processes:
            if coworker_size() % num_training_processes != 0:
                logger.warning(
                    "There are {} training processes and {} coworkers. The"
                    " number of coworkers is preferably divisible"
                    " by the number of training processes.".format(num_training_processes, coworker_size())
                )
            coworkers = []
            for coworker_rank in range(num_training_processes, world_size()):
                train_process_rank = coworker_rank % num_training_processes
                if train_process_rank == rank():
                    coworkers.append(str(coworker_rank))
            return coworkers

    def init_coworkers_for_workers(self):
        if self.num_rpc_workers > 0:
            self.num_workers_needs_rpc = self.num_rpc_workers
        elif self.coworker_process_percentage > 0:
            self.num_workers_needs_rpc = self.num_workers
        if self.num_workers_needs_rpc == 0:
            return
        coworker_distribution = [[] for _ in range(self.num_workers_needs_rpc)]
        if self.num_coworkers < self.num_workers_needs_rpc:
            if self.num_workers_needs_rpc % self.num_coworkers != 0:
                logger.warning(
                    "There are {} dataloader workers that need coworkers and"
                    " {} coworkers. The number of dataloader workers that need"
                    " coworkers is preferably divisible by the number of "
                    "coworkers.".format(self.num_workers_needs_rpc, self.num_coworkers)
                )
            coworker_idx = 0
            for i in range(self.num_workers_needs_rpc):
                coworker_distribution[i].append(self.coworkers[coworker_idx])
                coworker_idx += 1
                if coworker_idx == self.num_coworkers:
                    coworker_idx = 0
            return coworker_distribution
        else:
            if self.num_coworkers % self.num_workers_needs_rpc != 0:
                logger.warning(
                    "There are {} coworkers and {} dataloader workers that"
                    " need coworkers. The number of coworkers is preferably"
                    " divisible by the number of dataloader workers that"
                    " need coworkers.".format(self.num_coworkers, self.num_workers_needs_rpc)
                )
            for i, coworker in enumerate(self.coworkers):
                coworker_distribution[i % self.num_workers_needs_rpc].append(coworker)
            return coworker_distribution

    def init_cowokers_utils(self):
        if self.num_coworkers == 0 or self.num_workers_needs_rpc == 0:
            return
        self.indices_queues, self.results_queues = self.create_queues()
        self.multithreading_rpc()

    def create_queues(self):
        indices_queues, results_queues = [], []
        for _ in range(self.num_workers_needs_rpc):
            indices_queue = mp.Queue()
            results_queue = mp.Queue()
            indices_queues.append(indices_queue)
            results_queues.append(results_queue)
        return indices_queues, results_queues

    @staticmethod
    def process(func, *args):
        return func(*args)

    @staticmethod
    def process_multi(func, *args):
        utils = args[0]
        return [func(util) for util in utils]

    def rpc_coworker(self, indices_queue, results_queue, coworkers):
        while True:
            futures = []
            data_indices = indices_queue.get()
            coworker_index = 0
            for i in range(0, len(data_indices), self.num_rpc_fuse):
                # use `slice` to avoid pep8 E203
                indices_to_rpc = data_indices[slice(i, i + self.num_rpc_fuse)]
                if self.pre_process_fn is not None:
                    pre_process_results = [self.pre_process_fn(index) for index in indices_to_rpc]
                else:
                    pre_process_results = indices_to_rpc
                coworker_name = coworkers[coworker_index]
                future = torch.distributed.rpc.rpc_async(
                    coworker_name,
                    self.process_multi,
                    args=(self.process_fn, pre_process_results),
                    timeout=300,
                )
                futures.append(future)
                coworker_index += 1
                if coworker_index == len(coworkers):
                    coworker_index = 0
            results = []
            for future in futures:
                rpc_results = future.wait()
                for data in rpc_results:
                    results.append(data)
            results_queue.put(results)

    def multithreading_rpc(self):
        for i in range(self.num_workers_needs_rpc):
            coworkers_for_worker = self.coworkers_distribution[i]
            indices_queue = self.indices_queues[i]
            results_queue = self.results_queues[i]
            t = threading.Thread(
                target=self.rpc_coworker,
                args=(indices_queue, results_queue, coworkers_for_worker),
                daemon=True,
            )
            t.start()

    def preprocess_at_local(self, batch, start, end):
        results = []
        for i in range(start, end):
            if self.pre_process_fn is not None:
                res = self.process(self.process_fn, self.pre_process_fn(batch[i]))
            else:
                res = self.process(self.process_fn, batch[i])
            results.append(res)
        return results

    def collate_fn(self, batch):
        worker_id = torch.utils.data.get_worker_info().id
        batch_size = len(batch)

        if worker_id < self.num_local_workers:
            # workers whose id smaller than self.num_local_workers
            # use local CPUs to process all data that they are
            # responsible for.
            local_size = batch_size
        elif self.num_rpc_workers > 0:
            # workers whose id larger than self.num_local_workers
            # use CoWorkers to process all data that they are
            # responsible for.
            local_size = 0
        elif self.coworker_process_percentage > 0:
            # All workers use both local CPUs and CoWorkers to process every
            # batch of data.
            local_size = batch_size - int(self.coworker_process_percentage * batch_size)
        else:
            # Only use local CPUs to process data.
            local_size = batch_size

        remote_size = batch_size - local_size
        queue_id = worker_id % max(1, self.num_workers_needs_rpc)
        if remote_size > 0:
            # request remote process
            self.indices_queues[queue_id].put([batch[i] for i in range(remote_size)])
        if local_size > 0:
            # process locally
            results = self.preprocess_at_local(batch, remote_size, batch_size)
        else:
            results = []
        if remote_size > 0:
            # get remote process results
            remote_results = self.results_queues[queue_id].get()
            results.extend(remote_results)

        if self.need_padding_for_none:
            # Whether use CoWorkers or not, there maybe `None`
            # in the `results`. Change `None` to valid data.
            num_none = results.count(None)

            if num_none > 0:
                for i, data in enumerate(results):
                    if data is None:
                        new_data = None
                        while new_data is None:
                            img_idx = random.randint(0, self.__len__() - 1)
                            if self.pre_process_fn is not None:
                                new_data = self.process(
                                    self.process_fn,
                                    self.pre_process_fn(img_idx),
                                )
                            else:
                                new_data = self.process(self.process_fn, img_idx)
                        results[i] = new_data
        res = self.get_collate_fn()(results)
        return res


def build_coworker_dataloader(
    process_fn,
    dataset_size,
    pre_process_fn=None,
    num_workers=0,
    num_rpc_workers=0,
    coworker_process_percentage=0,
    batch_size=1,
    shuffle=True,
    custom_collate_fn=None,
    num_rpc_fuse=4,
    need_padding_for_none=False,
    **kwargs,
):
    """
    `build_coworker_dataloader` can use free CPU Pods(CoWorkers) in a cluster
     to accelerate processing data. It has two ways to use CoWorkers:
    (1) If num_rpc_workers > 0, num_rpc_workers dataloader workers will use
        CoWorkers and (num_workers - num_rpc_workers) workers will use local
        CPUs. If a dataloader worker use CoWorkers, it will use CoWorkers to
        process all the data it is responsible for.
    (2) If coworker_process_percentage > 0, all dataloader workers will use
        both local CPUs and CoWorkers to process images. When a worker is
        processing a batch of data, `batch_size * coworker_process_percentage`
        will be processed by CoWorkers, others will be processed by local
        CPUs.
    `num_rpc_workers` and `coworker_process_percentage` are mutually exclusive.
    If both `num_rpc_workers` and coworker_process_percentage are bigger than
    0, `build_coworker_dataloader` will set coworker_process_percentage to 0.

    Args:
        process_fn: Transformation function, executes either locally or in
            coworker.
        dataset_size: The size of dataset
        pre_process_fn: Optional process before process_fn, executed locally.
        batch_size: Training batch size
        num_workers: Number of dataloader workers
        num_rpc_workers: Number of dataloader workers calling CoWorkers by
            RPC.
        coworker_process_percentage: Percentage of data to be
            preprocessed by coworkers.
        shuffle: Set to True to have the data reshuffled at every epoch.
        custom_collate_fn: User-defined collate_fn
        num_rpc_fuse: Call RPC to process `num_rpc_fuse` data at a time to
            reduce RPC overhead and increase CoWorkers' CPU utilization.
        need_padding_for_none: During preprocessing, some data may fail to
            be preprocessed. As a result, the number of preprocessed data
            is less than batch_size. For data that fails to preprocess,
            if process_fn doesnot return any objects or returns None,
            `build_coworker_dataloader` will randomly select and copy the
            processed data from this batch, and put it into the return value
            so that the returned data is equal to batch_size.
        kwargs: Other DataLoader's arguments, including: pin_memory,
            drop_last, persistent_workers
    """
    if num_rpc_workers > 0 and coworker_process_percentage > 0:
        logger.warning(
            "Found that num_rpc_workers({})"
            " and coworker_process_percentage({}) are both bigger than"
            " 0. Set coworker_process_percentage to 0".format(num_rpc_workers, coworker_process_percentage)
        )
        coworker_process_percentage = 0
    dataset = CoworkerDataset(
        process_fn,
        dataset_size,
        pre_process_fn=pre_process_fn,
        num_workers=num_workers,
        num_rpc_workers=num_rpc_workers,
        coworker_process_percentage=coworker_process_percentage,
        custom_collate_fn=custom_collate_fn,
        num_rpc_fuse=num_rpc_fuse,
        need_padding_for_none=need_padding_for_none,
    )
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=shuffle)
    if not shuffle and num_rpc_workers > 0:
        logger.warning("When `num_rpc_workers`(=={}) > 0, the data order may be " "changed even though shuffle=False.")
    if num_rpc_workers > 0:
        DataLoader = UnorderedDataLoader
    else:
        DataLoader = torch.utils.data.DataLoader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
        sampler=sampler,
        **kwargs,
    )
    return data_loader, sampler
