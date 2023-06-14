import sys
from concurrent.futures import ThreadPoolExecutor, wait

import torch
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter, _SingleProcessDataLoaderIter

import atorch
from atorch.data import create_coworker_shm_context, get_sample_batch


def shm_collate_fn(batch):
    return batch[0]


def get_loader_size(dataset, **dataloader_args):
    batch_size = None
    drop_last = False
    sampler = None
    if "batch_size" in dataloader_args:
        batch_size = dataloader_args["batch_size"]
    if "drop_last" in dataloader_args:
        drop_last = dataloader_args["drop_last"]
    if "sampler" in dataloader_args:
        sampler = dataloader_args["sampler"]
    if sampler:
        sampler_size = len(dataloader_args["sampler"])
    else:
        sampler_size = len(dataset)
    if batch_size is None:
        batch_size = 1
    if drop_last:
        return sampler_size // batch_size
    else:
        return (sampler_size + batch_size - 1) // batch_size


class ShmDataset(IterableDataset):
    def __init__(self, shm_context):
        self.shm_context = shm_context

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id

        while True:
            data = self.shm_context.get_data(worker_id)
            if data is None:
                break
            yield data


class SizedShmDataset(ShmDataset):
    def __init__(self, shm_context, size):
        self.size = size
        super().__init__(shm_context)

    def __len__(self):
        return self.size


class _SingleProcessingShmDataLoaderIter(_SingleProcessDataLoaderIter):
    def __init__(self, loader, shm_context, executor):
        self.shm_context = shm_context
        self.executor = executor
        self.prefetch_task = None
        self.prefetch_data = None
        super().__init__(loader)

    def _reset(self, loader, first_iter=False):
        if self.prefetch_task is not None and self.prefetch_task.running():
            wait([self.prefetch_task])
        self.prefetch_task = None
        self.prefetch_data = None
        super()._reset(loader, first_iter=first_iter)

    def _next_data_prefetch_func(self):
        try:
            data = super()._next_data()
            if self.shm_context:
                self.shm_context.add_batch([data])
            self.prefetch_data = data
        except StopIteration:
            if self.shm_context:
                self.shm_context.add_batch(None)
            self.prefetch_data = None

    def _next_data(self):
        if self.prefetch_task is None:
            self.prefetch_task = self.executor.submit(self._next_data_prefetch_func)

        wait([self.prefetch_task])
        if self.prefetch_data is None:
            raise StopIteration
        data = self.prefetch_data
        self.prefetch_task = self.executor.submit(self._next_data_prefetch_func)
        return data


class _MultiProcessingShmDataLoaderIter(_MultiProcessingDataLoaderIter):
    def __init__(self, loader, shm_context, executor):
        self.shm_context = shm_context
        self.executor = executor
        self.prefetch_task = None
        self.prefetch_data = None
        super().__init__(loader)

    def _reset(self, loader, first_iter=False):
        if self.prefetch_task is not None and self.prefetch_task.running():
            wait([self.prefetch_task])
        self.prefetch_task = None
        self.prefetch_data = None
        super()._reset(loader, first_iter=first_iter)

    def _next_data_prefetch_func(self):
        try:
            data = super()._next_data()
            if self.shm_context:
                self.shm_context.add_batch([data])
            self.prefetch_data = data
        except StopIteration:
            if self.shm_context:
                self.shm_context.add_batch(None)
            self.prefetch_data = None

    def _next_data(self):
        if self.prefetch_task is None:
            self.prefetch_task = self.executor.submit(self._next_data_prefetch_func)

        wait([self.prefetch_task])
        if self.prefetch_data is None:
            raise StopIteration
        data = self.prefetch_data
        self.prefetch_task = self.executor.submit(self._next_data_prefetch_func)
        return data


class ShmDataloader(DataLoader):
    def __init__(
        self,
        dataset,
        dataloader_args,
        rank=None,
        group_size=None,
        shm_data_size=100,
        io_timeout=30,
        initialize_timeout=300,
        shm_name_prefix=None,
        need_sync_write=True,
    ):
        """
        TODO: support uneven data from coworker using UnorderedDataLoader
        """
        self.iter_count = 0
        self.shm_context = None
        self.rank = rank
        self.group_size = group_size
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.use_coworker = atorch.distributed.use_coworker()
        self.shm_context = create_coworker_shm_context(
            dataset=dataset,
            dataloader_args=dataloader_args,
            rank=rank,
            group_size=group_size,
            shm_data_size=shm_data_size,
            io_timeout=io_timeout,
            initialize_timeout=initialize_timeout,
            shm_name_prefix=shm_name_prefix,
            need_sync_write=need_sync_write,
        )
        if self.use_coworker or rank != 0:
            assert not atorch.distributed.is_coworker(), "ShmDataloader only called by worker for coworker"
            if self.use_coworker or not hasattr(dataset, "__len__"):
                # worker in coworker case does not know len(dataloader)
                iter_dataset = ShmDataset(self.shm_context)
            else:
                # model parallel case supports len(dataloader)
                loader_size = get_loader_size(dataset, **dataloader_args)
                iter_dataset = SizedShmDataset(self.shm_context, loader_size)
            super().__init__(
                dataset=iter_dataset,
                collate_fn=shm_collate_fn,
                batch_size=1,
                num_workers=atorch.distributed.coworker_num_per_node() if self.use_coworker else 1,
                pin_memory=torch.cuda.is_available(),
            )
        else:
            # model parallel dataloader's master (rank==0)
            assert rank == 0
            assert group_size is not None
            super().__init__(dataset, **dataloader_args)

    def __iter__(self):
        if self.iter_count > 0 and self.shm_context:
            self.shm_context.reset()
        self.iter_count += 1
        if self.use_coworker or self.rank != 0:  # worker
            shm_context = None
        else:
            shm_context = self.shm_context
        if self.num_workers == 0:
            return _SingleProcessingShmDataLoaderIter(self, shm_context, self.executor)
        else:
            self.check_worker_number_rationality()
            return _MultiProcessingShmDataLoaderIter(self, shm_context, self.executor)

    def __del__(self):
        if self.shm_context is not None:
            self.shm_context.tear_down()
            self.shm_context = None
        if self.executor is not None:
            self.executor.shutdown()
            self.executor = None

    def stop(self):
        if self.shm_context is not None:
            self.shm_context.set_stop_status()


def create_shm_dataloader(
    dataset,
    dataloader_args,
    coworker_data_process_func=None,
    shm_data_size=100,
    io_timeout=30,
    initialize_timeout=300,
    shm_name_prefix=None,
    coworker_wait_worker_read=True,
    coworker_wait_worker_read_timeout=60,
):
    """
    if type(shm_data_size)==list:
        create multiple shm_dataloaders with corresponding size.
        dataset, dataloader_args should be a list of same length with shm_data_size.
        shm_name_prefix must be a list of strings with different prefix names.
        coworker_data_process_func takes inputs as list of shm_context instead of shm_context.
        return a list of ShmDataloader
    """
    num_shms = 1
    if type(shm_data_size) == list:
        num_shms = len(shm_data_size)
        assert type(dataset) == list and len(dataset) == num_shms, f"dataset should be a list with length {num_shms}"
        assert (
            type(dataloader_args) == list and len(dataloader_args) == num_shms
        ), f"dataloader_args should be a list with length {num_shms}"
        assert (
            type(shm_name_prefix) == list and len(shm_name_prefix) == num_shms
        ), f"shm_name_prefix should be a list with length {num_shms}"
    else:
        dataset = [dataset]
        dataloader_args = [dataloader_args]
        shm_data_size = [shm_data_size]
        shm_name_prefix = [shm_name_prefix]
    if atorch.distributed.is_coworker():
        shm_contexts = []
        for idx in range(num_shms):
            sample_batch = get_sample_batch(dataset[idx], dataloader_args[idx])
            shm_context = create_coworker_shm_context(
                sample_batch=sample_batch,
                shm_data_size=shm_data_size[idx],
                shm_name_prefix=shm_name_prefix[idx],
                io_timeout=io_timeout,
                initialize_timeout=initialize_timeout,
            )
            shm_contexts.append(shm_context)
        coworker_data_process_func(shm_contexts if num_shms > 1 else shm_contexts[0])
        for idx in range(num_shms):
            shm_contexts[idx].tear_down(
                master_wait_for_worker=coworker_wait_worker_read, wait_timeout=coworker_wait_worker_read_timeout
            )
        sys.exit(0)
    else:
        dataloaders = []
        for idx in range(num_shms):
            dataloader = ShmDataloader(
                dataset=dataset[idx],
                dataloader_args=dataloader_args[idx],
                shm_data_size=shm_data_size[idx],
                io_timeout=io_timeout,
                initialize_timeout=initialize_timeout,
                shm_name_prefix=shm_name_prefix[idx],
            )
            dataloaders.append(dataloader)
        return dataloaders if num_shms > 1 else dataloaders[0]
