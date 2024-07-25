import multiprocessing as mp
import os
import queue

import grpc
from torch.utils.data import DataLoader, Dataset, get_worker_info

from atorch.common.log_utils import default_logger as logger
from atorch.data.elastic_dataset import SimpleElasticDataset
from atorch.distributed.distributed import (
    coworker_addrs,
    coworker_size,
    gpu_pod_addrs,
    nproc_per_node,
    rank,
    world_size,
)


def get_elastic_dataloader(
    dataset_size,
    num_epochs,
    batch_size,
    data_process_fn,
    shuffle=True,
    num_workers=0,
    collate_fn=None,
    pin_memory=False,
    drop_last=False,
    persistent_workers=False,
    num_minibatches_per_shard=2,
    **dataloader_kwargs,
):
    """
    This function will be called by `build_coworker_dataloader_with_elasticdl` and create a dataloader on
    coworkers using dynamic sharding.

    Args:
        dataset_size: The size of dataset(number of data).
        num_epochs: Number of training epochs.
        batch_size: Training batch size.
        data_process_fn: Data transformation function.
        shuffle: Set to True to have the data reshuffled at every epoch.
        num_workers: Same as `num_workers` of `torch.utils.data.DataLoader`.
        collate_fn: Same as `collate_fn` of `torch.utils.data.DataLoader`.
        pin_memory: Same as `pin_memory` of `torch.utils.data.DataLoader`.
        drop_last: Same as `drop_last` of `torch.utils.data.DataLoader`.
        persistent_workers: Same as `persistent_workers` of `torch.utils.data.DataLoader`.
        edl_master_addr: The address of Dynamic Data Shard Service, in form <host>:<port> (e.g. 1.2.3.4:50001).
            No need to set this parameter if Worker nodes are launched by EDL Master node. The default port of the
            Dynamic Data Sharding Servic is 50001.
        num_minibatches_per_shard: Number of mini-batches per data shard. Lets say num_minibatches_per_shard=2 and
            batch_size=4, then data shard will be [(0,7),(8,15), ..., (k, k+batch_size*num_minibatches_per_shard)]
        dataloader_kwargs: Other args supported by `torch.utils.data.DataLoader`.

    Returns:
        A dataloader
    """
    edl_master_addr = dataloader_kwargs.pop("edl_master_addr", "")
    if edl_master_addr:
        # ElasticDataset needs to connect the dlrover master by GPRC.
        os.environ["DLROVER_MASTER_ADDR"] = edl_master_addr
    actual_num_epochs = num_epochs + 1
    # Use `num_epochs + 1` instead of `num_epochs`. When using coworkers, at the end of each epoch, if every coworkers
    # get less than a batch of data, the Simple Dataloader on gpu pods cannot get any data. When using elastic
    # training, some workers may have less indices than others. Use `num_epochs + 1` to prevent these situation.
    elastic_dataset = SimpleElasticDataset(
        name="training-data",
        data_process_fn=data_process_fn,
        dataset_size=dataset_size,
        batch_size=batch_size,
        epochs=actual_num_epochs,
        shuffle=shuffle,
        num_minibatches_per_shard=num_minibatches_per_shard,
    )

    elastic_dataloader = DataLoader(
        dataset=elastic_dataset,
        batch_size=batch_size,
        pin_memory=pin_memory,
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=collate_fn,
        persistent_workers=persistent_workers,
        **dataloader_kwargs,
    )

    return elastic_dataloader


def _simple_data_loader_worker_init_fn(worker_id):
    """
    When Dataloader's num_workers > 0, create grpc clients in the
    subprocesses.
    """
    from atorch.service.rpc_clients import create_coworker_rpc_client, create_data_info_rpc_client

    worker_info = get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        dataset._data_info_rpc_client = create_data_info_rpc_client(dataset._data_info_service_ip_and_port)
        dataset._coworker_rpc_clients = create_coworker_rpc_client(dataset._coworker_addrs)


class SimpleCoworkerDataset(Dataset):
    """
    `SimpleCoworkerDataset` has a data_info_rpc_client and
    coworker_rpc_clients. data_info_rpc_client get data info from Data Info
    Service on worker0. There are coworker address and batch_num in the data
    info. coworker_rpc_clients send a request to coworkers whose addresses in
    the data info and get data from coworkers.

    When Dataloader's num_workers > 0, create grpc clients in the
    subprocesses. Otherwise, create grpc clients in the main process.
    """

    def __init__(self, size, data_info_service_ip_and_port, batch_num=1, num_workers=0):
        from atorch.service.rpc_clients import create_coworker_rpc_client, create_data_info_rpc_client

        self._size = size
        self._data_info_service_ip_and_port = data_info_service_ip_and_port
        self._batch_num = batch_num
        self._coworker_addrs = list(coworker_addrs().values())[1:]
        self._data_info_rpc_client = (
            create_data_info_rpc_client(self._data_info_service_ip_and_port) if num_workers == 0 else None
        )
        self._coworker_rpc_clients = create_coworker_rpc_client(self._coworker_addrs) if num_workers == 0 else None
        self._data_info_queue = mp.Queue() if self._batch_num > 1 else None

    def __len__(self):
        return self._size

    def __getitem__(self, index):
        """
        When data_info.batch_num > 1, data_info will be put into
        self._data_info_queue. In the next iteration, __getitem__ will fetch
        data info from self._data_info_queue instead of data info service.
        """
        data_info = None
        if self._data_info_queue is not None:
            try:
                data_info = self._data_info_queue.get_nowait()
            except queue.Empty:
                pass
        if data_info is None:
            data_info = self._data_info_rpc_client.get_data_info()
        try:
            batch_data = self._coworker_rpc_clients[data_info.coworker_addr].get_batch_data()
        except grpc.RpcError as rpc_error:
            logger.warning("Call {} failure: {}".format(data_info.coworker_addr, rpc_error))
            return self.__getitem__(0)
        data_info.batch_num -= 1
        if data_info.batch_num > 0:
            self._data_info_queue.put(data_info)
        return batch_data


def get_simple_dataloader(
    size,
    data_info_service_ip_and_port,
    num_workers,
    pin_memory=False,
    persistent_workers=False,
    batch_num=1,
    **kwargs,
):
    simple_coworker_dataset = SimpleCoworkerDataset(
        size, data_info_service_ip_and_port, batch_num, num_workers=num_workers
    )
    persistent_workers = num_workers > 0
    # Data getting from Coworker rpc service has already been batched.
    # Pass `None` to parameter `batch_size`.
    dataloader = DataLoader(
        dataset=simple_coworker_dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        worker_init_fn=_simple_data_loader_worker_init_fn,
        **kwargs,
    )
    return dataloader


def build_coworker_dataloader_with_elasticdl(
    dataset_size,
    num_epochs,
    batch_size,
    num_workers,
    data_process_fn,
    shuffle=True,
    pin_memory=False,
    drop_last=False,
    persistent_workers=False,
    custom_collate_fn=None,
    batch_num=1,
    data_queue_max_size=None,
    **kwargs,
):
    """
    `build_coworker_dataloader_with_elasticdl` create different Dataloaders on
    different pods.

    It creates Elastic Dataloaders on every coworkers(cpu pods)
    except coworker0. Elastic Dataloaders get data index from Dynamic Data
    Sharding Service, read data by index, and preprocess data. Then
    preprocessed data will be put into a batch data queue. On coworker1 ~
    coworkerN-1, coworker rpc services will be initialzed. Coworker rpc
    service shares the batch data queue with Elastic Dataloader. Training
    processes on gpu pods will send requests to coworker rpc service to get
    data.

    On gpu pods, `build_coworker_dataloader_with_elasticdl` create a
    Simple Dataloader on every training processes(workers). Simple Dataloaders
    get data from coworkers and output data to models.

    Args:
        dataset_size: The size of dataset
        num_epochs: Number of training epochs
        batch_size: Training batch size
        num_workers: Number of dataloader workers(subprocess) that proprecess
            training data
        data_process_fn: Transformation function that is executed on coworkers
        shuffle: Set to True to have the data reshuffled at every epoch.
        pin_memory: If True, the data loader will copy Tensors into CUDA
            pinned memory before  returning them.
        drop_last: set to True to drop the last incomplete batch, if the
            dataset size is not divisible by the batch size. If False and the
            size of dataset is not divisible by the batch size, then the last
            batch will be smaller.
        custom_collate_fn: User-defined collate_fn.
        batch_num: Every time `batch_num` data is put into batched data queue,
            gpu_pod_rpc_client will call `report_data_info`.
        data_queue_max_size: The max size of batched data queue.
        kwargs: Other DataLoader's arguments, including: prefetch_factor,
            timeout, worker_init_fn, etc.

    Returns:
        On coworkers, returns None.
        On gpu pods, returns a Simple Dataloader.
    """
    if coworker_size() is None:
        raise ValueError("Cannot get the number of coworkers(atorch.coworker_size() is None).")
    if coworker_size() < 2:
        raise RuntimeError("There must be at least 2 coworkers.")
    training_world_size = world_size() - coworker_size()
    if rank() > training_world_size:
        # On coworker 1 ~ n
        coworker_addrs_dict = coworker_addrs()
        if coworker_addrs_dict is None:
            raise ValueError("Cannot get the addresses and ports of coworkers.")
        coworker0_rank = training_world_size
        coworker0_ip_and_port = coworker_addrs_dict[coworker0_rank]
        elastic_dataloader = get_elastic_dataloader(
            dataset_size,
            num_epochs,
            batch_size,
            data_process_fn,
            shuffle=shuffle,
            num_workers=os.cpu_count(),
            collate_fn=custom_collate_fn,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            edl_master_addr=coworker0_ip_and_port,
            num_minibatches_per_shard=training_world_size,
            **kwargs,
        )
        from atorch.service.coworker_data_service import create_coworker_rpc_service
        from atorch.service.rpc_clients import create_gpu_pod_rpc_clients

        cur_coworker_ip_and_port = coworker_addrs_dict[rank()]
        port = cur_coworker_ip_and_port.split(":")[1]
        # Training speed may be slower than data preprocessing. Set the
        # maximum length of the batched_data_queue to prevent OOM errors
        # caused by not consuming data in time. Assuming that users only have
        # two coworkers, coworker 0 for running data sharding service,
        # coworker 1 for processing data. Coworker 1 should provide at least
        # batch_num * num_gpus data per iteration.
        if data_queue_max_size is None:
            min_capacity = batch_num * training_world_size
            data_queue_max_size = 16 if 16 > min_capacity else min_capacity
        batched_data_queue = mp.Queue(data_queue_max_size)
        # Create and start Coworker rpc service on coworker 1 ~ n
        coworker_rpc_server = create_coworker_rpc_service(port, batched_data_queue)

        coworker_rpc_server.start()
        logger.info("CoWorker RPC Server has started")

        data_info_service_ip_and_ports = []
        for global_rank, ip in gpu_pod_addrs().items():
            # data info services are created on workers whose local_rank == 0.
            if global_rank % nproc_per_node() == 0:
                data_info_service_ip_and_ports.append(ip)
        gpu_pod_rpc_clients = create_gpu_pod_rpc_clients(data_info_service_ip_and_ports)
        num_gpu_pods = len(gpu_pod_rpc_clients)
        # In order to make the data distribution more balanced, different
        # coworkers use different initial cur_gpu_pod_id.
        cur_gpu_pod_id = rank() % num_gpu_pods
        num_batches_per_communication = batch_num
        for data in elastic_dataloader:
            batched_data_queue.put(data)
            num_batches_per_communication -= 1
            if num_batches_per_communication > 0:
                continue
            else:
                num_batches_per_communication = batch_num
            gpu_pod_rpc_clients[cur_gpu_pod_id].report_data_info(cur_coworker_ip_and_port, batch_num)
            cur_gpu_pod_id += 1
            if cur_gpu_pod_id == num_gpu_pods:
                cur_gpu_pod_id = 0
    elif rank() < training_world_size:
        gpu_pod_addresses = gpu_pod_addrs()
        if gpu_pod_addresses is None:
            raise ValueError("Cannot get the addresses of gpu pods.")
        # gpu pod workers
        global_rank_of_local_worker0 = rank() // nproc_per_node() * nproc_per_node()
        data_info_service_ip_and_port = gpu_pod_addresses[global_rank_of_local_worker0]
        size = int(dataset_size / training_world_size / batch_size)
        if drop_last is False and dataset_size % (training_world_size * batch_size) != 0:
            size = size + 1
        simple_dataloader = get_simple_dataloader(
            size,
            data_info_service_ip_and_port,
            num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            batch_num=batch_num,
            **kwargs,
        )
        return simple_dataloader
