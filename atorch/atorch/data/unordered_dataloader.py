from torch._utils import ExceptionWrapper
from torch.utils.data import DataLoader, _utils
from torch.utils.data.dataloader import (
    _BaseDataLoaderIter,
    _DatasetKind,
    _MultiProcessingDataLoaderIter,
    _SingleProcessDataLoaderIter,
)


class _MultiProcessingUnorderedDataLoaderIter(_MultiProcessingDataLoaderIter):
    def __init__(self, *args, **kargs):
        super(_MultiProcessingUnorderedDataLoaderIter, self).__init__(*args, **kargs)

    def _next_data(self):
        while True:
            while self._rcvd_idx < self._send_idx and self._rcvd_idx in self._task_info:
                info = self._task_info[self._rcvd_idx]
                worker_id = info[0]
                if self._workers_status[worker_id]:  # is still active
                    break
                del self._task_info[self._rcvd_idx]
                self._rcvd_idx += 1

            if self._rcvd_idx >= self._send_idx:
                # no valid `self._rcvd_idx` is found (i.e., didn't break)
                if not self._persistent_workers:
                    self._shutdown_workers()
                raise StopIteration

            assert not self._shutdown and self._tasks_outstanding > 0
            idx, data = self._get_data()
            self._tasks_outstanding -= 1
            if self._dataset_kind == _DatasetKind.Iterable:
                # Check for _IterableDatasetStopIteration
                if isinstance(data, _utils.worker._IterableDatasetStopIteration):
                    if self._persistent_workers:
                        self._workers_status[data.worker_id] = False
                    else:
                        self._mark_worker_as_unavailable(data.worker_id)
                    self._try_put_index()
                    continue

            worker_idx = self._task_info[idx][0]
            del self._task_info[idx]
            return self._process_data(data, worker_idx)

    def _try_put_index(self, worker_idx=None):
        assert self._tasks_outstanding < self._prefetch_factor * self._num_workers

        try:
            index = self._next_index()
        except StopIteration:
            return
        if worker_idx is not None:
            worker_queue_idx = worker_idx
        else:
            for _ in range(self._num_workers):  # find the next active worker, if any
                worker_queue_idx = next(self._worker_queue_idx_cycle)
                if self._workers_status[worker_queue_idx]:
                    break
            else:
                # not found (i.e., didn't break)
                return

        self._index_queues[worker_queue_idx].put((self._send_idx, index))
        self._task_info[self._send_idx] = (worker_queue_idx,)
        self._tasks_outstanding += 1
        self._send_idx += 1

    def _process_data(self, data, worker_idx=None):
        self._rcvd_idx += 1
        self._try_put_index(worker_idx)
        if isinstance(data, ExceptionWrapper):
            data.reraise()
        return data


class UnorderedDataLoader(DataLoader):
    def __init__(self, *args, **kargs):
        super(UnorderedDataLoader, self).__init__(*args, **kargs)

    def _get_iterator(self) -> "_BaseDataLoaderIter":
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        else:
            self.check_worker_number_rationality()
            return _MultiProcessingUnorderedDataLoaderIter(self)
