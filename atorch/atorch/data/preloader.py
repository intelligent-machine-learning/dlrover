import collections

import torch

from atorch.common.util_func import data_to_device


class GpuPreLoader(object):
    def __init__(
        self,
        dataloader,
        device="cuda",
        mask=None,
        post_processing=None,
        manual_preload=False,
    ):
        """
        This is a wrapper for DataLoader to support gpu preload.
        Args:
          dataloader: the original dataloader to wrap
          device: prefetch gpu device, such as "cuda", "cuda:0", "1"
          mask: mask is a list of boolean if not None. set mask if not all
            data items are prefectched. Each data should
            be a list/tuple/dict of tensors, with the same length as
            the mask list. mask[i] indicates if i-th data  item should
            be prefetched to device.
          post_processing: if not None, it is a function to generate additional
            data from data. The input of this function is the data from
            dataloader.  Then, iter will return a 2-item tuple, with 1st item
            as the original data from dataloader, and the 2nd item as the
            output of the post_processing.
          manual_preload: if True, except the 1st data, user needs to call
            preload manually.
        """
        self.dataloader = dataloader
        self.loader = None
        self.device = device
        self.mask = mask
        self.post_processing = post_processing
        self.data_event = None
        self.processing_event = None
        self.cur_processing_event = None
        self.manual_preload = manual_preload
        self.preloaded = False

        use_gpu = True
        if "cuda" not in device and not device.isnumeric():
            use_gpu = False
        if mask is not None and not any(mask):
            use_gpu = False
        if use_gpu:
            self.stream = torch.cuda.Stream(self.device)
        else:
            self.stream = None
        self.gpu_tensors = []

    def add_to_gpu_tensors(self, data):
        if isinstance(data, collections.abc.Mapping):
            for key in data:
                self.add_to_gpu_tensors(data[key])
            return
        elif isinstance(data, collections.abc.Sequence):
            for v in data:
                self.add_to_gpu_tensors(v)
            return

        self.gpu_tensors.append(data)

    def add_gpu_tensors(self, data):
        if isinstance(data, collections.abc.Mapping):
            for key in data:
                self.add_gpu_tensors(data[key])
        elif isinstance(data, collections.abc.Sequence):
            for v in data:
                self.add_gpu_tensors(v)
        else:
            self.gpu_tensors.append(data)

    def to_gpu(self, data):
        device_data = data_to_device(data, self.device, non_blocking=True)
        self.add_gpu_tensors(device_data)
        return device_data

    def to_gpu_with_mask(self, data):
        if isinstance(data, collections.abc.Mapping):
            return {key: self.to_gpu(data[key] if self.mask[idx] else data[key]) for idx, key in enumerate(data)}
        if isinstance(data, collections.abc.Sequence):
            return [self.to_gpu(v) if self.mask[idx] else v for idx, v in enumerate(data)]
        raise Exception("mask only supports list/tuple/dict data type")

    def preload(self):
        if self.preloaded:
            return

        self.preloaded = True

        try:
            data = next(self.loader)
        except StopIteration:
            self.next_data = None
            return

        if self.stream is None:
            if self.post_processing is None:
                self.next_data = data
            else:
                output = self.post_processing(data)
                self.next_data = data, output
            return

        with torch.cuda.stream(self.stream):
            if self.mask is None:
                gpu_data = self.to_gpu(data)
            else:
                gpu_data = self.to_gpu_with_mask(data)

            self.data_event = self.stream.record_event()
            if self.post_processing is None:
                self.next_data = gpu_data
            else:
                output = self.post_processing(gpu_data)
                self.processing_event = self.stream.record_event()
                self.add_to_gpu_tensors(output)
                self.next_data = gpu_data, output

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        self.loader = iter(self.dataloader)
        self.data_event = None
        self.processing_event = None
        self.cur_processing_event = None
        self.preloaded = False
        self.preload()
        return self

    def __next__(self):
        if not self.preloaded:
            self.preload()

        if self.data_event:
            torch.cuda.current_stream().wait_event(self.data_event)
        data = self.next_data
        if data is None:
            raise StopIteration

        if self.stream:
            for t in self.gpu_tensors:
                t.record_stream(torch.cuda.current_stream())
            self.gpu_tensors = []
        self.cur_processing_event = self.processing_event

        self.preloaded = False

        if not self.manual_preload:
            self.preload()
        return data

    def wait_post_processing(self):
        if self.cur_processing_event:
            torch.cuda.current_stream().wait_event(self.cur_processing_event)

    @property
    def sampler(self):
        return self.dataloader.sampler

    @property
    def batch_size(self):
        return self.dataloader.batch_size

    @property
    def drop_last(self):
        return self.dataloader.drop_last

    @property
    def batch_sampler(self):
        return self.dataloader.batch_sampler

    @property
    def generator(self):
        return self.dataloader.generator

    @property
    def collate_fn(self):
        return self.dataloader.collate_fn

    @property
    def persistent_workers(self):
        return self.dataloader.persistent_workers

    @property
    def multiprocessing_context(self):
        return self.dataloader.multiprocessing_context
