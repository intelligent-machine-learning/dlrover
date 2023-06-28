import time

import torch

from atorch.common.log_utils import default_logger as logger


class ThroughputTimer:
    def __init__(
        self,
        batch_size,
        dp_world_size=None,
        start_step=2,
        steps_per_output=50,
    ):
        self.start_time = 0
        self.end_time = 0
        self.started = False
        self.batch_size = batch_size
        self.dp_world_size = dp_world_size
        if torch.distributed.is_initialized() and not dp_world_size:
            self.dp_world_size = torch.distributed.get_world_size()
        self.epoch_count = 0
        self.local_step_count = 0
        self.total_step_count = 0
        self.total_elapsed_time = 0
        self.start_step = start_step
        self.steps_per_output = steps_per_output

    def update_epoch(self):
        self.epoch_count += 1
        self.local_step_count = 0

    def start(self):
        self.started = True
        if self.total_step_count >= self.start_step:
            torch.cuda.synchronize()
            self.start_time = time.time()

    def stop(self, end_train=True, report_speed=True):
        if not self.started:
            return
        self.started = False
        self.total_step_count += 1
        self.local_step_count += 1
        if self.total_step_count >= self.start_step:
            torch.cuda.synchronize()
            self.end_time = time.time()
            duration = self.end_time - self.start_time
            self.total_elapsed_time += duration
            if self.local_step_count % self.steps_per_output == 0 and report_speed:
                logger.info(
                    "{}/{}, SamplesPerSec={}, MemAllocated={}GB, MaxMemAllocated={}GB".format(
                        self.epoch_count,
                        self.local_step_count,
                        self._avg_samples_per_sec(),
                        round(torch.cuda.memory_allocated() / 1024**3, 2),
                        round(torch.cuda.max_memory_allocated() / 1024**3, 2),
                    )
                )
            if end_train and report_speed:
                logger.info(
                    "{}/{}, SamplesPerSec={}, SecondsPerEpoch={}".format(
                        self.epoch_count,
                        self.local_step_count,
                        self._avg_samples_per_sec(),
                        self.total_elapsed_time / self.epoch_count,
                    )
                )

    def _avg_samples_per_sec(self):
        if self.total_step_count > 0:
            samples_per_step = self.batch_size * self.dp_world_size
            total_step_offset = self.total_step_count - self.start_step
            avg_time_per_step = self.total_elapsed_time / total_step_offset
            return samples_per_step / avg_time_per_step
        return float("-inf")
