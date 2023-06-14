import time
import traceback

import torch
import torch.distributed.rpc as torch_rpc

from atorch.auto.model_context import ModelContext, get_data_partition_rank_and_size
from atorch.common.log_utils import default_logger as logger
from atorch.distributed.distributed import local_rank, parallel_group_and_ranks, rank


class DryRunner(object):
    def __init__(self):
        self.methods = {}
        # register dynamic tune methods
        self.methods["batchsize"] = tune_batchsize

    @staticmethod
    def profile(model_context, warmup_step_num=10, profile_step_num=15):
        """
        Profile the model training.
        Input:
          model_context: model context.
          warmup_step_num: the number of training steps before profiling
          profile_step_num: the number of training steps for profiling
        Output:
          status, result
          status indicating if the dryrun is successful.
          result is a dict of profiling results
          - "throughput": N-sample/s
          - "max_gpu_memory": max gpu memory used in training
          - "data_latency_percentage": the percentage of non-overlapping data io/preprocessing time in training
        """
        model, optim, dataloader, loss_func, prepare_input, model_input_format, lr_scheduler = (
            model_context.model,
            model_context.optim,
            model_context.dataloader,
            model_context.loss_func,
            model_context.prepare_input,
            model_context.model_input_format,
            model_context.lr_scheduler,
        )
        if dataloader is not None:
            batch_size = dataloader.batch_size
            if profile_step_num <= 0 or warmup_step_num < 0:
                logger.error("profile_step_num should be larger than 0 and warmup_step_num non-negative.")
                return False, None
            if len(dataloader) < warmup_step_num + profile_step_num:
                logger.error(
                    "len(dataloader)={} is smaller than warmup_step_num + profile_step_num = {}".format(
                        len(dataloader), warmup_step_num + profile_step_num
                    )
                )
                return False, None
            data_iter = iter(dataloader)
        else:
            if model_context.sample_batch is None:
                logger.error("Both dataloader and sample_batch are None. Skip profile model training.")
                return False, None
            batch_size = model_context.sample_batch_size
        _, ddp_size = get_data_partition_rank_and_size()
        pipe_group, pipe_ranks = parallel_group_and_ranks("pipe")
        status = True
        results = None
        device = "cuda:{}".format(local_rank()) if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        compute_total_time = 0
        if warmup_step_num == 0:
            start_time = time.time()
        try:
            if pipe_ranks is None or rank() == pipe_ranks[0]:
                idx = 0
                while True:
                    if dataloader is not None:
                        data = next(data_iter)
                    else:
                        data = model_context.sample_batch
                    if prepare_input is not None:
                        data = prepare_input(data, device)
                    if idx >= warmup_step_num:
                        compute_start_time = time.time()
                    optim.zero_grad()
                    if model_input_format == "unpack_dict":
                        output = model(**data)
                    elif model_input_format == "unpack_sequence":
                        output = model(*data)
                    else:
                        output = model(data)
                    loss = ModelContext.get_loss_from_loss_func_output(loss_func(data, output))
                    if pipe_ranks is None:
                        loss.backward()
                    optim.step()
                    if lr_scheduler is not None:
                        lr_scheduler.step()
                    if idx >= warmup_step_num:
                        compute_time = time.time() - compute_start_time
                        compute_total_time += compute_time
                    if idx + 1 == warmup_step_num:
                        start_time = time.time()
                    if idx + 1 == warmup_step_num + profile_step_num:
                        # finish profiling, get profiling results
                        total_time = time.time() - start_time
                        results = {}
                        results["throughput"] = profile_step_num / total_time * batch_size * ddp_size
                        results["data_latency_percentage"] = (total_time - compute_total_time) / total_time
                        break
                    idx += 1
            else:
                results = {}
                results["throughput"] = None
                results["data_latency_percentage"] = None

            if pipe_group is not None:
                logger.info(f"Pipeline training, rank {rank()} wait until all ranks finish")
                torch_rpc.api._wait_all_workers()
                logger.info(f"all ranks finishes, {rank()} goes through")

            if torch.cuda.is_available():
                max_gpu_mem = torch.cuda.max_memory_allocated(device)
                # TODO: get max of max_gpu_mem from all processes
            else:
                max_gpu_mem = 0
            results["max_gpu_memory"] = max_gpu_mem
            results["max_gpu_memory_in_gigabyte"] = round(max_gpu_mem / 1e9, 3)

        except Exception as e:
            traceback.print_exc()
            logger.error(f"dryrun profiling failed: {e}")
            status = False
        return status, results

    def dynamic_tune(self, method_name, model_context, profiling=True, **kargs):
        return self.methods[method_name](model_context, profiling, **kargs)


def tune_batchsize(model_context, profiling=False, **kargs):
    pass


_DRY_RUNNER = None


def get_dryrunner():
    global _DRY_RUNNER
    if _DRY_RUNNER is None:
        _DRY_RUNNER = DryRunner()
    return _DRY_RUNNER
