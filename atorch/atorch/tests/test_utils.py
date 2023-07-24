import os
import tempfile

import torch

from atorch.common.util_func import find_free_port
from atorch.distributed.launch import main, parse_args
from atorch.distributed.run import elastic_run, parse_fault_tolerant_or_elastic_args


def run_multi_process_init_distributed(codes=None, nproc=2, training_script=None, training_script_args=""):
    if codes is not None:
        fd, training_script = tempfile.mkstemp(suffix="py")
        with open(fd, "w") as f:
            f.write(codes)

    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_PORT"] = str(find_free_port())
    args = parse_args()
    args.training_script = training_script
    args.training_script_args = training_script_args
    args.nproc_per_node = nproc
    main(args)


def elastic_run_multi_process(codes=None, nproc=2, training_script=None, training_script_args=""):
    if codes is not None:
        fd, training_script = tempfile.mkstemp(suffix="py")
        with open(fd, "w") as f:
            f.write(codes)

    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_PORT"] = str(find_free_port())
    args = parse_fault_tolerant_or_elastic_args(mode="fault_tolerant")
    args.training_script = training_script
    args.training_script_args = training_script_args
    args.nproc_per_node = nproc
    elastic_run(args)


def create_sample_batch(value=1, start_v=0, y_dtype=torch.int64):
    x = torch.ones([12], dtype=torch.float32).reshape(3, 4)
    y = torch.arange(start_v, start_v + 8, dtype=y_dtype).reshape(2, 4)
    z = torch.zeros([16])
    z[:] = value
    return x, {"y": y, "z": z}


def start_coverage():
    try:
        import coverage

        global ut_cov

        ut_cov = coverage.Coverage()
        ut_cov.start()
        return True
    except ImportError:
        return False


def stop_coverage():
    global ut_cov
    ut_cov.stop()
    ut_cov.save()
