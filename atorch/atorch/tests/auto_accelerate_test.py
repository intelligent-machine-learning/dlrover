import os
import pickle
import sys
import tempfile
import unittest

import torch

import atorch
from atorch.auto.accelerate import adjust_strategy, auto_accelerate, get_strategy, run_task, save_strategy
from atorch.auto.analyser.analyser import Analyser
from atorch.auto.device_context import get_device_context
from atorch.auto.dry_runner.dry_runner import DryRunner
from atorch.auto.opt_lib.optimization_library import OptimizationLibrary
from atorch.auto.strategy import Strategy
from atorch.auto.task import Task
from atorch.data import data_to_device
from atorch.tests.test_utils import run_multi_process_init_distributed, start_coverage, stop_coverage
from atorch.tests.toy_module import ToyModel, create_model_context, loss_func, optim_func, prepare_input, run_train


def task_run_code(data_size, batch_size):
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    res = atorch.init_distributed(backend, set_cuda_device_using_local_rank=True)
    if not res:
        raise Exception("init failed")
    model_context = create_model_context(data_size=data_size, batch_size=batch_size)
    opt_lib = OptimizationLibrary()
    dry_runner = DryRunner()
    analyser = Analyser()
    tasks = []
    task = Task(id=0, task_type="ANALYSE", process_mode="ONE_PROCESS", task_info=["analyse_basic"])
    tasks.append(task)
    pg_info = ([("data", 2)], None)
    task = Task(id=1, task_type="SETUP_PARALLEL_GROUP", process_mode="ALL_PROCESS", task_info=pg_info)
    tasks.append(task)
    task = Task(id=2, task_type="WAIT")
    tasks.append(task)
    strategy = Strategy([("parallel_mode", pg_info, False)])
    task = Task(id=3, task_type="DRYRUN", process_mode="ALL_PROCESS", task_info=strategy)
    tasks.append(task)
    task = Task(id=4, task_type="FINISH", process_mode="ALL_PROCESS", task_info=strategy)
    tasks.append(task)
    for task in tasks:
        status, result = run_task(model_context, task, opt_lib=opt_lib, dry_runner=dry_runner, analyser=analyser)
        assert status is True
        if task.type == "FINISH":
            m_model, m_optim, m_dataloader, m_loss_func, m_prepare_input = (
                result.model,
                result.optim,
                result.dataloader,
                result.loss_func,
                result.prepare_input,
            )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num = run_train(m_model, m_dataloader, m_optim, m_prepare_input, m_loss_func, device)
    assert num == data_size // batch_size, f"num={num}"

    atorch.reset_distributed()


class FakeEasyDLClient(object):
    def __init__(self, addr, port):
        tasks = []
        task = Task(id=0, task_type="ANALYSE", process_mode="ONE_PROCESS", task_info=["analyse_basic"])
        tasks.append(task)
        pg_info = ([("data", 2)], None)
        task = Task(id=1, task_type="SETUP_PARALLEL_GROUP", process_mode="ALL_PROCESS", task_info=pg_info)
        tasks.append(task)
        task = Task(id=2, task_type="WAIT")
        tasks.append(task)
        strategy = Strategy([("parallel_mode", pg_info, False)])
        task = Task(id=3, task_type="DRYRUN", process_mode="ALL_PROCESS", task_info=strategy)
        tasks.append(task)
        task = Task(id=4, task_type="FINISH", process_mode="ALL_PROCESS", task_info=strategy)
        tasks.append(task)
        self.tasks = tasks
        self.index = 0

    def get_task(self):
        if self.index < len(self.tasks):
            self.index += 1
            return self.tasks[self.index - 1]
        return None

    def report_task_result(self, task, status, result):
        pass


class FakeEasyDLClientWithFailTask(FakeEasyDLClient):
    def __init__(self, addr, port):
        tasks = []
        task = Task(id=0, task_type="ANALYSE", process_mode="ONE_PROCESS", task_info=["analyse_basic"])
        tasks.append(task)
        task = Task(id=4, task_type="FAIL", process_mode="ALL_PROCESS", task_info=None)
        tasks.append(task)
        self.tasks = tasks
        self.index = 0


class FakeEngine(object):
    def __init__(
        self,
        device_context,
        included_opts=None,
        excluded_opts=None,
        load_strategy=None,
        time_limit=None,
        verbose=False,
    ):
        pass

    def start_service(self, port):
        pass

    def tear_down(self):
        pass


def func_run_code(data_size, batch_size):
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    res = atorch.init_distributed(backend, set_cuda_device_using_local_rank=True)
    if not res:
        raise Exception("init failed")
    model_context = create_model_context(data_size=data_size, batch_size=batch_size)

    atorch.auto.accelerate.EngineClient = FakeEasyDLClient
    atorch.auto.accelerate.AccelerationEngine = FakeEngine

    status, res, best_strategy = auto_accelerate(
        model_context.model,
        model_context.optim_func,
        model_context.dataset,
        loss_func=model_context.loss_func,
        prepare_input=model_context.prepare_input,
        optim_args=model_context.optim_args,
        dataloader_args=model_context.dataloader_args,
    )
    assert len(best_strategy) == 1
    assert status
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num = run_train(res.model, res.dataloader, res.optim, res.prepare_input, res.loss_func, device)
    assert num == data_size // batch_size, f"num={num}"

    model_context = create_model_context(data_size=data_size, batch_size=batch_size)
    atorch.auto.accelerate.EngineClient = FakeEasyDLClientWithFailTask
    status, _, _ = auto_accelerate(
        model_context.model,
        model_context.optim_func,
        model_context.dataset,
        loss_func=model_context.loss_func,
        prepare_input=model_context.prepare_input,
        optim_args=model_context.optim_args,
        dataloader_args=model_context.dataloader_args,
    )
    assert status is False
    atorch.reset_distributed()


def save_load_code(data_size, batch_size):
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    res = atorch.init_distributed(backend, set_cuda_device_using_local_rank=True)
    if not res:
        raise Exception("init failed")
    model_context = create_model_context(data_size=data_size, batch_size=batch_size)

    b_data = [None, None]
    if atorch.rank() == 0:
        pg_info = ([("data", 1)], None)
        strategy = Strategy([("parallel_mode", pg_info, False)])
        _, filename = tempfile.mkstemp(suffix="st")
        save_strategy(strategy, filename)
        _, save_filename = tempfile.mkstemp(suffix="st")
        b_data = [filename, save_filename]

    torch.distributed.broadcast_object_list(b_data, src=0)

    status, res, best_strategy = auto_accelerate(
        model_context.model,
        model_context.optim_func,
        model_context.dataset,
        loss_func=model_context.loss_func,
        prepare_input=model_context.prepare_input,
        optim_args=model_context.optim_args,
        dataloader_args=model_context.dataloader_args,
        load_strategy=b_data[0],
        save_strategy_to_file=b_data[1],
    )
    assert status
    assert len(best_strategy) == 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num = run_train(res.model, res.dataloader, res.optim, res.prepare_input, res.loss_func, device)
    assert num == data_size // batch_size, f"num={num}"

    status, strategy = get_strategy(b_data[1])
    assert status
    assert strategy == best_strategy

    pg_info = ([("data", 8)], None)
    strategy = Strategy([("parallel_mode", pg_info, False)])
    s_data = pickle.dumps(strategy)
    status, res, strategy = auto_accelerate(
        model_context.model,
        model_context.optim_func,
        model_context.dataset,
        loss_func=model_context.loss_func,
        prepare_input=model_context.prepare_input,
        optim_args=model_context.optim_args,
        dataloader_args=model_context.dataloader_args,
        load_strategy=s_data,
    )
    assert status
    assert len(strategy) == 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num = run_train(res.model, res.dataloader, res.optim, res.prepare_input, res.loss_func, device)
    assert num == data_size // batch_size, f"num={num}"
    assert strategy == best_strategy

    atorch.reset_distributed()


class FakeEasyDLClientOnlyPass(FakeEasyDLClient):
    def __init__(self, addr, port):
        tasks = []
        task = Task(id=0, task_type="ANALYSE", process_mode="ONE_PROCESS", task_info=["analyse_basic"])
        tasks.append(task)
        pg_info = ([("data", 2)], None)
        task = Task(id=1, task_type="SETUP_PARALLEL_GROUP", process_mode="ALL_PROCESS", task_info=pg_info)
        tasks.append(task)
        task = Task(id=2, task_type="WAIT")
        tasks.append(task)
        strategy = Strategy([("parallel_mode", pg_info, False)])
        task = Task(id=3, task_type="DRYRUN", process_mode="ALL_PROCESS", task_info=strategy)
        tasks.append(task)
        task = Task(id=4, task_type="FINISH", process_mode="ALL_PROCESS", task_info=strategy)
        tasks.append(task)
        self.tasks = tasks
        self.index = 0


def only_pass_model_code(dp_size):
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    res = atorch.init_distributed(backend, set_cuda_device_using_local_rank=True)
    if not res:
        raise Exception("init_distributed failed")
    model = ToyModel()
    atorch.auto.accelerate.EngineClient = FakeEasyDLClientOnlyPass
    atorch.auto.accelerate.AccelerationEngine = FakeEngine

    pg_info = ([("data", dp_size)], None)
    strategy = Strategy([("parallel_mode", pg_info, False)])

    status, res, best_strategy = auto_accelerate(model, load_strategy=strategy)
    assert len(best_strategy) == 1
    assert status
    assert res.model is not None
    assert res.dataloader is None
    assert res.optim is None
    assert res.prepare_input is data_to_device
    assert res.loss_func is None
    assert best_strategy == strategy


class FakeEasyDLClientPassSampleBatch(FakeEasyDLClient):
    def __init__(self, addr, port):
        tasks = []
        task = Task(id=0, task_type="ANALYSE", process_mode="ONE_PROCESS", task_info=["analyse_basic"])
        tasks.append(task)
        pg_info = ([("data", 2)], None)
        task = Task(id=1, task_type="SETUP_PARALLEL_GROUP", process_mode="ALL_PROCESS", task_info=pg_info)
        tasks.append(task)
        task = Task(id=2, task_type="WAIT")
        tasks.append(task)
        strategy = Strategy([("parallel_mode", pg_info, False)])
        task = Task(id=3, task_type="DRYRUN", process_mode="ALL_PROCESS", task_info=strategy)
        tasks.append(task)
        task = Task(id=4, task_type="FINISH", process_mode="ALL_PROCESS", task_info=strategy)
        tasks.append(task)
        self.tasks = tasks
        self.index = 0


def pass_sample_batch_code(dp_size):
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    res = atorch.init_distributed(backend, set_cuda_device_using_local_rank=True)
    if not res:
        raise Exception("init_distributed failed")
    model = ToyModel()
    batch_size = 2
    sample_batch = [torch.ones([batch_size, 16], dtype=torch.float32), torch.ones([batch_size, 4], dtype=torch.float32)]
    atorch.auto.accelerate.EngineClient = FakeEasyDLClientPassSampleBatch
    atorch.auto.accelerate.AccelerationEngine = FakeEngine

    pg_info = ([("data", dp_size)], None)
    strategy = Strategy([("parallel_mode", pg_info, False)])

    status, res, best_strategy = auto_accelerate(
        model,
        prepare_input=prepare_input,
        optim_func=optim_func,
        load_strategy=strategy,
        loss_func=loss_func,
        sample_batch=sample_batch,
        batch_size=batch_size,
    )
    assert len(best_strategy) == 1
    assert status
    assert res.model is not None
    assert res.dataloader is None
    assert res.optim is not None
    assert res.prepare_input is not None
    assert res.loss_func is not None
    assert best_strategy == strategy


class AutoAccelerateTest(unittest.TestCase):
    def test_tasks(self):
        run_dist_code("test_tasks")

    def test_func(self):
        run_dist_code("test_func")

    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2, "Skip when gpu not exists or only 1 gpu"
    )
    def test_only_pass_model_to_auto_accelerate(self):
        run_dist_code("only_pass_model")

    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2, "Skip when gpu not exists or only 1 gpu"
    )
    def test_pass_sample_batch(self):
        run_dist_code("pass_sample_batch")


class LoadSaveStrategyTest(unittest.TestCase):
    def test_load_save_strategy(self):
        pg_info = ([("data", 2)], None)
        strategy = Strategy([("parallel_mode", pg_info, False), ("amp_native", None, False)])
        _, filename = tempfile.mkstemp(suffix="st")
        save_strategy(strategy, filename)
        status, loaded_strategy = get_strategy(filename)
        self.assertTrue(status)
        self.assertEqual(loaded_strategy, strategy)
        data = pickle.dumps(strategy)
        status, loaded_strategy = get_strategy(data)
        self.assertTrue(status)
        self.assertEqual(loaded_strategy, strategy)
        status, _ = get_strategy("_bad_filename.st")
        self.assertFalse(status)
        easydl_strategy = strategy.convert_strategy_to_easydl_format()
        self.assertEqual(len(easydl_strategy), 2)

    def test_adjust_strategy(self):
        pg_info = ([("model", 2), ("data", 2)], None)
        strategy = Strategy([("parallel_mode", pg_info, False), ("amp_native", None, False)])
        device_context = get_device_context()
        device_context._node_num = 2
        device_context._nproc_per_node = 8
        opt_lib = OptimizationLibrary()
        finetune_strategy = False
        status, strategy = adjust_strategy(strategy, device_context, finetune_strategy, opt_lib)
        self.assertTrue(status)
        data_parallel_size = 1
        found, p_mode = strategy.get_parallel_mode()
        self.assertTrue(found)
        for name, size in p_mode[0]:
            if name == "data":
                data_parallel_size = size
        self.assertEqual(data_parallel_size, 8)
        device_context._node_num = 1
        device_context._nproc_per_node = 1
        status, _ = adjust_strategy(strategy, device_context, finetune_strategy, opt_lib)
        self.assertFalse(status)

        device_context._node_num = 10
        device_context._nproc_per_node = 2
        finetune_strategy = True
        strategy = ["parallel_mode", "amp_native"]
        status, strategy = get_strategy(strategy)
        self.assertTrue(status)
        status, strategy = adjust_strategy(strategy, device_context, finetune_strategy, opt_lib)
        self.assertTrue(status)
        data_parallel_size = 1
        _, p_mode = strategy.get_parallel_mode()
        for name, size in p_mode[0]:
            if name == "data":
                data_parallel_size = size
        self.assertEqual(data_parallel_size, 20)

        strategy = Strategy(
            [
                ("amp_native", None, False),
                ("module_replace", None, False),
                ("fsdp", None, False),
            ]
        )
        removed_items = strategy.remove_distributed_method(opt_lib)
        self.assertEqual(len(strategy), 2)
        self.assertEqual(len(removed_items), 1)

    def test_load_save_with_api(self):
        run_dist_code("save_load")


def run_dist_code(name, nproc=2):
    code_path = os.path.abspath(__file__)
    run_multi_process_init_distributed(nproc=nproc, training_script=code_path, training_script_args=(name,))


if __name__ == "__main__":
    cov_status = start_coverage()
    if sys.argv[1] == "test_tasks":
        task_run_code(data_size=100, batch_size=2)
    elif sys.argv[1] == "test_func":
        func_run_code(data_size=100, batch_size=2)
    elif sys.argv[1] == "save_load":
        save_load_code(data_size=100, batch_size=2)
    elif sys.argv[1] == "only_pass_model":
        only_pass_model_code(dp_size=2)
    elif sys.argv[1] == "pass_sample_batch":
        pass_sample_batch_code(dp_size=2)
    if cov_status:
        stop_coverage()
