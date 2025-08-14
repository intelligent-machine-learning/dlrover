import asyncio
from unittest.mock import Mock

from dlrover.python.unified.backend.common.base_worker import BaseWorker
from dlrover.python.unified.common.workload_base import ActorInfo, JobInfo
from dlrover.python.unified.common.workload_desc import SimpleWorkloadDesc


async def test_start_base():
    info = ActorInfo(
        name="worker1",
        role="worker",
        spec=SimpleWorkloadDesc(entry_point=f"{__name__}._entrypoint"),
    )
    worker = BaseWorker(Mock(JobInfo), info)

    global _entrypoint
    _entrypoint = Mock()

    assert worker.stage == "READY"
    worker.start()
    assert worker.stage == "RUNNING"
    while worker.stage != "FINISHED":
        await asyncio.sleep(0.1)
    assert _entrypoint.call_count == 1


async def test_start_class():
    info = ActorInfo(
        name="worker1",
        role="worker",
        spec=SimpleWorkloadDesc(entry_point=f"{__name__}._entrypoint"),
    )
    worker = BaseWorker(Mock(JobInfo), info)

    init_called = False

    class entrypoint_class:
        def __init__(self) -> None:
            nonlocal init_called
            init_called = True

        run = staticmethod(Mock())

    global _entrypoint
    _entrypoint = entrypoint_class

    assert worker.stage == "READY"
    worker.start()
    assert worker.stage == "RUNNING"
    assert init_called
    assert _entrypoint.run.call_count == 0
    while worker.stage != "FINISHED":
        await asyncio.sleep(0.1)
    assert _entrypoint.run.call_count == 1
