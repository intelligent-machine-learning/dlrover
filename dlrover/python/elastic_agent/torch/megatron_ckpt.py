import os
import torch
import torch.distributed as dist
from datetime import timedelta
from typing import List
from dlrover.python.elastic_agent.torch.ckpt_saver import (
    CheckpointSaver,
    CheckpointEngine,
    SharedMemoryHandler,
    _init_dir,
    _SAVE_STEP_QNAME_PREFIX,
    _SHM_LOCK_NAME_PREFIX,
    _WIRTING_SHM,
)

from dlrover.python.common.multi_process import (
    SharedQueue,
    SharedLock,
)
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common import env_utils


class MegatronShardingSaver(CheckpointSaver):
    """
    The saver only saves the state dict without sharding
    from the shared memory created by local rank 0 to the storage.
    """

    def __init__(self, checkpoint_dir: str, num_shard: int) -> None:
        super().__init__(checkpoint_dir)
        self._num_shard = num_shard
        self._shm_handlers: List[SharedMemoryHandler] = []
        self._to_save_queues: List[SharedQueue] = []
        self._shm_locks: List[SharedLock] = []
        for i in range(num_shard):
            self._shm_handlers.append(SharedMemoryHandler(i))
            qname = _SAVE_STEP_QNAME_PREFIX + str(i)
            self._to_save_queues.append(SharedQueue(name=qname, create=True))
            lock_name = _SHM_LOCK_NAME_PREFIX + str(i)
            self._shm_locks.append(SharedLock(name=lock_name, create=True))

    def __del__(self):
        self.close()

    def close(self):
        for i in range(self._num_shard):
            if self._shm_handlers[i]:
                self._shm_handlers[i].close()
                self._shm_handlers[i].unlink()
            self._to_save_queues[i].unlink()
            self._shm_locks[i].unlink()

    def _sync_shm_to_storage(self):
        """
        The loop to persist the state dict from the memory
        buffer into the storage.
        """
        logger.info("Async checkpoint saver starts!")
        while True:
            paths = []
            for q in self._to_save_queues:
                paths.append(q.get())

    def _save_shm_to_storage(self, step, ckpt_path):
        """
        Save all the local state dict in the shared memory into the storage
        for step.
        """
        logger.info(
            f"Rank {self._node_rank} start save checkpoint to storage, "
            f"step: {step}"
        )
        self._writing_storage = True

        if os.path.exists(ckpt_path):
            logger.info(f"Checkpoint for step {step} already exists, skip")
            self._writing_storage = False
            return

        def _save_stage(local_rank: int, write_path: str):
            try:
                shm_handler = self._shm_handlers[local_rank]
                shm_lock = self._shm_locks[local_rank]
                if shm_handler.empty():
                    shm_handler.init_tensor_shm(create=False)

                shm_lock.acquire()
                logger.info(
                    f"Local rank {local_rank} Save checkpoint from the shared "
                    f"memory into the storage {write_path}."
                )
                _, state_dict = shm_handler.load_state_dict()
                self._persist_to_storage(state_dict, write_path, step)
                shm_lock.release()
                return True

            except Exception as e:
                logger.error(
                    f"Rank {local_rank} save checkpoint failed, error: {e}",
                    exc_info=True,
                )
                shm_lock.release()
                return False

        stage_path = os.path.join(self._get_stage_path(), str(step))
        os.makedirs(stage_path, exist_ok=True)
        step_done_path = os.path.join(
            self._get_stage_path(), str(step) + ".done"
        )
        os.makedirs(step_done_path, exist_ok=True)

        step_done_file = os.path.join(step_done_path, str(self._node_rank))

        write_success = False
        if os.path.exists(step_done_file):
            logger.info(f"Rank {self._node_rank} already done for step {step}")
            write_success = True
        else:
            # save to stage path for each local rank
            futures = []
            for i in range(self.num_shard):
                future = self._executor.submit(_save_stage, i, stage_path)
                futures.append(future)

            success_count = 0
            for (i, future) in enumerate(futures):
                if future.result():
                    success_count += 1
                else:
                    logger.error(
                        f"Rank {i} save checkpoint failed for step {step}"
                    )

            if success_count == self.num_shard:
                # all local rank done success
                with open(step_done_file, "w") as f:
                    f.write("done")
                write_success = True

        if not write_success:
            logger.error(
                f"Rank {self._node_rank} save checkpoint failed for "
                f"step {step}"
            )
            return

        # commit checkpoint
        if self._is_agent_rank_0:
            self._commit_checkpoint(
                step,
                step_done_dir=step_done_path,
                tmp_path=stage_path,
                target_path=ckpt_path,
            )

        self._writing_storage = False

    def _persist_to_storage(self, state_dict, path):
        """Persist the checkpoint from CPU memory buffer into the storage."""
        checkpoint_dir = os.path.dirname(path)
        state_dict.pop(_WIRTING_SHM, None)
        _init_dir(checkpoint_dir)
        torch.save(state_dict, path)

    def save_shm_to_storage(self):
        """
        Save the state dict in the shared memory into the storage. The agent
        can call the method to save the state dict into the storage if the
        training process fails or the agent wants to restart training
        processes.
        """
        if self._shm_handler.empty():
            return
        acquired = self._shm_lock.acquire()
        if not acquired:
            # The training process does not release the lock because it fails
            # when writing the state dict into the shared memory. The shared
            # memory may be dirty and the saver cannot save it to the storage.
            return
        step, state_dict = self._shm_handler.load_state_dict()
        if state_dict:
            path = os.path.join(
                self.checkpoint_dir, f"checkpoint-{step}/checkpoint.pt"
            )
            self._persist_to_storage(state_dict, path)
            logger.info(
                "Save the checkpointing state dict from the shared "
                f"memory to {path}."
            )
        self._shm_lock.release()


class NoShardingCheckpointEngine(CheckpointEngine):
    """
    The checkpoint engine synchronously writes the state dict into
    the shared memory and notify the agent in main process to
    asynchronously save the state dict from the shared memory into
    the storage. Writing to memory is significantly quicker
    than writing to storage. The engine only blocks the training
    with a little time. Users can frequently call `save_to_memory` in
    the training loop and call `save_to_storage`.

    If the training process fail, the agent in main process can continuely
    saves the the state dict from the shared memory into the storage.
    The engine saves the model and optimizer state dict without sharding
    in a local or DDP job.

    Attributes:
        checkpoint_dir (str):  the directory to save the temp checkpoint
            if the training process fails.

    Examples::
        >>> engine = NoShardingCheckpointEngine(
        >>>     checkpoint_dir="/tmp/checkpoint/"
        >>> )
        >>> for step, data in enumerate(dataloader):
        >>>     ...
        >>>     state_dict = model.state_dict()
        >>>     if step % 5 == 0:
        >>>         engine.save_to_memory(state_dict, step)
        >>>     elif step % 100 == 0:
        >>>         path = f"/tmp/checkpoint/ckpt-{step}.pt"
        >>>         engine.save_to_storage(state_dict, path, step)
        >>> sate_dict = engine.load()
    """

    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        if dist.is_initialized():
            self._rank = dist.get_rank()
            self._local_rank = int(os.environ["LOCAL_RANK"])
            saver_ranks = self._get_saver_ranks()
            self._saver_group = dist.new_group(
                ranks=saver_ranks,
                backend="gloo",
                timeout=timedelta(seconds=30),
            )
        else:
            self._rank = 0
            self._local_rank = int(os.getenv("LOCAL_RANK", 0))
            self._saver_group = None

        self._buffer_size = 0
        self._cached_step = 0
        self._restart_count = env_utils.get_torch_restart_count()

        lock_name = _SHM_LOCK_NAME_PREFIX + str(0)
        self._shm_lock = SharedLock(name=lock_name, create=False)
        qname = _SAVE_STEP_QNAME_PREFIX + str(0)
        self._to_save_queue = SharedQueue(name=qname, create=False)
        self._shm_handler = SharedMemoryHandler(0, host=False)
        self._notify_agent_to_create_saver()

    def _get_saver_ranks(self):
        """
        Only the local rank 0 in each node saves the state dict into the
        memory. They need to synchronize the saving status.
        """
        group_size = env_utils.get_group_world_size()
        local_world_size = env_utils.get_local_world_size()
        save_ranks = []
        for i in range(group_size):
            saver_rank = i * local_world_size
            save_ranks.append(saver_rank)
        return save_ranks

    def _notify_agent_to_create_saver(self):
        if self._local_rank != 0:
            return
        if self._restart_count > 0:
            # Only local rank 0 notify to initialize the saver in
            # the main process at the first start.
            # Avoid the lock is locked by a failed process.
            self._shm_lock.release()
            return
        queue = SharedQueue(name="factory")
        class_meta = SaverClassMeta(
            module_path="dlrover.python.elastic_agent.torch.ckpt_saver",
            class_name="NoShardingSaver",
            init_args={"checkpoint_dir": self.checkpoint_dir},
        )
        queue.put(class_meta)
        queue.unlink()

    def __del__(self):
        self.close()

    def close(self):
        self._shm_handler.close()

    @timer
    def save_to_memory(self, state_dict, step):
        """
        Synchonously Saves the state dict into the shared memory with the main
        process. If the agent in the main process is saving the shared memory
        into the storage, the method will skip to write the shared memory.
        Only local rank 0 save the state dict into the memory because the
        state dict is replicated across all ranks.

        Args:
            state_dict (dict): the state dict of model and optimizer to save.
            step (int): the iteration step.
        """
        if self._local_rank != 0:
            return
        if "step" not in state_dict:
            state_dict["step"] = step
        if _WIRTING_SHM in state_dict:
            raise ValueError(f"state_dict cannot have the key {_WIRTING_SHM}.")

        self._shm_handler.make_state_dict_buffer(state_dict)
        acquired = self._shm_lock.acquire(blocking=False)
        all_rank_ready = _check_all_rank_ready(self._saver_group, acquired)
        if not all_rank_ready:
            logger.info(
                f"Rank {self._rank} skips the save the checkpoint "
                f"in CPU memory since it is saving the latest "
                "checkpoint from the CPU memory into the storage."
            )
            if acquired:
                self._shm_lock.release()
            return
        self._shm_handler.save_state_dict(state_dict)

        if acquired:
            self._shm_lock.release()
        self._cached_step = step

    @timer
    def save_to_storage(self, state_dict, path, step):
        """
        Asynchonously saves the state dict into the storage. It synchonously
        saves the state dict into the shared memory and put the path
        into a shared queue. The agent in the main process waits for the queue
        for save the state dict in the shared memory into the storage.
        Only rank 0 saves the state dict into the storage.

        Args:
            state_dict (dict): the state dict of model and optimizer to save.
            path (str): optional, the file path to save the checkpoint. If the
                path is not defined, the engine will save the state dict into
                the shared memory not the storage.
            step (int): the iteration step.
        """
        if self._rank != 0:
            return
        if step > self._cached_step:
            self.save_to_memory(state_dict, step)
        if path:
            self._to_save_queue.put(path)

    def load(self, resume_path=""):
        """
        The method firstly try to load the state dict from the shared memory.
        If there is no state dict in the shared memory, the method will
        load the state dict from the storage.

        Returns:
            A dict.
        """
        step, state_dict = self._shm_handler.load_state_dict()
        if state_dict:
            return step, state_dict
        state_dict = self._load_from_storage(resume_path)
        step = state_dict.get("step", 0)
        return step, state_dict

    def _load_from_storage(self, resume_path=""):
        """
        Load the state dict from the CPU memory if the state dict is complete
        in CPU memory. Otherwise, the function will load the state dict from
        the storage.

        Args:
            resume_path (str, optional): , If the resume_path is an empty
                string, the function will load the latest checkpoint file in
                the checkpoint directory.

        Returns:
            A dict:
                a dictionary containing a whole state of the modules in the
                checkpointing file.
        """
        if resume_path:
            state_dict = torch.load(resume_path)
        else:
            state_dict = _load_from_historic_checkpoint(self.checkpoint_dir)
        return state_dict
