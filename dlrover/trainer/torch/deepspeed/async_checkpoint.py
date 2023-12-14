from deepspeed.runtime.checkpoint_engine.torch_checkpoint_engine import CheckpointEngine
from deepspeed.runtime.engine import DeepSpeedEngine
from dlrover.python.elastic_agent.torch.ckpt_saver import DeepSpeedCheckpointEngine


class AsyncSaveEngine(CheckpointEngine):

    def __init__(self):
        self.model_sd = None
        self.model_path = ""
        self.optimizer_sd = None
        self.optimizer_path = ""
    
    def create(self, tag):
        # create checkpoint on give tag for save/load.
        pass

    def save(self, state_dict, path: str):
        if "optimizer_state_dict" in state_dict:
            self.optimizer_sd = state_dict
            self.optimizer_path = path
        else:
            self.model_sd = state_dict
            self.model_path = path

    def load(self, path: str, map_location=None):
        pass

    def commit(self, tag):
        # to tell checkpoint services if all files are ready.
        pass


class DeepSpeedCheckpointManger(object):
    def __init__(self, engine: DeepSpeedEngine, checkpoint_dir):
        self.engine = engine
        self.engine.checkpoint_engine = AsyncSaveEngine()
        dp_size = self.engine.dp_world_size()
        self._async_save_engine = DeepSpeedCheckpointEngine(
            checkpoint_dir,
            dp_size=dp_size,
        )

    def save_checkpoint_to_memory(self, save_dir, tag=None, client_state={}, save_latest=True):
        self.engine.save_checkpoint(save_dir, tag, client_state, save_latest)
        state_dict = self._merge_model_and_optmizer_state_dict()
        self._async_save_engine.save_to_memory(
            tag,
            state_dict,
            model_path=self.engine.checkpoint_engine.model_path,
            optimizer_path=self.engine.checkpoint_engine.optimizer_path
        )

    def save_checkpoint_to_storage(self, save_dir, tag=None, client_state={}, save_latest=True):
        self.engine.save_checkpoint(save_dir, tag, client_state, save_latest)
        state_dict = self._merge_model_and_optmizer_state_dict()
        self._async_save_engine.save_to_storage(
            tag,
            state_dict,
            model_path=self.engine.checkpoint_engine.model_path,
            optimizer_path=self.engine.checkpoint_engine.optimizer_path
        )

    def _merge_model_and_optmizer_state_dict(self):
        merged_state_dict = {}
        if self.engine.checkpoint_engine.model_sd:
            merged_state_dict["deepspeed_model"] = self.engine.checkpoint_engine.model_sd
        if self.engine.checkpoint_engine.optimizer_sd:
            merged_state_dict["deepspeed_optimizer"] = self.engine.checkpoint_engine.optimizer_sd
        return merged_state_dict
