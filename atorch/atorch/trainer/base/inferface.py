from pathlib import Path

import torch
from accelerate.utils import get_pretty_name, save

from atorch.common.log_utils import default_logger as logger


# TODO: Use torch 2.4's torch.distribute.State to replace the following class
class Stateful:
    """
    following the way of accelerator saving and loading training state while saving checkpoint,

    """

    def state_dict(self):
        raise NotImplementedError

    def load_state_dict(self, state_dict: dict):
        raise NotImplementedError

    def save_state(self, path, cpt_name_index: str = "default", save_on_each_node: bool = False):
        """
        Saves the state of `obj` to `{path}/custom_checkpoint_{index}.pkl`
        """
        save_location = Path(path) / f"custom_checkpoint_{cpt_name_index}.pkl"
        logger.info(f"Saving the state of {get_pretty_name(self)} to {save_location}")
        save(self.state_dict(), save_location, save_on_each_node=save_on_each_node)

    def load_state(self, path, cpt_name_index: str = "default"):
        """
        Loads the state of `obj` at `{path}/custom_checkpoint_{index}.pkl`
        """
        load_location = f"{path}/custom_checkpoint_{cpt_name_index}.pkl"
        logger.info(f"Loading the state of {get_pretty_name(self)} from {load_location}")
        self.load_state_dict(torch.load(load_location, map_location="cpu"))
