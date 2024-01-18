import logging
import os
from importlib.metadata import version

from .distributed.distributed import coworker_size, init_distributed, local_rank, rank, reset_distributed, world_size

try:
    __version__ = version("atorch")
except ImportError:
    __version__ = "0.0.1dev"

os.environ["PIPPY_PIN_DEVICE"] = "0"

# patch with atorch addon if exists and not disabled by ATORCH_DISABLE_ADDON env.
disable_addon = False
disable_addon_env = os.getenv("ATORCH_DISABLE_ADDON")
if disable_addon_env is not None and disable_addon_env.lower() in ["true", "t", "1", "y", "yes"]:
    disable_addon = True

if disable_addon:
    logging.warning("atorch_addon disabled by env ATORCH_DISABLE_ADDON.")

addon_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "atorch_addon.py")

if not disable_addon and os.path.exists(addon_file):
    try:
        import atorch.atorch_addon
    except ImportError:
        logging.warning("Failed to import atorch_addon!")
