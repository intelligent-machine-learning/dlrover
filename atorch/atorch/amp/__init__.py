from atorch.common.log_utils import default_logger as logger

try:
    from .amp import initialize, load_state_dict, master_params, scale_loss, state_dict
    from .hook import sample_list_to_type
except ImportError:
    logger.info("Apex not available")
