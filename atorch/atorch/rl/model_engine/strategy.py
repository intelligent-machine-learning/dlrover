# This file defines some commonly used optimization strategies.
from atorch.utils.import_util import import_module_from_py_file


# use constant to refer actor/critic/reward_model/ref_model
def get_strategy(file_path, model_type="actor"):
    """
    Actor/critic/ref_model/reward_models' strategy
    could be defined in the same or seperate file.
    If they are defined in the same file, each strategy
    should be explicitly announced. For example:
        actor_strategy = []
        critic_strategy = []
        ref_model_strategy = []
        reward_model_strategy = []
    """
    module = import_module_from_py_file(file_path)
    strategy = None
    role_strategy = "{}_strategy".format(model_type)
    if hasattr(module, role_strategy):
        strategy = getattr(module, role_strategy)
    elif hasattr(module, "strategy"):
        strategy = getattr(module, "strategy")
    else:
        # if user doesn't define the strategy, atorch would't modify anyting
        strategy = "torch_native"
    return strategy


def ddp_strategy():
    return ["parallel_mode"]


def zero1_strategy():
    return ["parallel_mode", "zero1"]
