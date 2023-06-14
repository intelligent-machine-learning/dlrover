import inspect


class AutoAccelerateContext:
    """
    AutoAccelerateContext is a global storage for auto_accelerate.
    Use AutoAccelerateContext.add_ac_attr to add/update an attribute with name and value.
    To access an added attribute with attr_name, use AutoAccelerateContext.attr_name.
    Use AutoAccelerateContext.reset to delete all attrs added by add_ac_attr.
    """

    # Number of times the function has been called
    counter = 0

    @classmethod
    def add_ac_attr(cls, name, value):
        if hasattr(cls, name):
            cls.name = value
        else:
            setattr(cls, name, value)

    @classmethod
    def reset(cls):
        reset_white_list = {"counter", "skip_dryrun"}
        method_list = inspect.getmembers(cls, predicate=inspect.ismethod)
        method_name_list = [method_tuple[0] for method_tuple in method_list]
        for attr in dir(cls):
            if attr in reset_white_list:
                continue
            if not (attr.startswith("__") and attr.endswith("__")) and attr not in method_name_list:
                delattr(cls, attr)
