import importlib
import os


def import_module_from_py_file(file_path):
    model_module = None
    if os.path.exists(file_path):
        model_class_path = file_path.replace(".py", "").strip("./")
        model_class_path = model_class_path.replace("/", ".").strip(".")
        model_module = importlib.import_module(model_class_path)
    return model_module


def import_module(module_name):
    func = module_name.split(".")[-1]
    module_path = module_name.replace("." + func, "")
    module = importlib.import_module(module_path)
    return getattr(module, func)
