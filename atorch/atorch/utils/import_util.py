import importlib
import os


def import_module_from_py_file(file_path):
    model_module = None
    if os.path.exists(file_path):
        model_class_path = file_path.replace(".py", "").strip("./")
        model_class_path = model_class_path.replace("/", ".").strip(".")
        model_module = importlib.import_module(model_class_path)
    return model_module
