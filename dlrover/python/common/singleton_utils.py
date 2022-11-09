import threading


def singleton(cls):
    _instance = {}
    _instace_lock = threading.Lock()

    def _singleton(*args, **kwargs):
        with _instace_lock:
            if cls not in _instance:
                _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]

    return _singleton
