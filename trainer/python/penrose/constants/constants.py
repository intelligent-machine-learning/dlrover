class Constant(object):
    def __init__(self, name, default=None):
        self._name = name
        self._default = default

    def __call__(self):
        return self._default

    @property
    def default(self):
        return self._default

    @default.setter
    def default(self, val):
        self._default = val

    @property
    def name(self):
        return self._name


class Constants(object):
    Executor = Constant("executor")
    PsExecutor = Constant("ps_executor")