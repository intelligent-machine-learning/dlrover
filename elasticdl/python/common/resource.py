class NodeResource(object):
    def __init__(self, cpu, memory, gpu=None):
        self.cpu = cpu
        self.memory = memory
        self.gpu = gpu
