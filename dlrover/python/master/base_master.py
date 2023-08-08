
from abc import ABCMeta, abstractclassmethod


class JobMaster(metaclass=ABCMeta):
    @abstractclassmethod
    def prepare(self):
        pass

    @abstractclassmethod
    def run(self):
        pass

    @abstractclassmethod
    def stop(self):
        pass

    @abstractclassmethod
    def request_stop(self):
        pass
