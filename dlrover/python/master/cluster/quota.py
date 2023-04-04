import sys
from abc import ABCMeta, abstractmethod


class QuotaChecker(meta=ABCMeta):

    @abstractmethod
    def get_avaliable_worker_num(self):
        pass


class UnlimitedQuotaChecker(QuotaChecker):
    """No resource limits."""
    def get_avaliable_worker_num(self):
        """Assume there is always enough resource."""
        return sys.maxsize
