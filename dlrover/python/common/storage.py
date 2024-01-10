from abc import ABCMeta, abstractmethod


class CheckpointStorage(metaclass=ABCMeta):
    """
    We can implement interfaces of CheckpointStorage to
    write/read the data.
    """

    @abstractmethod
    def write(self, content, path):
        """
        Write the content into the path of storage.

        Args:
            content (str/bytes): the content to write.
            path (str): the path of storage.
        """
        pass

    @abstractmethod
    def write_state_dict(self, state_dict, path):
        """
        Write the state dict of PyTorch into the path of storage.

        Args:
            state_dict (dict): a state dict.
            path (str): the path of storage.
        """
        pass

    @abstractmethod
    def read(self, path):
        """
        Read string from  the path.

        Args:
            path(str): the file path of storage.
        """
        pass

    @abstractmethod
    def read_state_dict(self, path):
        """
        Read state dict from the path.

        Args:
            path(str): the file path of storage.
        """
        pass
    
    @abstractmethod
    def safe_rmtree(self, dir):
        pass

    @abstractmethod
    def safe_remove(self, path):
        pass

    @abstractmethod
    def record(self, step):
        """
        We can implement the method to record the checkpoint step.

        Args:
            step (int): the iteration step.
        """
        pass
