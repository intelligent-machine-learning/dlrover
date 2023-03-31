from dlrover.trainer.tensorflow.util.common_util import singleton
from dlrover.trainer.util.log_util import default_logger as logger

@singleton
class ReaderRegistry(dict):
    """
    """
    def __init__(self):
        self.reader_registry = {}

    def register_reader(self, reader_name, reader_class):
        if reader_name in self.reader_registry.keys():
            logger.info("reader {} is registered in reader_registry before, it would be replaced by {}".format(reader_name, reader_class))
        self.register_reader[reader_name] = reader_class
        logger.info("reader {} is registered in reader_registry, reader class is {}".format(reader_name, reader_class))
        
    def get_reader(self, reader_name=None):
        if reader_name not in self.reader_registry.keys():
            raise Exception("reader is not registered in reader_registry before")
        reader_class = self.reader_registry[reader_name]
        logger.info("reader {} is registered in reader_registry, reader class is {}".format(reader_name, reader_class))
        return reader_class
 
