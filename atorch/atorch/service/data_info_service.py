import multiprocessing as mp
from concurrent import futures

import grpc
from google.protobuf import empty_pb2  # type: ignore

from atorch.common.log_utils import default_logger as logger

try:
    from atorch.protos import coworker_pb2_grpc  # type: ignore
except ImportError:
    coworker_pb2_grpc = None


def create_data_info_service(port, pool_size=10):
    """
    Create and return a Data Info Service.

    Args:
        port: Data Info Service will listen at `port` when started.

    Returns:
        A grpc server representing Data Info Service. User should
        start it manually.
    """
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=pool_size),
    )
    data_info_servicer = DataInfoServicer()
    coworker_pb2_grpc.add_DataInfoServiceServicer_to_server(data_info_servicer, server)
    server.add_insecure_port("[::]:{}".format(port))
    logger.info("Data Info Service is created with port {}.".format(port))
    return server


class DataInfoServicer(coworker_pb2_grpc.DataInfoServiceServicer):
    """
    Data Info Service receive data info from Coworker1 ~ Coworkern-1 and put
    data info to data info queue. Workers on gpu pod will try to get data info
    from Data Info Service.

    On each gpu pod, Data Info Service is created on worker0.
    """

    def __init__(self):
        self._data_info_queue = mp.Queue()

    def report_data_info(self, info, context):
        """
        By calling `report_data_info` through grpc, gpu pod clients on
        Coworkers send data infos to Data Info Service. Data Info Service
        put data infos into data info queue.
        """
        self._data_info_queue.put(info)
        return empty_pb2.Empty()

    def get_data_info(self, request, context):
        """
        By calling `get_data_info` through grpc, data info rpc clients on
        gpu pods get data info from Data Info Service.
        """
        data_info = self._data_info_queue.get()
        return data_info
