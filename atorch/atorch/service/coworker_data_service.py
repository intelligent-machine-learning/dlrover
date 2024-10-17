import os
import pickle
from concurrent import futures

import grpc

from atorch.common.constants import GrpcEnv
from atorch.common.log_utils import default_logger as logger

try:
    from atorch.protos import coworker_pb2, coworker_pb2_grpc
except ImportError:
    coworker_pb2 = coworker_pb2_grpc = None


def create_coworker_rpc_service(port, data_queue):
    """
    Create and return a Coworker Rpc Service.

    Args:
        port: Coworker Rpc Service will listen at `port` when started.
        data_queue: A queue communicate between Coworker Rpc Service and
            Elastic Dataloader. Elastic Dataloader will put data into
            data_queue and coworker rpc client will get data from data_queue.

    Returns:
        A grpc server representing Coworker Rpc Service. User should
        start it manually.
    """
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=os.cpu_count()),
        options=[
            ("grpc.max_send_message_length", GrpcEnv.MAX_SEND_MESSAGE_LENGTH),
            (
                "grpc.max_receive_message_length",
                GrpcEnv.MAX_RECEIVE_MESSAGE_LENGTH,
            ),
        ],
    )
    data_info_servicer = CoworkerRpcServicer(data_queue)
    coworker_pb2_grpc.add_CoworkerRpcServiceServicer_to_server(data_info_servicer, server)
    server.add_insecure_port("[::]:{}".format(port))
    logger.info("Coworker Rpc Service has been created with port {}.".format(port))
    return server


class CoworkerRpcServicer(coworker_pb2_grpc.CoworkerRpcServiceServicer):
    """
    Coworker Rpc Service shares a batched data queue with Elastic Dataloader.
    Elastic Dataloader put preprocessed data into the batched data queue. And
    coworker rpc clients on gpu pods get data through grpc.
    """

    def __init__(self, batched_data_queue):
        self._batched_data_queue = batched_data_queue

    def get_batch_data(self, request, context):
        """
        By calling `get_batch_data` through grpc, coworker rpc clients
        get preprocessed data from Coworker Rpc Service's batched data
        queue.
        """
        data = self._batched_data_queue.get()
        bytes_data = pickle.dumps(data)
        batch_data = coworker_pb2.BatchData(data=bytes_data)
        return batch_data
