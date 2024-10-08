import pickle

import grpc
from google.protobuf import empty_pb2  # type: ignore

from atorch.common.constants import GrpcEnv
from atorch.common.util_func import wait_for_server_started

try:
    from atorch.protos import coworker_pb2, coworker_pb2_grpc  # type: ignore
except ImportError:
    coworker_pb2 = coworker_pb2_grpc = None  # type: ignore


class GpuPodRpcClient(object):
    """
    GPU Pod Rpc Clients will be created on Cowoker 1 ~ Coworker N-1.
    They call `report_data_info` through grpc to send data info to
    Data Info Service.
    """

    def __init__(self, grpc_addr):
        self._channel = grpc.insecure_channel(grpc_addr)
        self._stub = coworker_pb2_grpc.DataInfoServiceStub(self._channel)
        ip, port = grpc_addr.split(":")
        wait_for_server_started(ip, int(port), timeout=900)

    def __del__(self):
        self._channel.close()

    def report_data_info(self, coworker_addr, batch_num):
        """
        Args:
            coworker_addr: IP address and port of the Coworker where
                this GPU Pod Rpc Client is created.
            batch_num: The unique number of data created by Elastic Dataloader.
        """
        info = coworker_pb2.DataInfo(coworker_addr=coworker_addr, batch_num=batch_num)
        self._stub.report_data_info(info)
        return empty_pb2.Empty()


class DataInfoRpcClient(object):
    """
    Data info rpc clients will be created on every workers(ranks) on gpu pods.
    They will get data info from Data Info Service.
    """

    def __init__(self, grpc_addr):
        self._channel = grpc.insecure_channel(grpc_addr)
        self._stub = coworker_pb2_grpc.DataInfoServiceStub(self._channel)
        ip, port = grpc_addr.split(":")
        wait_for_server_started(ip, int(port), timeout=900)

    def __del__(self):
        self._channel.close()

    def get_data_info(self):
        request = empty_pb2.Empty()
        info = self._stub.get_data_info(request)
        return info


class CoworkerRpcClient(object):
    """
    Coworker rpc clients will be created on every workers(ranks) on gpu pods.
    They will get preprocessed data from coworkers' coworker rpc servers.
    """

    def __init__(self, grpc_addr):
        self._channel = grpc.insecure_channel(
            grpc_addr,
            options=[
                (
                    "grpc.max_send_message_length",
                    GrpcEnv.MAX_SEND_MESSAGE_LENGTH,
                ),
                (
                    "grpc.max_receive_message_length",
                    GrpcEnv.MAX_RECEIVE_MESSAGE_LENGTH,
                ),
            ],
        )
        self._stub = coworker_pb2_grpc.CoworkerRpcServiceStub(self._channel)
        ip, port = grpc_addr.split(":")
        wait_for_server_started(ip, int(port), timeout=900)

    def __del__(self):
        self._channel.close()

    def get_batch_data(self):
        request = empty_pb2.Empty()
        batch_data = self._stub.get_batch_data(request)
        return pickle.loads(batch_data.data)


def create_gpu_pod_rpc_clients(gpu_pod_addrs):
    return [GpuPodRpcClient(addr) for addr in gpu_pod_addrs]


def create_data_info_rpc_client(addr):
    return DataInfoRpcClient(addr)


def create_coworker_rpc_client(coworker_addrs):
    return {addr: CoworkerRpcClient(addr) for addr in coworker_addrs}
