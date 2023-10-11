import os
import socket
from contextlib import closing

from torch.distributed.elastic.agent.server.api import SimpleElasticAgent, _get_fq_hostname, _get_socket_with_port


def hook_set_master_addr_port():
    def _hook(store, master_addr, master_port, local_dir=None):
        """
        PyTorch use master node's hostname as the MASTER_ADDR of process group. However, hostname may not be resolved
        in some Kubernetes environments. This function get master's ip address from POD_IP environment variable and
        set ip address as MASTER_ADDR.
        """
        if master_port is None:
            sock = _get_socket_with_port()
            with closing(sock):
                master_port = sock.getsockname()[1]

            if master_addr is None:
                if local_dir is not None:
                    master_addr = local_dir
                else:
                    master_addr = os.environ.get("POD_IP", socket.gethostbyname(_get_fq_hostname()))

        store.set("MASTER_ADDR", master_addr.encode(encoding="UTF-8"))
        store.set("MASTER_PORT", str(master_port).encode(encoding="UTF-8"))

    # hook SimpleElasticAgent._set_master_addr_port
    setattr(SimpleElasticAgent, "_set_master_addr_port", staticmethod(_hook))
