from torch.distributed.run import main

from dlrover.python.elastic_agent.torch.rdzv_backend import (
    register_dlrover_backend
)

register_dlrover_backend()


if __name__ == "__main__":
    main()
