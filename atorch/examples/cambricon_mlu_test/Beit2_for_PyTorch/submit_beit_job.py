from pypai.job import PythonJobBuilder
from pypai.conf import ExecConf, KMConf
from aistudio_common.openapi.models import DataStore
import argparse
from datetime import datetime


def main():
    args = parse_args()

    # https://aistudio.alipay.com/project/assets/docker-image/detail/6120124
    # torch==2.1.0+cu118
    image = "reg.docker.alibaba-inc.com/aii/aistudio:6120124-20231024151104"

    node_gpu_num = 8  # gpu number per node
    node_num = args.node_num    # node number
    cpu_core = 96     # pod cpu core
    memory = 800 * 1024   # pod memory size in MB
    disk = 800 * 1024  # pod disk size in MB
    gpu_type = ""
    app = "elasticdl"
    k8s_priority = "low"

    dt = datetime.now()
    time_stamp = dt.strftime("%Y%m%d-%H%M%S")

    command = """bash cmm_gpu_pcache.sh"""

    print(f"using {node_num} nodes. command is {repr(command)}")
    master = ExecConf(
        num=1,
        cpu=cpu_core,
        memory=memory,
        gpu_num=node_gpu_num,
        gpu_type=gpu_type,
        disk_m=disk,
    )

    worker = None
    if node_num > 1:
        worker = ExecConf(
            num=node_num-1,
            cpu=cpu_core,
            memory=memory,
            gpu_num=node_gpu_num,
            gpu_type=gpu_type,
            disk_m=disk,
        )

    km_conf = KMConf(
        image=image,
        # cluster=cluster
    )

    job = PythonJobBuilder(source_root='./',
                           command=command,
                           main_file='',
                           master=master,
                           worker=worker,
                           k8s_priority=k8s_priority,
                           k8s_app_name=app,
                           km_conf=km_conf,
                           runtime='pytorch',
                           rdma=True,
                           host_network=True,
                           name=f"beit2-{app}-{node_num}nodes",
                           )

    # job.run()
    job.run(enable_wait=False)  


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--node_num", type=int, default=1, help="Num of nodes."
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
