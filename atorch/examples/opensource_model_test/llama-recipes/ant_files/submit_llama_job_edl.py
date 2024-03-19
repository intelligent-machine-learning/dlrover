import argparse
from pypai.job import PythonJobBuilder
from pypai.conf import ExecConf, KMConf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--node_num", type=int, default=1, help="Num of nodes."
    )
    parser.add_argument(
        "--cpu_memory", type=int, default=1228800, help="MB"
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    app = "elasticdl"
    image = "reg.docker.alibaba-inc.com/aii/aistudio:6300124-20240110142926"

    node_gpu_num = 8

    worker = ExecConf(
        cpu=96, memory=800*1024, gpu_num=node_gpu_num, num=args.node_num, disk_m=400*1024, gpu_type="a100",
    )

    ## $NODE_NUM 是 EasyDL Job 根据用户配置的节点数量自动配置的环境变量。
    command = f"""printenv | sort && \
    nvidia-smi && \
    pip install dlrover[torch]==0.4.0 -U -i https://artifacts.antgroup-inc.cn/simple/ && \
    bash ant_files/env.sh && bash ant_files/run.sh"""

    edl_job_args = """ --relaunch_always """  # 设置Pod 重启策略，只要出错就重启 Pod，默认最多重启3次.

    data_stores = []

    # =================================
    # runtime='easydl' 提交弹性容错作业
    # distribution_strategy="AllreduceStrategy" 指定运行作业为 allreduce
    # 如果需要使用 RDMA 在PythonJobBuilder中加上 rdma=True, host_network=True,
    # EasyDL的弹性容错job 只需要配置worker，无需配置 master。
    # =================================
    job = PythonJobBuilder(source_root="./",
                           command=command,
                           platform="kubemaker",
                           main_file="",
                           worker=worker,
                           k8s_priority="low",  # 配置 GPU 资源优先级
                           k8s_app_name=app,
                           km_conf=KMConf(image=image),
                           runtime="easydl",
                           distribution_strategy="AllreduceStrategy",
                           data_stores=data_stores,  # 挂载 DataStore
                           rdma=True,  # 开启 RDMA
                           host_network=True,  # 开启 host network, rdma=True 时需要为True。
                           edl_job_args=edl_job_args,  # 指定 EasyDL job 参数
                           tag="type=SFT,basemodel=bailing-10B-Base",
                          )
    job.run(enable_wait=False)


if __name__ == '__main__':
    main()
