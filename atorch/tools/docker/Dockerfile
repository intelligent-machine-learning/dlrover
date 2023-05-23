##############################################################################
# default versions
# PYTHON_VERSION=3.8
# TORCH_VERSION=1.9.1
# CUDA_VERSON=11.1
##############################################################################

FROM reg.docker.alibaba-inc.com/alipay/7u2-common:202202.0T as base

USER root
WORKDIR /root

ENV BASH_ENV /root/.bashrc
ENV LANGUAGE zh_cn
ENV LC_ALL zh_CN.UTF-8
ENV SHELL /bin/bash

RUN mkdir -p ~/.pip && \
    echo -e "\n\
    [global]\n\
    index-url = https://artifacts.antgroup-inc.cn/simple/\n\
    trusted-host = artifacts.antgroup-inc.cn mirrors.aliyun.com\n\
    " | sed 's/^ \+//g' > ~/.pip/pip.conf

##############################################################################
# conda, python, osscmd, ossutil64, pangu, gcc6, libaio-devel, nfs, pyodps
# ca certs
##############################################################################
FROM base as util-base
ARG PYTHON_VERSION=3.8
ARG CONDA_PATH=/opt/conda
ENV PATH ${CONDA_PATH}/bin:$PATH
ARG CA_PATH=/etc/ssl/certs/ca-bundle.crt

RUN cd ~/ && \
    wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh --no-check-certificate -O miniconda-install.sh && \
    chmod +x miniconda-install.sh && \
    ~/miniconda-install.sh -b -p ${CONDA_PATH} && \
    rm ~/miniconda-install.sh && \
    conda install -y python=${PYTHON_VERSION} conda-build pyyaml numpy ipython && \
    conda clean -ya && \
    pip install osscmd aistudio-common typeguard easydict tensorboardX pyodps && \
    wget -O ${CONDA_PATH}/bin/ossutil64  https://gosspublic.alicdn.com/ossutil/1.7.7/ossutil64?spm=a2c4g.11186623.0.0.128b1c342qUHJs && \
    chmod a+x ${CONDA_PATH}/bin/ossutil64 && \
    yum install alios7u-2_24-gcc-8-repo -y && \
    yum install gcc libaio-devel -y && \
    wget "http://yum.tbsite.net/taobao/7/x86_64/current/pangu-client/pangu-client-0.16.1.1.1100190-7u2.x86_64.rpm" -O pangu-client.rpm && \
    rpm -ivh pangu-client.rpm && \
    rm -rf pangu-client.rpm && \
    yum install nfs-utils -y && \
    touch ${CA_PATH} && \
    wget -O - http://148757.oss-cn-hangzhou-zmf.aliyuncs.com/tmp/echoca_prod.sh | sh

##############################################################################
# rdma, cuda compat, cuda toolkit, torch 1.9.1
##############################################################################
FROM util-base as torch-base

ARG CUDA_VERSION=11-1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

# for non-interacvite login
ENV BASH_ENV /etc/bashrc

COPY tools/docker/handle_driver_compat.sh ./

RUN echo "/usr/local/cuda/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/lib64" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "ldconfig > /dev/null 2>&1 " >> /etc/bashrc && \
    yum install -y libnl3 && \
    wget http://dmsint.cn-hangzhou.alipay.aliyun-inc.com/aistudio%2Fgpu%2Frdma-a100.tgz -O /tmp/rdma-a100.tgz && \
    tar -xvf /tmp/rdma-a100.tgz -C /tmp && \
    rpm -Uvh /tmp/rdma-a100/nic-libs-mellanox-rdma-5.2-2.x86_64.rpm && \
    rm -rf /tmp/rdma-a100.tgz /tmp/rdma-a100/ && \
    wget -O /etc/yum.repos.d/nvidia-cuda.repo http://yum.tbsite.net/nvidia-cuda/rhel7/x86_64/cuda-rhel7.repo \
    && sed -i "s#https://developer.download.nvidia.com/compute/cuda/repos#http://yum.tbsite.net/nvidia-cuda#" /etc/yum.repos.d/nvidia-cuda.repo && \
    yum makecache && \
    yum install -y cuda-compat-${CUDA_VERSION}-455.45.01 cuda-toolkit-${CUDA_VERSION} --nogpgcheck && \
    rm -rf /usr/local/cuda && ln -s /usr/local/cuda-11.1 /usr/local/cuda && \
    rm -rf /var/cache/yum && \
    cat handle_driver_compat.sh >> /etc/bashrc && \
    rm -rf handle_driver_compat.sh

RUN wget -O torch-1.9.1+cu111-cp38-cp38-linux_x86_64.whl http://alps-common.oss-cn-hangzhou-zmf.aliyuncs.com/users%2Fya%2Ftorch-1.9.1%2Bcu111-cp38-cp38-linux_x86_64.whl && \
    wget -O ./torchvision-0.10.1+cu111-cp38-cp38-linux_x86_64.whl http://alps-common.oss-cn-hangzhou-zmf.aliyuncs.com/users%2Fya%2Ftorchvision-0.10.1%2Bcu111-cp38-cp38-linux_x86_64.whl && \
    pip --no-cache-dir install ./torch-1.9.1+cu111-cp38-cp38-linux_x86_64.whl ./torchvision-0.10.1+cu111-cp38-cp38-linux_x86_64.whl torchaudio==0.9.1 && \
    rm -rf ./torch-1.9.1+cu111-cp38-cp38-linux_x86_64.whl ./torchvision-0.10.1+cu111-cp38-cp38-linux_x86_64.whl

##############################################################################
# apex, fairscale, atorch
##############################################################################
FROM torch-base as atorch-base
# apex, fairscale, deepspeed
ENV USE_NCCL=1
ARG TORCH_CUDA_ARCH_LIST="6.0 7.0 7.5 8.0 8.6+PTX"
RUN wget -O apex.zip http://alps-common.oss-cn-hangzhou-zmf.aliyuncs.com/users%2Fya%2Fapex-master.zip && \
    unzip apex.zip -d ./ && \
    cd ./apex-master && \
    MAKEFLAGS="-j$(nproc)" MAX_JOBS=16 \
    pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ && \
    cd .. && rm -rf apex* && \
    yum install libnccl-2.16.2-1+cuda11.0 libnccl-devel-2.16.2-1+cuda11.0 -y && \
    pip install http://alipay-cognition.cn-hangzhou.alipay.aliyun-inc.com/datacube2/packages/library/feiliu/fastmoe.tar.gz && \
    pip install dm-tree && \
    pip install http://alipay-cognition.cn-hangzhou.alipay.aliyun-inc.com/datacube2/packages/debugger/zy267523/lib/flash-attention-ant-230224.tar.gz && \
    pip install kubernetes grpcio==1.34.1 grpcio-tools==1.34.1 && \
    pip install elasticdl elasticdl_client elasticai_api --no-deps && \
    pip install easydl_sdk==0.0.3

##############################################################################
# glpk
##############################################################################
FROM atorch-base as atorch-mip-base
RUN wget -O ./glpk-5.0.tar.gz http://alps-common.oss-cn-hangzhou-zmf.aliyuncs.com/users%2Fgaoning%2Fglpk-5.0.tar.gz && \
    tar -zxvf glpk-5.0.tar.gz && \
    cd ./glpk-5.0 && \
    ./configure --prefix=/opt && \
    make && make install && \
    cd .. && \
    rm -rf glpk-5.0.tar.gz glpk-5.0

ENV PATH="${PATH}:/opt/bin:/opt/lib"

##############################################################################
# image process tools
# imgaug, augly, opencv-python, nvidia-dali
##############################################################################   
FROM atorch-mip-base as image-process-base

RUN pip install imgaug augly opencv-python-headless && \
    pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda110

COPY atorch/requirements.txt ./
RUN pip install -r requirements.txt && \
    pip install --upgrade atorch \
