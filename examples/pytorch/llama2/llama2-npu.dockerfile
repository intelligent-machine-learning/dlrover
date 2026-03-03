FROM ubuntu:18.04

ARG HOST_ASCEND_BASE=/usr/local/Ascend
ARG NNAE_PATH=/usr/local/Ascend/nnae/latest
ARG INSTALL_ASCEND_PKGS_SH=install_ascend_pkgs.sh
WORKDIR /tmp

# 系统包
RUN apt update && \
    apt install -y --no-install-recommends wget vim sudo bzip2 wget make tar curl g++ pkg-config unzip numactl \
    libopenblas-dev libblas3 liblapack3 liblapack-dev libblas-dev gfortran libhdf5-dev libffi-dev libicu60 libxml2 libbz2-dev libssl-dev git patch libfreetype6-dev pkg-config libpng-dev libgl1-mesa-glx liblzma-dev less htop && \
    apt clean && rm -rf /var/lib/apt/lists/*


ENV LD_LIBRARY_PATH=/usr/local/python3.8.16/lib: \
    PATH=/usr/local/python3.8.16/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# 安装python和pip
RUN umask 0022  && \
    curl -k https://repo.huaweicloud.com/python/3.8.16/Python-3.8.16.tar.xz -o Python-3.8.16.tar.xz && \
    tar -xf Python-3.8.16.tar.xz && cd Python-3.8.16 && ./configure --prefix=/usr/local/python3.8.16 --enable-shared && \
    make && make install && \
    ln -sf /usr/local/python3.8.16/bin/python3 /usr/bin/python3 && \
    ln -sf /usr/local/python3.8.16/bin/python3 /usr/bin/python && \
    ln -sf /usr/local/python3.8.16/bin/pip3 /usr/bin/pip3 && \
    ln -sf /usr/local/python3.8.16/bin/pip3 /usr/bin/pip && \
    cd .. && \
    rm -rf Python* && \
    mkdir -p ~/.pip  && \
    echo '[global] \n\
    index-url=http://mirrors.aliyun.com/pypi/simple/\n\
    trusted-host=mirrors.aliyun.com' >> ~/.pip/pip.conf && \
    pip3 install pip -U

#set env
ENV LD_LIBRARY_PATH=$TOOLKIT_PATH/x86_64-linux/lib64/:$TOOLKIT_PATH/fwkacllib/lib64/:/usr/local/python3.8.16/lib/python3.8/site-packages/torch/lib:/usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:/usr/local/Ascend/add-ons/:/usr/lib/aarch64_64-linux-gnu:$LD_LIBRARY_PATH \
PATH=$PATH:$TOOLKIT_PATH/fwkacllib/ccec_compiler/bin/:$TOOLKIT_PATH/toolkit/tools/ide_daemon/bin/ \
ASCEND_OPP_PATH=$TOOLKIT_PATH/opp/ \
OPTION_EXEC_EXTERN_PLUGIN_PATH=$TOOLKIT_PATH/fwkacllib/lib64/plugin/opskernel/libfe.so:$TOOLKIT_PATH/fwkacllib/lib64/plugin/opskernel/libaicpu_engine.so:$TOOLKIT_PATH/fwkacllib/lib64/plugin/opskernel/libge_local_engine.so \
PYTHONPATH=$TOOLKIT_PATH/fwkacllib/python/site-packages/:$TOOLKIT_PATH/fwkacllib/python/site-packages/auto_tune.egg/auto_tune:$TOOLKIT_PATH/fwkacllib/python/site-packages/schedule_search.egg:$PYTHONPATH \
ASCEND_AICPU_PATH=$TOOLKIT_PATH

# create user HwHiAiUser
RUN groupadd  HwHiAiUser -g 1000 && \
    useradd -d /home/HwHiAiUser -u 1000 -g 1000 -m -s /bin/bash HwHiAiUser


ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8

# 用户需要从PyTorch社区下载CPU版torch-2.1.0+cpu: https://pytorch.org/get-started/previous-versions/
COPY torch-2.1.0+cpu-cp38-cp38-linux_x86_64.whl .
# 用户需要从昇腾社区下载torch-npu和CANN包(7.0.1.1版本): 
# https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/262097058?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C251168373
COPY torch_npu-2.1.0.post2-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl .
COPY Ascend-cann-toolkit_7.0.1.1_linux-x86_64.run .
COPY Ascend-cann-kernels-910b_7.0.1.1_linux.run .

RUN pip3 install cmake dm-tree packaging ninja setuptools wheel && \
    pip3 install attrs cloudpickle decorator numpy==1.23.1 protobuf==3.20.3 psutil requests setuptools sympy==1.4 tqdm typing_extensions wheel -i http://mirrors.aliyun.com/pypi/simple/ && \
    pip3 install torch-2.1.0+cpu-cp38-cp38-linux_x86_64.whl --force-reinstall && \
    umask 0022 && chmod +x *.run && \
    pip3 install ./torch_npu-2.1.0.post2-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl --force-reinstall --no-deps && \
    echo y | ./Ascend-cann-toolkit_7.0.1.1_linux-x86_64.run --install-path=/usr/local/Ascend/ --install --install-for-all && \
    echo y | ./Ascend-cann-kernels-910b_7.0.1.1_linux.run --install && \
    rm -rf Ascend-cann* torch-2.1.0* torch_npu*


RUN apt update && \
    apt install -y --no-install-recommends inetutils-ping nfs-common net-tools iproute2 telnet zip gdb ethtool tmux make dnsutils


RUN pip3 install accelerate==0.24.1 \
    appdirs==1.4.4 \
    black \
    fire==0.5.0 \
    datasets==2.14.6 \
    loralib \
    peft \
    matplotlib \
    transformers==4.37.2 \
    sentencepiece==0.1.97 \
    tensorboardX==2.6 \
    gradio==3.23.0 \
    sentencepiece \
    scipy \
    py7zr

