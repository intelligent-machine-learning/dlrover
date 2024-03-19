#!/bin/bash
set -euxo pipefail

pip install -U --no-deps atorch==1.2.0 -i https://artifacts.antgroup-inc.cn/simple/ && \
pip install -U transformers==4.37.2 && \
pip install -r requirements.txt

# upgrade torch to 2.1.2 to use SPDA
torch_version_match() {
    version=$1
    if [[ $version =~ ^2\.1\.2\+cu(118|121)$ ]]; then
        return 0 # 符合要求
    else
        return 1 # 不符合要求
    fi
}

torch_version=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)

if torch_version_match "$torch_version"; then
    echo "torch==$torch_version is already installed."
else
    echo "Installing torch==2.1.2..."
    pip install -U --no-deps http://alps-common.oss-cn-hangzhou-zmf.aliyuncs.com/users%2Fsichuan%2Ftorch-2.1.2%2Bcu121-cp38-cp38-linux_x86_64.whl \
                         http://alps-common.oss-cn-hangzhou-zmf.aliyuncs.com/users%2Fsichuan%2Ftorchaudio-2.1.2%2Bcu121-cp38-cp38-linux_x86_64.whl \
                         http://alps-common.oss-cn-hangzhou-zmf.aliyuncs.com/users%2Fsichuan%2Ftorchvision-0.16.2%2Bcu121-cp38-cp38-linux_x86_64.whl \
                         http://alps-common.oss-cn-hangzhou-zmf.aliyuncs.com/users%2Fsichuan%2Ftriton-2.1.0-0-cp38-cp38-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
fi



# mount nas
if [ ! -d /datacube_nas ]; then
    sudo mkdir /datacube_nas
fi
if [ "`ls -A /datacube_nas`" = "" ]; then
    sudo mount -t nfs -o vers=3,nolock,proto=tcp,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport alipayshnas-002-kya2.cn-shanghai-eu13-a01.nas.aliyuncs.com:/ /datacube_nas
fi

pip list
