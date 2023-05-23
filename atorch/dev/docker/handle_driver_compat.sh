MIN_DIRVER_VERSION=450.80.02
CUDA_HOME=/usr/local/cuda

version_lte(){
    [  "$1" = "`echo -e "$1\n$2" | sort -V | head -n1`" ];
}
version_lt(){
    [ "$1" = "$2" ] && return 1 || version_lte $1 $2;
}
which nvidia-smi >> /dev/null 2>&1 && \
version_lt `nvidia-smi --query-gpu=driver_version --format=csv,noheader | awk 'NR==1{print}'` ${MIN_DIRVER_VERSION} && \
export LD_LIBRARY_PATH=${CUDA_HOME}/compat:$LD_LIBRARY_PATH

