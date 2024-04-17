#!/bin/bash
source /root/.bashrc >/dev/null 2>&1
source /opt/conda/bin/activate >/dev/null 2>&1
echo "================================="
echo "PPU SDK Version:    ${PPU_VERSION}"
echo "CUDA Wrapper:       ${CUDA_SDK_VER}"
echo "PyTorch Version:    ${PYTORCH_VERSION}"
echo "Python Version:     $(python3 -c 'import torch; print(torch.__version__, end ="")')"
echo "================================="
test -f "${PPU_SDK}/envsetup.sh" && source "${PPU_SDK}/envsetup.sh"
export UMD_PLATFORM_TYPE=1
export HGGC_DRIVER_CANDIDATE=UMD
echo '[INFO] Get Dockerfile at /Dockerfile-ppu'
test -e /dev/alixpu_ctl || echo '[WARNING] /dev/alixpu_ctl not exist in container, ppu-smi cannot work well, add --device=/dev/alixpu_ctl to fix'
test -e /dev/alixpu || echo '[WARNING] /dev/alixpu not exist in container, PPU cannot work well, add --device=/dev/alixpu to fix'
test -n "$(find /dev -type c \( -name 'alixpu_ppu*' -o -name 'alixpu-cap*' \) -print -quit)" || echo '[WARNING] Not found any PPU or MIG device in container'
find /sys/kernel/debug -name alixpu_version_info -exec grep -h version {} + -quit
if which ppu-smi > /dev/null 2>&1; then ppu-smi -q | grep --color=never -e Version -e '^PPU'; fi
if [[ "$(df --output=size /dev/shm | tail -1 | xargs)" = '65536' ]]; then
    echo '[WARNING] shm-size 64M(docker default value) which may be not enough, add --shm-size=4g or bigger depend on your model'
fi
if [[ "$#" -eq 0 ]]; then
    exec "/bin/bash"
else
    exec "$@"
fi