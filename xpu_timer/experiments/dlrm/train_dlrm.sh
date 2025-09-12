if [[ "install" == $1 ]]; then
  pip config set global.index-url https://pypi.antfin-inc.com/artifact/repositories/simple
  pip install -e ../logging
  pip install tensorboard
  pip install --force-reinstall /prs/xpu_timer_whl/py_xpu_timer-1.1+cu124-cp311-cp311-linux_x86_64.whl
fi

#export CUDA_VISIBLE_DEVICES=4,5,6,7
export XPU_TIMER_DEBUG_MODE=1
export XPU_TIMER_BASEPORT=28888
export NCCL_DEBUG=WARN
export WORLD_SIZE=8
export LOCAL_WORLD_SIZE=8

#xpu_timer_launch python dlrm_s_pytorch.py --mini-batch-size=16 --data-size=1000000 --use-gpu
#python -m torch.distributed.launch --nproc_per_node=8 dlrm_s_pytorch.py --mini-batch-size=16 --data-size=1000000 --use-gpu
xpu_timer_launch python -m torch.distributed.launch --nproc_per_node=8 dlrm_s_pytorch.py --arch-embedding-size="80000-80000-80000-80000-80000-80000-80000-80000" --arch-sparse-feature-size=128 --arch-mlp-bot="128-128-128-128" --arch-mlp-top="512-512-512-256-1" --max-ind-range=40000000 --data-generation=random --loss-function=bce --round-targets=True --learning-rate=1.0 --mini-batch-size=2048 --print-freq=2 --print-time --test-freq=2 --test-mini-batch-size=2048 --memory-map --use-gpu --num-batches=100 --dist-backend=nccl
