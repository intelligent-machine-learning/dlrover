#!/bin/sh
#****************************************************************#
# ScriptName: train.sh
# Author: $SHTERM_REAL_USER@alibaba-inc.com
# Create Date: 2024-12-26 10:27
# Modify Author: $SHTERM_REAL_USER@alibaba-inc.com
# Modify Date: 2025-08-07 14:48
# Function: 
#***************************************************************#
# export XPU_TIMER_SM_COUNT=20
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
# export  NCCL_MAX_NCHANNELS=20
#nsys profile --stats true -w true -t cuda,nvtx,osrt,cudnn,cublas xpu_timer_launch python -m torch.distributed.launch --nproc_per_node=6 train_llama.py 
# xpu_timer_launch 
export CUDA_VISIBLE_DEVICES=4,5,6,7
export XPU_TIMER_DEBUG_MODE=1
export XPU_TIMER_BASEPORT=28888
export NCCL_DEBUG=WARN
export WORLD_SIZE=4
export LOCAL_WORLD_SIZE=4

# export GLOG_v=5

# CUDA_DEVICE_MAX_CONNECTIONS=1 TORCH_NCCL_ENABLE_TIMING=1 
xpu_timer_launch python dlrm_s_pytorch.py --mini-batch-size=16 --data-size=1000000
