#!/bin/bash

#SBATCH -J vasp02

#SBATCH -p cnall #（使用cnall队列）

#SBATCH --nodes 32 # (使用2个节点)

#SBATCH -e /home/xiangd/WORK/zbw/ompi_build2/ompi-4.1.5/output/err/%j.err #（错误输出）

#SBATCH --ntasks-per-node=1 #（每个节点占用的核数）


cd /home/xiangd/WORK/zbw/intel_mpi/src_c_hpcx
export HPCX_HOME=/home/xiangd/WORK/zbw/hpcx
source $HPCX_HOME/hpcx-init.sh
np=32

hpcx_load

mkdir /home/xiangd/WORK/zbw/ompi_build2/ompi-4.1.5/output/output20240530_$np
export SLURM_OUTPUT="/home/xiangd/WORK/zbw/ompi_build2/ompi-4.1.5/output/output20240530_$np/$SLURM_JOB_ID-hpcx-Algo$method-Mid$m_group_num-MidAlgo$MID_ALGO-Inplace$INPLACE.log"

/home/xiangd/WORK/zbw/hpcx/ompi/bin/mpirun -x UCX_TLS=dc,shm,self  -x HCOLL_ENABLE_SHARP=3 -x SHARP_COLL_ENABLE_SAT=1 \
-np $np ./IMB-MPI1 -npmin $np $@ > $SLURM_OUTPUT
hpcx_unload
