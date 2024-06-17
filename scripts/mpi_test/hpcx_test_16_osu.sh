#!/bin/bash

#SBATCH -J vasp02

#SBATCH -p cnall #（使用cnall队列）

#SBATCH --nodes 16 # (使用2个节点)

#SBATCH -e /home/xiangd/WORK/zbw/ompi_build/openmpi-4.1.5/output/err/%j.err #（错误输出）

#SBATCH --ntasks-per-node=1 #（每个节点占用的核数）

#SBATCH --nodelist=ibc12b07n[37-40],ibc12b08n[37-40],ibc12b09n[37-40],ibc12b10n[37-40]


cd /home/xiangd/WORK/zbw/osu_mpi/osu_hpcx/mpi/collective
export HPCX_HOME=/home/xiangd/WORK/zbw/hpcx
source $HPCX_HOME/hpcx-init.sh
np=16

hpcx_load

export SLURM_OUTPUT="/home/xiangd/WORK/zbw/ompi_build/openmpi-4.1.5/output/20240114_16_osu/$SLURM_JOB_ID-hpcx-Algo$method-Mid$m_group_num-MidAlgo$MID_ALGO-Inplace$INPLACE.log"

/home/xiangd/WORK/zbw/hpcx/ompi/bin/mpirun -x UCX_TLS=dc,shm,self  -x HCOLL_ENABLE_SHARP=3 -x SHARP_COLL_ENABLE_SAT=1 \
-np $np ./osu_allreduce $@ > $SLURM_OUTPUT
hpcx_unload
