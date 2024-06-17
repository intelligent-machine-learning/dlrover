#!/bin/bash

#SBATCH -J vasp02

#SBATCH -p cnall #（使用cnall队列）

#SBATCH --nodes 1 # (使用2个节点)

#SBATCH -e /home/xiangd/WORK/zbw/ompi_build/openmpi-4.1.5/output/err/%j-intel.err #（错误输出）

#SBATCH --ntasks-per-node=16 #（每个节点占用的核数）


cd /home/xiangd/WORK/zbw/intel_mpi/src_c_osu
method=${M_AR_METHOD:--1} 
m_group_num=${M_GROUP_NUM:-4}
return=${RETURN:-10}
np=16
MID_ALGO=${MID_ALGO:-6}
INPLACE=${INPLACE:-0}
INPMV2_SHMEM_ALLREDUCE_MSG=${INPMV2_SHMEM_ALLREDUCE_MSG:-32768}

export SLURM_OUTPUT="/home/xiangd/WORK/zbw/ompi_build/openmpi-4.1.5/output/output20240114_intra/$SLURM_JOB_ID-osu-Algo$method-Mid$m_group_num-MidAlgo$MID_ALGO-Inplace$INPLACE.log"


echo INPMV2_SHMEM_ALLREDUCE_MSG=$INPMV2_SHMEM_ALLREDUCE_MSG NP=$np M_AR_METHOD=$method M_GROUP_NUM=$m_group_num MID_ALGO=$MID_ALGO INPLACE=$INPLACE> $SLURM_OUTPUT
echo /home/xiangd/WORK/zbw/osumpi_build/osu_mpi/bin/mpirun -np $np ./IMB-MPI1 -npmin $np -npmax $np Allreduce $@ > $SLURM_OUTPUT #（可执行程序exe）
INPMV2_SHMEM_ALLREDUCE_MSG=$INPMV2_SHMEM_ALLREDUCE_MSG NP=$np M_AR_METHOD=$method M_GROUP_NUM=$m_group_num MID_ALGO=$MID_ALGO INPLACE=$INPLACE \
/home/xiangd/WORK/zbw/osumpi_build/osu_mpi/bin/mpirun -np $np ./IMB-MPI1 -npmin $np -npmax $np  \
Allreduce $@ > $SLURM_OUTPUT
