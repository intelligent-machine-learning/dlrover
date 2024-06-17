#!/bin/bash

#SBATCH -J vasp02

#SBATCH -p cnall #（使用cnall队列）

#SBATCH --nodes 1 # (使用2个节点)

#SBATCH -e /home/xiangd/WORK/zbw/ompi_build/openmpi-4.1.5/output/err/%j-intel.err #（错误输出）

#SBATCH --ntasks-per-node=16 #（每个节点占用的核数）

module load compilers/intel/oneapi-2023/config
cd /home/xiangd/WORK/zbw/intel_mpi/src_c_o
method=${M_AR_METHOD:--1} 
m_group_num=${M_GROUP_NUM:-4}
return=${RETURN:-10}
np=16
MID_ALGO=${MID_ALGO:-6}
INPLACE=${INPLACE:-0}

export SLURM_OUTPUT="/home/xiangd/WORK/zbw/ompi_build/openmpi-4.1.5/output/output20240114_intra/$SLURM_JOB_ID-intel-intra-Algo$method-Mid$m_group_num-MidAlgo$MID_ALGO-Inplace$INPLACE.log"


echo NP=$np M_AR_METHOD=$method M_GROUP_NUM=$m_group_num MID_ALGO=$MID_ALGO INPLACE=$INPLACE> $SLURM_OUTPUT
echo /apps/compilers/intel/oneapi/v2023.0.0.25537/mpi/2021.8.0/bin/mpirun -np $np ./IMB-MPI1 -npmin $np -npmax $np Allreduce $@ > $SLURM_OUTPUT #（可执行程序exe）
NP=$np M_AR_METHOD=$method M_GROUP_NUM=$m_group_num MID_ALGO=$MID_ALGO INPLACE=$INPLACE \
/apps/compilers/intel/oneapi/v2023.0.0.25537/mpi/2021.8.0/bin/mpirun -np $np ./IMB-MPI1 -npmin $np -npmax $np  \
Allreduce $@ > $SLURM_OUTPUT
