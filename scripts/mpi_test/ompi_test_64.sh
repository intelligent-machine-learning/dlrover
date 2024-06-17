#!/bin/bash

#SBATCH -J vasp02

#SBATCH -p cnall #（使用cnall队列）

#SBATCH --nodes 64 # (使用2个节点)

#SBATCH -e /home/xiangd/WORK/zbw/ompi_build2/ompi-4.1.5/output/err/%j-intel.err #（错误输出）

#SBATCH --ntasks-per-node=1 #（每个节点占用的核数）



module load compilers/intel/oneapi-2023/config
cd /home/xiangd/WORK/zbw/intel_mpi/src_c_o
method=${M_AR_METHOD:--1} 
m_group_num=${M_GROUP_NUM:-4}
return=${RETURN:-10}
np=64
MID_ALGO=${MID_ALGO:-6}
INPLACE=${INPLACE:-0}

mkdir /WORK/xiangd_work/zbw/ompi_build2/ompi-4.1.5/output/output20240530_$np
export SLURM_OUTPUT="/WORK/xiangd_work/zbw/ompi_build2/ompi-4.1.5/output/output20240530_$np/$SLURM_JOB_ID-intel.log"

echo mpirun -np $np ./IMB-MPI1 -npmin $np Allreduce $@ > $SLURM_OUTPUT #（可执行程序exe）
I_MPI_HYDRA_IFACE="ib0" mpirun -np $np ./IMB-MPI1 -npmin $np  \
 $@ > $SLURM_OUTPUT
