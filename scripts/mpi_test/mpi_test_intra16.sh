#!/bin/bash

#SBATCH -J vasp02

#SBATCH -p cnall #（使用cnall队列）

#SBATCH --nodes 1 # (使用2个节点)

#SBATCH -e /home/xiangd/WORK/zbw/ompi_build2/ompi-4.1.5/output/err/%j.err #（错误输出）

#SBATCH --ntasks-per-node=16 #（每个节点占用的核数）


cd /home/xiangd/WORK/zbw/intel_mpi/src_c
method=${M_AR_METHOD:--1} 
m_group_num=${M_GROUP_NUM:-4}
return=${RETURN:-10}
np=16
MID_ALGO1=${MID_ALGO1:-6}
MID_ALGO2=${MID_ALGO2:-6}
INPLACE=${INPLACE:-0}

mkdir /home/xiangd/WORK/zbw/ompi_build2/ompi-4.1.5/output/output20240612_intra$np
export SLURM_OUTPUT="/home/xiangd/WORK/zbw/ompi_build2/ompi-4.1.5/output/output20240612_intra$np/$SLURM_JOB_ID-Algo$method-Mid$m_group_num-MidAlgo$MID_ALGO1,$MID_ALGO2-intra.log"


echo NP=$np M_AR_METHOD=$method M_GROUP_NUM=$m_group_num MID_ALGO1=$MID_ALGO1 MID_ALGO2=$MID_ALGO2 INPLACE=$INPLACE> $SLURM_OUTPUT
echo /home/xiangd/WORK/zbw/ompi_build/my_ompi/bin/mpirun -np $np ./IMB-MPI1 -npmin $np -npmax $np Allreduce $@ > $SLURM_OUTPUT #（可执行程序exe）
NP=$np M_AR_METHOD=$method M_GROUP_NUM=$m_group_num MID_ALGO1=$MID_ALGO1 MID_ALGO2=$MID_ALGO2 INPLACE=$INPLACE \
/home/xiangd/WORK/zbw/ompi_build/my_ompi/bin/mpirun -np $np ./IMB-MPI1 -npmin $np \
 $@ > $SLURM_OUTPUT

