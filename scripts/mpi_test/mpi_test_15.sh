#!/bin/bash

#SBATCH -J vasp02

#SBATCH -p cnall #（使用cnall队列）

#SBATCH --nodes 15 # (使用2个节点)

#SBATCH -e /home/xiangd/WORK/zbw/ompi_build2/ompi-4.1.5/output/err/%j.err #（错误输出）

#SBATCH --ntasks-per-node=1 #（每个节点占用的核数）


cd /home/xiangd/WORK/zbw/intel_mpi/src_c
method=${M_AR_METHOD:--1} 
return=${RETURN:-10}
np=15
INPLACE=${INPLACE:-0}

mkdir /home/xiangd/WORK/zbw/ompi_build2/ompi-4.1.5/output/output20240612_$np
export SLURM_OUTPUT="/home/xiangd/WORK/zbw/ompi_build2/ompi-4.1.5/output/output20240612_$np/$SLURM_JOB_ID-Algo$method-Mid$m_group_num-MidAlgo$MID_ALGO1,$MID_ALGO2-Inplace$INPLACE.log"

echo NP=$np M_AR_METHOD=$method INPLACE=$INPLACE> $SLURM_OUTPUT
echo /home/xiangd/WORK/zbw/ompi_build/my_ompi/bin/mpirun -np $np ./IMB-MPI1 -npmin $np -npmax $np Allreduce $@ > $SLURM_OUTPUT #（可执行程序exe）
NP=$np M_AR_METHOD=$method INPLACE=$INPLACE \
/home/xiangd/WORK/zbw/ompi_build/my_ompi/bin/mpirun -np $np ./IMB-MPI1 -npmin $np \
 $@ > $SLURM_OUTPUT

