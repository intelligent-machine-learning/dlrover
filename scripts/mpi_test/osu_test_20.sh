#!/bin/bash

#SBATCH -J vasp02

#SBATCH -p cnall #（使用cnall队列）

#SBATCH --nodes 20 # (使用2个节点)

#SBATCH -e /WORK/xiangd_work/zbw/ompi_build2/ompi-4.1.5/output/err/%j-intel.err #（错误输出）

#SBATCH --ntasks-per-node=1 #（每个节点占用的核数）


cd /home/xiangd/WORK/zbw/intel_mpi/src_c_osu
method=${M_AR_METHOD:--1} 
m_group_num=${M_GROUP_NUM:-4}
return=${RETURN:-10}
np=20
MID_ALGO=${MID_ALGO:-6}
INPLACE=${INPLACE:-0}

mkdir /WORK/xiangd_work/zbw/ompi_build2/ompi-4.1.5/output/output20240612_$np
export SLURM_OUTPUT="/WORK/xiangd_work/zbw/ompi_build2/ompi-4.1.5/output/output20240612_$np/$SLURM_JOB_ID-osu-Algo$method-Mid$m_group_num-MidAlgo$MID_ALGO-Inplace$INPLACE.log"


echo NP=$np M_AR_METHOD=$method M_GROUP_NUM=$m_group_num MID_ALGO=$MID_ALGO INPLACE=$INPLACE> $SLURM_OUTPUT
echo /home/xiangd/WORK/zbw/osumpi_build/osu_mpi/bin/mpirun -np $np ./IMB-MPI1 -npmin $np -npmax $np Allreduce $@ > $SLURM_OUTPUT #（可执行程序exe）
NP=$np M_AR_METHOD=$method M_GROUP_NUM=$m_group_num MID_ALGO=$MID_ALGO INPLACE=$INPLACE \
/home/xiangd/WORK/zbw/osumpi_build/osu_mpi/bin/mpirun -np $np ./IMB-MPI1 -npmin $np -npmax $np  \
 $@ > $SLURM_OUTPUT
