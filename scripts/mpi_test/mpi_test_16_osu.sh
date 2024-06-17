#!/bin/bash

#SBATCH -J vasp02

#SBATCH -p cnall #（使用cnall队列）

#SBATCH --nodes 16 # (使用2个节点)

#SBATCH -e /home/xiangd/WORK/zbw/ompi_build2/ompi-4.1.5/output/err/%j.err #（错误输出）

#SBATCH --ntasks-per-node=1 #（每个节点占用的核数）

echo SBATCH --nodelist=ibc12b02n[25-28],ibc12b03n[37-40],ibc12b07n[20-23],ibc12b08n[20-23]


cd /home/xiangd/WORK/zbw/osu_mpi/osu_openmpi/mpi/collective
method=${M_AR_METHOD:--1} 
m_group_num=${M_GROUP_NUM:-4}
return=${RETURN:-10}
np=16
MID_ALGO1=${MID_ALGO1:-6}
MID_ALGO2=${MID_ALGO2:-6}
INPLACE=${INPLACE:-0}

mkdir /home/xiangd/WORK/zbw/ompi_build2/ompi-4.1.5/output/output20240327_16osu
export SLURM_OUTPUT="/home/xiangd/WORK/zbw/ompi_build2/ompi-4.1.5/output/output20240327_16osu/$SLURM_JOB_ID-Algo$method-Mid$m_group_num-MidAlgo$MID_ALGO1,$MID_ALGO2-Inplace$INPLACE.log"

NP=$np M_AR_METHOD=$method M_GROUP_NUM=$m_group_num MID_ALGO1=$MID_ALGO1 MID_ALGO2=$MID_ALGO2 INPLACE=$INPLACE \
/home/xiangd/WORK/zbw/ompi_build/my_ompi/bin/mpirun -np $np ./osu_allreduce $@ > $SLURM_OUTPUT

