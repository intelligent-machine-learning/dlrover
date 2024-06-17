#!/bin/bash

#SBATCH -J vasp02

#SBATCH -p cnall #（使用cnall队列）

#SBATCH --nodes NODES_NUM # (使用2个节点)

#SBATCH -e /home/xiangd/WORK/zbw/ompi_build2/ompi-4.1.5/output/err/%j.err #（错误输出）

#SBATCH --ntasks-per-node=1 #（每个节点占用的核数）

echo SBATCH --nodelist=ibc12b07n[19-23],ibc12b08n[19-23],ibc12b09n[19-23]


cd /home/xiangd/WORK/zbw/intel_mpi/src_c
method=${M_AR_METHOD:--1} 
m_group_num=${M_GROUP_NUM:-4}
return=${RETURN:-10}
np=NODES_NUM
MID_ALGO=${MID_ALGO:-6}
INPLACE=${INPLACE:-0}

mkdir /home/xiangd/WORK/zbw/ompi_build2/ompi-4.1.5/output/ring_nodes3
export SLURM_OUTPUT="/home/xiangd/WORK/zbw/ompi_build2/ompi-4.1.5/output/ring_nodes3/$SLURM_JOB_ID-N$np-Algo$method-Mid$m_group_num-MidAlgo$MID_ALGO-Inplace$INPLACE.log"


echo NP=$np M_AR_METHOD=$method M_GROUP_NUM=$m_group_num MID_ALGO=$MID_ALGO INPLACE=$INPLACE> $SLURM_OUTPUT
echo /home/xiangd/WORK/zbw/ompi_build/my_ompi/bin/mpirun -np $np ./IMB-MPI1 -npmin $np -npmax $np Allreduce $@ > $SLURM_OUTPUT #（可执行程序exe）
NP=$np M_AR_METHOD=$method M_GROUP_NUM=$m_group_num MID_ALGO=$MID_ALGO INPLACE=$INPLACE \
/home/xiangd/WORK/zbw/ompi_build/my_ompi/bin/mpirun -np $np ./IMB-MPI1 -npmin $np \
-msglen /WORK/xiangd_work/zbw/ompi_build2/ompi-4.1.5/backup/test/msglen.txt \
Allreduce $@ > $SLURM_OUTPUT

