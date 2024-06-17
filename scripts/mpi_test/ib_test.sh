#!/bin/bash

#SBATCH -J vasp02

#SBATCH -p cnall #（使用cnall队列）

#SBATCH --nodes 2 # (使用2个节点)

#SBATCH -o /home/xiangd/WORK/zbw/ompi_build/openmpi-4.1.5/output.log #（屏幕输出）
#SBATCH -e /home/xiangd/WORK/zbw/ompi_build/openmpi-4.1.5/output.err #（屏幕输出）

#SBATCH --ntasks-per-node=1 #（每个节点占用的核数）

export SLURM_OUTPUT="/home/xiangd/WORK/zbw/ompi_build/openmpi-4.1.5/output/$SLURM_JOB_ID-ib.log"
cd /home/xiangd/WORK/zbw/ompi_build/openmpi-4.1.5
./rdma_init
/home/xiangd/WORK/zbw/ompi_build/my_ompi/bin/mpirun -np 2 ./rdma_ib > $SLURM_OUTPUT
# cd /home/xiangd/WORK/zbw/intel_mpi/src_c_rdma
# /home/xiangd/WORK/zbw/ompi_build/my_ompi/bin/mpirun -np 2 ./IMB-MPI1 PingPong

# module load compilers/intel/oneapi-2023/config
# cd /home/xiangd/WORK/zbw/ompi_build/openmpi-4.1.5
# /apps/compilers/intel/oneapi/v2023.0.0.25537/mpi/2021.8.0/bin/mpirun -np 2 ./rdma_init

