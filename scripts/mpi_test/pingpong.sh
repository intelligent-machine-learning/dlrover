#!/bin/bash

#SBATCH -J vasp02

#SBATCH -p cnall #（使用cnall队列）

#SBATCH --nodes 1 # (使用2个节点)

#SBATCH -o /WORK/xiangd_work/zbw/ompi_build2/ompi-4.1.5/output/%j-ib.log #（屏幕输出）
#SBATCH -e /WORK/xiangd_work/zbw/ompi_build2/ompi-4.1.5/output/%j-ib.err #（屏幕输出）

#SBATCH --ntasks-per-node=2 #（每个节点占用的核数）


# cd /home/xiangd/WORK/zbw/ompi_build/openmpi-4.1.5
# /home/xiangd/WORK/zbw/ompi_build/my_ompi/bin/mpirun -np 2 ./rdma_ib
cd /home/xiangd/WORK/zbw/intel_mpi/src_c
# /home/xiangd/WORK/zbw/ompi_build/my_ompi/bin/mpirun -np 2 ./IMB-MPI1 PingPong 
# /home/xiangd/WORK/zbw/ompi_build/my_ompi/bin/mpirun -np 2 ./IMB-MPI1 PingPing
/home/xiangd/WORK/zbw/ompi_build/my_ompi/bin/mpirun -np 2 ./IMB-MPI1 PingPong
# /home/xiangd/WORK/zbw/ompi_build/my_ompi/bin/mpirun --mca pml ucx -np 2 ./IMB-MPI1 PingPong

# module load compilers/intel/oneapi-2023/config
# cd /home/xiangd/WORK/zbw/ompi_build/openmpi-4.1.5
# /apps/compilers/intel/oneapi/v2023.0.0.25537/mpi/2021.8.0/bin/mpirun -np 2 ./rdma_init

