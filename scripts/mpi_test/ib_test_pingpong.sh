#!/bin/bash

#SBATCH -J vasp02

#SBATCH -p cnall #（使用cnall队列）

#SBATCH --nodes 2 # (使用2个节点)

#SBATCH -o /home/xiangd/WORK/zbw/ompi_build/openmpi-4.1.5/output/%j-ib.log #（屏幕输出）
#SBATCH -e /home/xiangd/WORK/zbw/ompi_build/openmpi-4.1.5/output/%j-ib.err #（屏幕输出）

#SBATCH --ntasks-per-node=1 #（每个节点占用的核数）


# cd /home/xiangd/WORK/zbw/ompi_build/openmpi-4.1.5
# /home/xiangd/WORK/zbw/ompi_build/my_ompi/bin/mpirun -np 2 ./rdma_ib
cd /home/xiangd/WORK/zbw/intel_mpi/src_c_rdma
# /home/xiangd/WORK/zbw/ompi_build/my_ompi/bin/mpirun -np 2 ./IMB-MPI1 PingPong 
/home/xiangd/WORK/zbw/ompi_build/my_ompi/bin/mpirun -np 2 ./IMB-MPI1 PingPong -msglen ../msg_len.txt

# module load compilers/intel/oneapi-2023/config
# cd /home/xiangd/WORK/zbw/ompi_build/openmpi-4.1.5
# /apps/compilers/intel/oneapi/v2023.0.0.25537/mpi/2021.8.0/bin/mpirun -np 2 ./rdma_init

