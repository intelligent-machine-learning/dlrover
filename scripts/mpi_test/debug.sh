#!/bin/bash

#SBATCH -J vasp02

#SBATCH -p cnall #（使用cnall队列）

#SBATCH --nodes 4 # (使用2个节点)

#SBATCH -o debug.txt #（屏幕输出）
#SBATCH -e /tmp/output.err #（屏幕输出）

#SBATCH --ntasks-per-node=1 #（每个节点占用的核数）

#SBATCH --nodelist=ibc13b08n[01-04]

cd /home/xiangd/WORK/xiangd_work/zbw/ompi_build/openmpi-4.1.5
/home/xiangd/WORK/zbw/ompi_build/my_ompi/bin/mpirun -n 4 ./mpi_test
