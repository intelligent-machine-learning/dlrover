#!/bin/bash

#SBATCH -J vasp02

#SBATCH -p cnall #（使用cnall队列）

#SBATCH --nodes 16 # (使用2个节点)

#SBATCH -o /home/xiangd/WORK/zbw/ompi_build/openmpi-4.1.5/output/%j-osu.log #（屏幕输出）

#SBATCH -e /home/xiangd/WORK/zbw/ompi_build/openmpi-4.1.5/output/%j-osu.err #（错误输出）

#SBATCH --ntasks-per-node=1 #（每个节点占用的核数）

#SBATCH --nodelist=ibc12b07n[37-40],ibc12b08n[37-40],ibc12b09n[37-40],ibc12b10n[37-40]


cd /home/xiangd/WORK/zbw/intel_mpi/src_c_osu
echo /home/xiangd/WORK/zbw/osumpi_build/osu_mpi/bin/mpirun $@ #（可执行程序exe）
/home/xiangd/WORK/zbw/osumpi_build/osu_mpi/bin/mpirun $@ #（可执行程序exe）
