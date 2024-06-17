username=xiangd

# 等待所有作业完成
# for i in {4..32}
# do
#     while squeue | grep -q $username; do
#         sleep 1
#     done
#     python3 /WORK/xiangd_work/zbw/ompi_build2/ompi-4.1.5/backup/test/mpi_test_nodes.py -n $i
#     M_AR_METHOD=4 sbatch /tmp/nodes.sh
# done
for i in 16 32
do 
    echo $i
done

for i in intra16
do
    while squeue | grep -q $username; do
        sleep 1
    done
    M_AR_METHOD=3 sbatch backup/test/mpi_test_$i.sh Allreduce

    while squeue | grep -q $username; do
        sleep 1
    done
    M_AR_METHOD=4 sbatch backup/test/mpi_test_$i.sh Allreduce

    while squeue | grep -q $username; do
        sleep 1
    done
    M_AR_METHOD=6 sbatch backup/test/mpi_test_$i.sh Allreduce

    while squeue | grep -q $username; do
        sleep 1
    done
    M_AR_METHOD=-1 sbatch backup/test/mpi_test_$i.sh Allreduce

    while squeue | grep -q $username; do
        sleep 1
    done
    sbatch backup/test/ompi_test_$i.sh Allreduce

    while squeue | grep -q $username; do
        sleep 1
    done
    MV2_SHMEM_ALLREDUCE_MSG=0 sbatch backup/test/osu_test_$i.sh Allreduce

    while squeue | grep -q $username; do
        sleep 1
    done
    MV2_SHMEM_ALLREDUCE_MSG=2147483648 sbatch backup/test/osu_test_$i.sh Allreduce

    while squeue | grep -q $username; do
        sleep 1
    done
    sbatch backup/test/hpcx_test_$i.sh Allreduce

    while squeue | grep -q $username; do
        sleep 1
    done
    M_AR_METHOD=8 sbatch backup/test/mpi_test_$i.sh Allreduce

    # while squeue | grep -q $username; do
    #     sleep 1
    # done
    # M_AR_METHOD=ring sbatch backup/test/ompi_test_$i.sh Allreduce

    # while squeue | grep -q $username; do
    #     sleep 1
    # done
    # M_AR_METHOD=rabenseifner sbatch backup/test/ompi_test_$i.sh Allreduce

    # while squeue | grep -q $username; do
    #     sleep 1
    # done
    # M_AR_METHOD=recursive_doubling sbatch backup/test/ompi_test_$i.sh Allreduce

    # while squeue | grep -q $username; do
    #     sleep 1
    # done
    # M_AR_METHOD=2d sbatch backup/test/ompi_test_$i.sh Allreduce

done

# for i in 15 20
# do
#     while squeue | grep -q $username; do
#         sleep 1
#     done
#     sbatch backup/test/hpcx_test_$i.sh Allreduce
# done
