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

for i in 16
do

    while squeue | grep -q $username; do
        sleep 1
    done
    I_MPI_ADJUST_ALLREDUCE=1 sbatch backup/test/ompi_test_$i.sh Allreduce

    while squeue | grep -q $username; do
        sleep 1
    done
    I_MPI_ADJUST_ALLREDUCE=2 sbatch backup/test/ompi_test_$i.sh Allreduce

    while squeue | grep -q $username; do
        sleep 1
    done
    I_MPI_ADJUST_ALLREDUCE=3 sbatch backup/test/ompi_test_$i.sh Allreduce


    while squeue | grep -q $username; do
        sleep 1
    done
    I_MPI_ADJUST_ALLREDUCE=5 sbatch backup/test/ompi_test_$i.sh Allreduce

    while squeue | grep -q $username; do
        sleep 1
    done
    I_MPI_ADJUST_ALLREDUCE=7 sbatch backup/test/ompi_test_$i.sh Allreduce

    while squeue | grep -q $username; do
        sleep 1
    done
    I_MPI_ADJUST_ALLREDUCE=8 sbatch backup/test/ompi_test_$i.sh Allreduce

    while squeue | grep -q $username; do
        sleep 1
    done
    I_MPI_ADJUST_ALLREDUCE=9 sbatch backup/test/ompi_test_$i.sh Allreduce

    while squeue | grep -q $username; do
        sleep 1
    done
    sbatch backup/test/ompi_test_$i.sh Allreduce

done
