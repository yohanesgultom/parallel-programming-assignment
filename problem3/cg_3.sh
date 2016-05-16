#!/bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#
exe="mpi_conjugate_gradient"
data="data2048"
out="pcg.2048.fasilkom.result"
clear &&
mpirun -hostfile ~/mpi_hostfile -np 2 $exe 2048 0.000001 1000 < $data >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 3 $exe 2048 0.000001 1000 < $data >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 4 $exe 2048 0.000001 1000 < $data >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 5 $exe 2048 0.000001 1000 < $data >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 6 $exe 2048 0.000001 1000 < $data >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 7 $exe 2048 0.000001 1000 < $data >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 8 $exe 2048 0.000001 1000 < $data >> $out
