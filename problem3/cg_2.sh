#!/bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#
exe="mpi_conjugate_gradient"
data="data1024"
out="pcg.1024.fasilkom.result"
clear &&
mpirun -hostfile ~/mpi_hostfile -np 2 $exe 1024 0.000001 1000 < $data >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 3 $exe 1024 0.000001 1000 < $data >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 4 $exe 1024 0.000001 1000 < $data >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 5 $exe 1024 0.000001 1000 < $data >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 6 $exe 1024 0.000001 1000 < $data >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 7 $exe 1024 0.000001 1000 < $data >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 8 $exe 1024 0.000001 1000 < $data >> $out
