#!/bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#
name="mm_fox"
prog="$name.c"
exe="$name.o"
out="$name.fasilkom.result"
clear && mpicc -o $exe $prog -lm &&
echo "NP\tData\tComm Time\tProcess Time" > $out &&
mpirun -hostfile ~/mpi_hostfile -np 2 $exe 500 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 2 $exe 500 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 2 $exe 500 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 2 $exe 500 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 2 $exe 500 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 4 $exe 500 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 4 $exe 500 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 4 $exe 500 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 4 $exe 500 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 4 $exe 500 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 8 $exe 500 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 8 $exe 500 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 8 $exe 500 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 8 $exe 500 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 8 $exe 500 >> $out
