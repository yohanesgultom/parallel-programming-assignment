#!/bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#
name="mm_rowwise"
prog="$name.c"
exe="$name.o"
out="$name.fasilkom.result"
clear && mpicc -o $exe $prog -lm &&
echo "NP\tData\tComm Time\tProcess Time" > $out &&
mpirun -hostfile ~/mpi_hostfile -np 4 $exe 360 360 360 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 4 $exe 360 360 360 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 4 $exe 360 360 360 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 4 $exe 360 360 360 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 4 $exe 360 360 360 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 2 $exe 576 576 576 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 2 $exe 576 576 576 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 2 $exe 576 576 576 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 2 $exe 576 576 576 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 2 $exe 576 576 576 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 4 $exe 576 576 576 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 4 $exe 576 576 576 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 4 $exe 576 576 576 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 4 $exe 576 576 576 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 4 $exe 576 576 576 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 8 $exe 576 576 576 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 8 $exe 576 576 576 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 8 $exe 576 576 576 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 8 $exe 576 576 576 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 8 $exe 576 576 576 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 16 $exe 576 576 576 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 16 $exe 576 576 576 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 16 $exe 576 576 576 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 16 $exe 576 576 576 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 16 $exe 576 576 576 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 16 $exe 576 576 576 >> $out
