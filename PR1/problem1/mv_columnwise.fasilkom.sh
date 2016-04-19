#!/bin/bash
name="mv_columnwise"
prog="$name.c"
exe="$name.o"
out="$name.fasilkom.result"
clear && mpicc -o $exe $prog -lm &&
echo "NP\tData\tComm Time\tProcess Time" > $out &&
mpirun -hostfile ~/mpi_hostfile -np 5 $exe 360 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 5 $exe 360 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 5 $exe 360 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 5 $exe 360 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 5 $exe 360 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 3 $exe 1152 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 3 $exe 1152 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 3 $exe 1152 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 3 $exe 1152 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 3 $exe 1152 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 5 $exe 1152 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 5 $exe 1152 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 5 $exe 1152 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 5 $exe 1152 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 5 $exe 1152 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 7 $exe 1152 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 7 $exe 1152 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 7 $exe 1152 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 7 $exe 1152 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 7 $exe 1152 >> $out
