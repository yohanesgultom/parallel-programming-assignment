#!/bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#
name="dns"
exe="$name"
out="$name.825.fasilkom.result"
clear && make &&
mpirun -hostfile ~/mpi_hostfile -np 8 $exe 512 2 0.5 0.5 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 8 $exe 1024 2 0.5 0.5 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 8 $exe 2048 2 0.5 0.5 >> $out &&
mpirun -hostfile ~/mpi_hostfile -np 8 $exe 4096 2 0.5 0.5 >> $out
