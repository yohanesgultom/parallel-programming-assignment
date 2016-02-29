#!/bin/bash
name="mm_rowwise"
prog="$name.c"
exe="$name.o"
out="$name.rocks.result"
clear && mpicc -o $exe $prog -lm &&
echo "NP\tData\tComm Time\tProcess Time" > $out &&
mpiexec -machinefile ~/machines -np 2 $exe 4000 4000 4000 >> $out &&
mpiexec -machinefile ~/machines -np 2 $exe 4000 4000 4000 >> $out &&
mpiexec -machinefile ~/machines -np 2 $exe 4000 4000 4000 >> $out &&
mpirun -machinefile ~/machines -np 2 $exe 4000 4000 4000 >> $out &&
mpirun -machinefile ~/machines -np 4 $exe 4000 4000 4000 >> $out &&
mpirun -machinefile ~/machines -np 4 $exe 4000 4000 4000 >> $out &&
mpirun -machinefile ~/machines -np 4 $exe 4000 4000 4000 >> $out &&
mpirun -machinefile ~/machines -np 4 $exe 4000 4000 4000 >> $out &&
mpirun -machinefile ~/machines -np 4 $exe 4000 4000 4000 >> $out
