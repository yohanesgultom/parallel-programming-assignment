#!/bin/bash
name="mv_columnwise"
prog="$name.c"
exe="$name.o"
out="$name.rocks.result"
clear && mpicc -o $exe $prog -lm &&
echo "NP\tData\tComm Time\tProcess Time" > $out &&
mpirun -np 3 -machinefile ~/machines $exe 1152 >> $out &&
mpirun -np 3 -machinefile ~/machines $exe 1152 >> $out &&
mpirun -np 3 -machinefile ~/machines $exe 1152 >> $out &&
mpirun -np 3 -machinefile ~/machines $exe 1152 >> $out
