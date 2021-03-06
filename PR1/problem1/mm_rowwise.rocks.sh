#!/bin/bash
name="mm_rowwise"
prog="$name.c"
exe="$name.o"
out="$name.rocks.result"
clear && mpicc -o $exe $prog -lm &&
echo "NP\tData\tComm Time\tProcess Time" > $out &&
mpirun -machinefile ~/machines -np 5 $exe 360 360 360 >> $out &&
mpirun -machinefile ~/machines -np 5 $exe 360 360 360 >> $out &&
mpirun -machinefile ~/machines -np 5 $exe 360 360 360 >> $out &&
mpirun -machinefile ~/machines -np 5 $exe 360 360 360 >> $out &&
mpirun -machinefile ~/machines -np 5 $exe 360 360 360 >> $out &&
mpirun -machinefile ~/machines -np 3 $exe 576 576 576 >> $out &&
mpirun -machinefile ~/machines -np 3 $exe 576 576 576 >> $out &&
mpirun -machinefile ~/machines -np 3 $exe 576 576 576 >> $out &&
mpirun -machinefile ~/machines -np 3 $exe 576 576 576 >> $out &&
mpirun -machinefile ~/machines -np 5 $exe 576 576 576 >> $out &&
mpirun -machinefile ~/machines -np 5 $exe 576 576 576 >> $out &&
mpirun -machinefile ~/machines -np 5 $exe 576 576 576 >> $out &&
mpirun -machinefile ~/machines -np 5 $exe 576 576 576 >> $out &&
mpirun -machinefile ~/machines -np 5 $exe 576 576 576 >> $out
