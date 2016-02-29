#!/bin/bash
name="mm_fox"
prog="$name.c"
exe="$name.o"
out="$name.rocks.result"
clear && mpicc -o $exe $prog -lm &&
echo "NP\tData\tComm Time\tProcess Time" > $out &&
mpiexec -np 2 $exe 500 >> $out &&
mpiexec -np 2 $exe 500 >> $out &&
mpiexec -np 2 $exe 500 >> $out &&
mpiexec -np 2 $exe 500 >> $out &&
mpiexec -np 2 $exe 500 >> $out &&
mpiexec -np 4 $exe 500 >> $out &&
mpiexec -np 4 $exe 500 >> $out &&
mpiexec -np 4 $exe 500 >> $out &&
mpiexec -np 4 $exe 500 >> $out &&
mpiexec -np 4 $exe 500 >> $out
