#!/bin/bash
name="mv_checkerboard"
prog="$name.c"
exe="$name.o"
out="$name.fasilkom.result"
clear && mpicc -o $exe $prog -lm &&
echo "NP\tData\tComm Time\tProcess Time" > $out &&
mpiexec -np 5 $exe 360 >> $out &&
mpiexec -np 5 $exe 360 >> $out &&
mpiexec -np 5 $exe 360 >> $out &&
mpiexec -np 5 $exe 360 >> $out &&
mpiexec -np 5 $exe 360 >> $out &&
mpiexec -np 5 $exe 1152 >> $out &&
mpiexec -np 5 $exe 1152 >> $out &&
mpiexec -np 5 $exe 1152 >> $out &&
mpiexec -np 5 $exe 1152 >> $out &&
mpiexec -np 5 $exe 1152 >> $out &&
mpiexec -np 17 $exe 1152 >> $out &&
mpiexec -np 17 $exe 1152 >> $out &&
mpiexec -np 17 $exe 1152 >> $out &&
mpiexec -np 17 $exe 1152 >> $out &&
mpiexec -np 17 $exe 1152 >> $out
