#!/bin/bash
name="manager"
name_worker="worker"
prog="$name.c"
prog_worker="$name_worker.c"
out="$name.result"
clear && mpicc -o $name_worker $prog_worker && mpicc -o $name $prog &&
echo "Workers\tComm Time\tProcess Time" > $out &&
mpiexec -n 1 $name 2 900 >> $out &&
mpiexec -n 1 $name 2 900 >> $out &&
mpiexec -n 1 $name 2 900 >> $out &&
mpiexec -n 1 $name 2 900 >> $out &&
mpiexec -n 1 $name 2 900 >> $out &&
mpiexec -n 1 $name 3 900 >> $out &&
mpiexec -n 1 $name 3 900 >> $out &&
mpiexec -n 1 $name 3 900 >> $out &&
mpiexec -n 1 $name 3 900 >> $out &&
mpiexec -n 1 $name 3 900 >> $out &&
mpiexec -n 1 $name 4 900 >> $out &&
mpiexec -n 1 $name 4 900 >> $out &&
mpiexec -n 1 $name 4 900 >> $out &&
mpiexec -n 1 $name 4 900 >> $out &&
mpiexec -n 1 $name 4 900 >> $out
