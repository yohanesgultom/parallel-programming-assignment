#!/bin/bash
name="manager"
name_worker="worker"
prog="$name.c"
prog_worker="$name_worker.c"
out="$name.nbcr.result"
clear && mpicc -o $name_worker $prog_worker && mpicc -o $name $prog &&
echo "Workers\tProcess Time" > $out &&
mpiexec $name 4 420 >> $out &&
mpiexec $name 4 420 >> $out &&
mpiexec $name 4 420 >> $out &&
mpiexec $name 4 420 >> $out &&
mpiexec $name 4 420 >> $out &&
mpiexec $name 5 420 >> $out &&
mpiexec $name 5 420 >> $out &&
mpiexec $name 5 420 >> $out &&
mpiexec $name 5 420 >> $out &&
mpiexec $name 5 420 >> $out &&
mpiexec $name 6 420 >> $out &&
mpiexec $name 6 420 >> $out &&
mpiexec $name 6 420 >> $out &&
mpiexec $name 6 420 >> $out &&
mpiexec $name 6 420 >> $out &&
mpiexec $name 7 420 >> $out &&
mpiexec $name 7 420>> $out &&
mpiexec $name 7 420>> $out &&
mpiexec $name 7 420>> $out &&
mpiexec $name 7 420>> $out
