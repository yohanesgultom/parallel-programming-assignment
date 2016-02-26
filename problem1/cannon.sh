clear && mpicc -o cannon.o cannon.c -lm &&
echo "NP\tData\tProcess Time" > cannon.result &&
mpiexec -np 4 ./cannon.o 360 >> cannon.result &&
mpiexec -np 4 ./cannon.o 360 >> cannon.result &&
mpiexec -np 4 ./cannon.o 360 >> cannon.result &&
mpiexec -np 4 ./cannon.o 360 >> cannon.result &&
mpiexec -np 4 ./cannon.o 360 >> cannon.result &&
mpiexec -np 4 ./cannon.o 1440 >> cannon.result &&
mpiexec -np 4 ./cannon.o 1440 >> cannon.result &&
mpiexec -np 4 ./cannon.o 1440 >> cannon.result &&
mpiexec -np 4 ./cannon.o 1440 >> cannon.result &&
mpiexec -np 4 ./cannon.o 1440 >> cannon.result
