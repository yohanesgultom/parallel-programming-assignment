clear && mpicc -o cannon.o cannon.c -lm &&
echo "NP\tData\tComm Time\tProcess Time" > cannon.result &&
mpiexec -np 4 ./cannon.o 360 >> cannon.result &&
mpiexec -np 4 ./cannon.o 360 >> cannon.result &&
mpiexec -np 4 ./cannon.o 360 >> cannon.result &&
mpiexec -np 4 ./cannon.o 360 >> cannon.result &&
mpiexec -np 4 ./cannon.o 360 >> cannon.result &&
mpiexec -np 4 ./cannon.o 500 >> cannon.result &&
mpiexec -np 4 ./cannon.o 500 >> cannon.result &&
mpiexec -np 4 ./cannon.o 500 >> cannon.result &&
mpiexec -np 4 ./cannon.o 500 >> cannon.result &&
mpiexec -np 4 ./cannon.o 500 >> cannon.result
