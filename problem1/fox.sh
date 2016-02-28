clear && mpicc -o fox.o fox.c -lm &&
echo "NP\tData\tComm Time\tProcess Time" > fox.result &&
mpiexec -np 4 ./fox.o 360 >> fox.result &&
mpiexec -np 4 ./fox.o 360 >> fox.result &&
mpiexec -np 4 ./fox.o 360 >> fox.result &&
mpiexec -np 4 ./fox.o 360 >> fox.result &&
mpiexec -np 4 ./fox.o 360 >> fox.result &&
mpiexec -np 4 ./fox.o 500 >> fox.result &&
mpiexec -np 4 ./fox.o 500 >> fox.result &&
mpiexec -np 4 ./fox.o 500 >> fox.result &&
mpiexec -np 4 ./fox.o 500 >> fox.result &&
mpiexec -np 4 ./fox.o 500 >> fox.result
