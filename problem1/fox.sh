clear && mpicc -o fox.o fox.c -lm &&
echo "NP\tData\tProcess Time" > fox.result &&
mpiexec -np 4 ./fox.o 240 >> fox.result &&
mpiexec -np 4 ./fox.o 240 >> fox.result &&
mpiexec -np 4 ./fox.o 240 >> fox.result &&
mpiexec -np 4 ./fox.o 240 >> fox.result &&
mpiexec -np 4 ./fox.o 240 >> fox.result &&
mpiexec -np 4 ./fox.o 480 >> fox.result &&
mpiexec -np 4 ./fox.o 480 >> fox.result &&
mpiexec -np 4 ./fox.o 480 >> fox.result &&
mpiexec -np 4 ./fox.o 480 >> fox.result &&
mpiexec -np 4 ./fox.o 480 >> fox.result
