nvcc -o mmul_cuda.o mmul_cuda.cu &&
echo "A_row\tA_col/v_row\tv_col\tblocksize\treps\tT(s)" > mmul_cuda.result &&
./mmul_cuda.o 1000 1000 1 100 10 >> mmul_cuda.result &&
./mmul_cuda.o 2000 2000 1 100 10 >> mmul_cuda.result &&
./mmul_cuda.o 3000 3000 1 100 10 >> mmul_cuda.result &&
./mmul_cuda.o 4000 4000 1 100 10 >> mmul_cuda.result &&
./mmul_cuda.o 5000 5000 1 100 10 >> mmul_cuda.result &&
./mmul_cuda.o 6000 6000 1 100 10 >> mmul_cuda.result &&
./mmul_cuda.o 7000 7000 1 100 10 >> mmul_cuda.result &&
./mmul_cuda.o 8000 8000 1 100 10 >> mmul_cuda.result &&
./mmul_cuda.o 9000 9000 1 100 10 >> mmul_cuda.result &&
./mmul_cuda.o 10000 10000 1 100 10 >> mmul_cuda.result
