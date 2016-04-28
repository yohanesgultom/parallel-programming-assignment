nvcc -o mmul_cublas.o -lcublas -lcurand mmul_cublas.cu &&
echo "A_row\tA_col/v_row\tv_col\treps\tT(s)" > mmul_cublas.result &&
./mmul_cublas.o 1000 1000 1 10 >> mmul_cublas.result &&
./mmul_cublas.o 2000 2000 1 10 >> mmul_cublas.result &&
./mmul_cublas.o 3000 3000 1 10 >> mmul_cublas.result &&
./mmul_cublas.o 4000 4000 1 10 >> mmul_cublas.result &&
./mmul_cublas.o 5000 5000 1 10 >> mmul_cublas.result &&
./mmul_cublas.o 6000 6000 1 10 >> mmul_cublas.result &&
./mmul_cublas.o 7000 7000 1 10 >> mmul_cublas.result &&
./mmul_cublas.o 8000 8000 1 10 >> mmul_cublas.result &&
./mmul_cublas.o 9000 9000 1 10 >> mmul_cublas.result &&
./mmul_cublas.o 10000 10000 1 10 >> mmul_cublas.result
