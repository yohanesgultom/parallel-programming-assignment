nvcc -o mmul_cublas.o -lcublas -lcurand mmul_cublas.cu &&
echo "A_row\tA_col/v_row\tv_col\treps\tT(s)" > mmul_cublas.result &&
./mmul_cublas.o 1000 1000 1000 2 >> mmul_cublas.result &&
./mmul_cublas.o 2000 2000 2000 2 >> mmul_cublas.result &&
./mmul_cublas.o 3000 3000 3000 2 >> mmul_cublas.result &&
./mmul_cublas.o 4000 4000 4000 2 >> mmul_cublas.result &&
./mmul_cublas.o 5000 5000 5000 2 >> mmul_cublas.result
