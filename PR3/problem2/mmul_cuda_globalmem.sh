nvcc -o mmul_cuda.o mmul_cuda.cu &&
echo "A_row\tA_col/v_row\tv_col\tblocksize\treps\tT(s)" > mmul_cuda_globalmem.result &&
./mmul_cuda.o 10000 10000 1 0 2 0 >> mmul_cuda_globalmem.result &&
./mmul_cuda.o 15000 15000 1 0 2 0 >> mmul_cuda_globalmem.result &&
./mmul_cuda.o 20000 20000 1 0 2 0 >> mmul_cuda_globalmem.result &&
./mmul_cuda.o 25000 25000 1 0 2 0 >> mmul_cuda_globalmem.result &&
./mmul_cuda.o 30000 30000 1 0 2 0 >> mmul_cuda_globalmem.result &&
./mmul_cuda.o 1000 1000 1000 0 2 0 >> mmul_cuda_globalmem.result &&
./mmul_cuda.o 2000 2000 2000 0 2 0 >> mmul_cuda_globalmem.result &&
./mmul_cuda.o 3000 3000 3000 0 2 0 >> mmul_cuda_globalmem.result &&
./mmul_cuda.o 4000 4000 4000 0 2 0 >> mmul_cuda_globalmem.result &&
./mmul_cuda.o 5000 5000 5000 0 2 0 >> mmul_cuda_globalmem.result
