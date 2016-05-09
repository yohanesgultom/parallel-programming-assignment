#!/bin/sh

nvcc mmul_cuda.cu -o mmul_cuda.o
nvcc mmul_cuda_20.cu -o mmul_cuda_20.o
nvcc mmul_cuda_30.cu -o mmul_cuda_30.o
nvcc mmul_cuda_40.cu -o mmul_cuda_40.o
nvcc mmul_cuda_50.cu -o mmul_cuda_50.o
nvcc mmul_cuda_60.cu -o mmul_cuda_60.o
nvcc mmul_cuda_70.cu -o mmul_cuda_70.o
echo "A_row\tA_col/v_row\tv_col\tblocksize\treps\tT(s)" > mmul_cuda_blockvar.result
./mmul_cuda.o 5000 5000 5000 0 3 1 0 >> mmul_cuda_blockvar.result
./mmul_cuda_20.o 5000 5000 5000 0 3 1 0 >> mmul_cuda_blockvar.result
./mmul_cuda_30.o 5000 5000 5000 0 3 1 0 >> mmul_cuda_blockvar.result
./mmul_cuda_40.o 5000 5000 5000 0 3 1 0 >> mmul_cuda_blockvar.result
./mmul_cuda_50.o 5000 5000 5000 0 3 1 0 >> mmul_cuda_blockvar.result
./mmul_cuda_60.o 5000 5000 5000 0 3 1 0 >> mmul_cuda_blockvar.result
./mmul_cuda_70.o 5000 5000 5000 0 3 1 0 >> mmul_cuda_blockvar.result
