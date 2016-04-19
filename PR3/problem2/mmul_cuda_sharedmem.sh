nvcc -o mmul_cuda.o mmul_cuda.cu &&
echo "A_row\tA_col/v_row\tv_col\tblocksize\treps\tT(s)" > mmul_cuda_sharedmem.result &&
./mmul_cuda.o 100 100 1 10 10 Y >> mmul_cuda_sharedmem.result &&
./mmul_cuda.o 200 200 1 10 10 Y >> mmul_cuda_sharedmem.result &&
./mmul_cuda.o 300 300 1 10 10 Y >> mmul_cuda_sharedmem.result &&
./mmul_cuda.o 400 400 1 10 10 Y >> mmul_cuda_sharedmem.result &&
./mmul_cuda.o 500 500 1 10 10 Y >> mmul_cuda_sharedmem.result &&
./mmul_cuda.o 600 600 1 10 10 Y >> mmul_cuda_sharedmem.result &&
./mmul_cuda.o 700 700 1 10 10 Y >> mmul_cuda_sharedmem.result &&
./mmul_cuda.o 800 800 1 10 10 Y >> mmul_cuda_sharedmem.result &&
./mmul_cuda.o 900 900 1 10 10 Y >> mmul_cuda_sharedmem.result &&
./mmul_cuda.o 1000 1000 1 10 10 Y >> mmul_cuda_sharedmem.result &&
./mmul_cuda.o 100 100 100 10 10 Y >> mmul_cuda_sharedmem.result &&
./mmul_cuda.o 200 200 200 10 10 Y >> mmul_cuda_sharedmem.result &&
./mmul_cuda.o 300 300 300 10 10 Y >> mmul_cuda_sharedmem.result &&
./mmul_cuda.o 400 400 400 10 10 Y >> mmul_cuda_sharedmem.result &&
./mmul_cuda.o 500 500 500 10 10 Y >> mmul_cuda_sharedmem.result &&
./mmul_cuda.o 600 600 600 10 10 Y >> mmul_cuda_sharedmem.result &&
./mmul_cuda.o 700 700 700 10 10 Y >> mmul_cuda_sharedmem.result &&
./mmul_cuda.o 800 800 800 10 10 Y >> mmul_cuda_sharedmem.result &&
./mmul_cuda.o 900 900 900 10 10 Y >> mmul_cuda_sharedmem.result &&
./mmul_cuda.o 1000 1000 1000 10 10 Y >> mmul_cuda_sharedmem.result
