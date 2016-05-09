#!/bin/bash

cuda_name="mmul_cuda"
clear &&
nvcc "$cuda_name.cu" -o "$cuda_name.o" &&
echo "Data\tBlock Size\tTime" > "$cuda_name.result" &&
./"$cuda_name.o" 256 32 >> "$cuda_name.result" &&
./"$cuda_name.o" 512 32 >> "$cuda_name.result" &&
./"$cuda_name.o" 1024 32 >> "$cuda_name.result" &&
./"$cuda_name.o" 2048 32 >> "$cuda_name.result" &&
./"$cuda_name.o" 4096 32 >> "$cuda_name.result"

opencl_name="mmul_opencl"
gcc "$opencl_name.c" -o "$opencl_name.o" -l OpenCL &&
clear &&
echo "Data\tBlock Size\tTime" > "$opencl_name.result" &&
./"$opencl_name.o" 256 32 >> "$opencl_name.result" &&
./"$opencl_name.o" 512 32 >> "$opencl_name.result" &&
./"$opencl_name.o" 1024 32 >> "$opencl_name.result" &&
./"$opencl_name.o" 2048 32 >> "$opencl_name.result" &&
./"$opencl_name.o" 4096 32 >> "$opencl_name.result"

echo "Done."
