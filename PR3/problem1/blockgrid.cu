/**
 * By yohanes.gultom@gmail.com
 * Observing GPU block and grid behavior by playing with 1D (1 Dimension) block and grid
 */

#include <stdio.h>
#include <string.h>

__global__ void kernel1( int *a )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    a[idx] = 9;
}

__global__ void kernel2( int *a )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    a[idx] = blockIdx.x;
}

__global__ void kernel3( int *a )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    a[idx] = threadIdx.x;
}

void printarray(int *a, int n)
{
    int i = 0;
    for (i = 0; i < n; i++) printf("%d ", a[i]);
    printf("\n");
}

/**
 * main program
 */
int main(int argc, char** argv)
{
    if (argc < 3) {
        printf("insufficient args. usage: blockgrid [data array length] [threadsPerBlock] [numBlocks]");
        return EXIT_SUCCESS;
    }
    int N = atoi(argv[1]);
    int isize = N*sizeof(int);
    dim3 threadsPerBlock(atoi(argv[2]));
    dim3 numBlocks(atoi(argv[3]));

    // function kernel1
    printf("=== constant ===\n");
	int *a_h, *a_d;
    a_h = (int*) malloc(isize);
	cudaMalloc((void**)&a_d, isize);
    // printarray("before", a_h, N);
    kernel1<<<numBlocks, threadsPerBlock>>>(a_d);
	cudaMemcpy(a_h, a_d, isize, cudaMemcpyDeviceToHost);
    printarray(a_h, N);
    cudaFree(a_d);
    free(a_h);

    // function kernel2
    printf("=== blockIdx.x ===\n");
    int *b_h, *b_d;
    b_h = (int*) malloc(isize);
    cudaMalloc((void**)&b_d, isize);
    // printarray("before", b_h, N);
    kernel2<<<numBlocks, threadsPerBlock>>>(b_d);
    cudaMemcpy(b_h, b_d, isize, cudaMemcpyDeviceToHost);
    printarray(b_h, N);
    cudaFree(b_d);
    free(b_h);

    // function kernel3
    printf("=== threadIdx.x ===\n");
    int *c_h, *c_d;
    c_h = (int*) malloc(isize);
    cudaMalloc((void**)&c_d, isize);
    // printarray("before", c_h, N);
    kernel3<<<numBlocks, threadsPerBlock>>>(c_d);
    cudaMemcpy(c_h, c_d, isize, cudaMemcpyDeviceToHost);
    printarray(c_h, N);
    cudaFree(c_d);
    free(c_h);

	return EXIT_SUCCESS;
}
