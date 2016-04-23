// Multiply two matrices A * B = C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
// #include <matrixMul_kernel.cu>

// Thread block size
// #define BLOCK_SIZE 16
// #define TILE_SIZE  16
//
// #define WA 1024   // Matrix A width
// #define HA 1024   // Matrix A height
// #define WB 1024   // Matrix B width
// #define HB WA     // Matrix B height
// #define WC WB     // Matrix C width
// #define HC HA     // Matrix C height

// CUDA Kernel
__global__ void matrixMul( float* C, float* A, float* B, int wA, int wB)
{

    // 1. 2D Thread ID
    // int tx = blockIdx.x * TILE_SIZE + threadIdx.x;
    // int ty = blockIdx.y * TILE_SIZE + threadIdx.y;
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    // value stores the element that is
    // computed by the thread
    float value = 0;
    for (int i = 0; i < wA; ++i)
    {
        float elementA = A[ty * wA + i];
        float elementB = B[i * wB + tx];
        value += elementA * elementB;
    }

    // Write the matrix to device memory each
    // thread writes one element
    C[ty * wA + tx] = value;
}

// Allocates a matrix with random float entries.
void randomInit(float* data, int size)
{
    for (int i = 0; i < size; ++i)
    data[i] = rand() / (float)RAND_MAX;
}

void init(float* data, int size, float val)
{
    for (int i = 0; i < size; ++i)
    data[i] = val;
}


/////////////////////////////////////////////////////////
// Program main
/////////////////////////////////////////////////////////

int main(int argc, char** argv)
{
    float t0 = clock();

    int HA, WA, HB, WB, HC, WC;
    HA = atoi(argv[1]);
    WA = HA; HB = HA; WB = HA; HC = HA; WC = HA;
    int BLOCK_SIZE = atoi(argv[2]);
    int print = (argc >= 4) ? 1 : 0;

    // set seed for rand()
    srand(2006);

    // 1. allocate host memory for matrices A and B
    unsigned int size_A = WA * HA;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float* h_A = (float*) malloc(mem_size_A);

    unsigned int size_B = WB * HB;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float* h_B = (float*) malloc(mem_size_B);

    // 2. initialize host memory
    // randomInit(h_A, size_A);
    // randomInit(h_B, size_B);
    init(h_A, size_A, 1);
    init(h_B, size_B, 1);

    // 8. allocate device memory
    float* d_A;
    float* d_B;
    cudaMalloc((void**) &d_A, mem_size_A);
    cudaMalloc((void**) &d_B, mem_size_B);

    // 9. copy host memory to device
    cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);

    // 4. allocate host memory for the result C
    unsigned int size_C = WC * HC;
    unsigned int mem_size_C = sizeof(float) * size_C;
    float* h_C = (float*) malloc(mem_size_C);

    // 10. allocate device memory for the result
    float* d_C;
    cudaMalloc((void**) &d_C, mem_size_C);

    // 5. perform the calculation
    // setup execution parameters
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(WC / threads.x, HC / threads.y);

    // execute the kernel
    matrixMul<<< grid, threads >>>(d_C, d_A, d_B, WA, WB);

    // 11. copy result from device to host
    cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);

    // 6. print out the results

    // print out matrices
    if (print == 1) {
        printf("\n\nMatrix A\n");
        for(int i = 0; i < size_A; i++)
        {
            printf("%f ", h_A[i]);
            if(((i + 1) % WA) == 0)
            printf("\n");
        }

        printf("\n\nMatrix B\n");
        for(int i = 0; i < size_B; i++)
        {
            printf("%f ", h_B[i]);
            if(((i + 1) % WB) == 0)
            printf("\n");
        }

        printf("\n\nMatrix C (Results)\n");
        for(int i = 0; i < size_C; i++)
        {
            printf("%f ", h_C[i]);
            if(((i + 1) % WC) == 0)
            printf("\n");
        }
        printf("\n");
    }

    // 7. clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    t0 = ((float)(clock() - t0) / CLOCKS_PER_SEC);
    printf("%d\t%d\t%f\n", WA, BLOCK_SIZE, t0);

    return 0;
}
