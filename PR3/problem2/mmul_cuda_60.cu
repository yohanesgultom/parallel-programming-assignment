// author: yohanes.gultom@gmail.com
// partial source: https://gist.github.com/wh5a/4313739
#include <stdio.h>
#include <time.h>

#define TILE_WIDTH 60

// create random matrix row-major-format
float* create_flat_matrix_rand(int row, int col, int max)
{
    float* m = (float*)malloc(row*col*sizeof(float));
    int i, j = 0;
    for (i = 0; i < row; i++) {
        for (j = 0; j < col; j++) {
            float val = (max > 0) ? (float)(rand() % max) : 0.0f;
            m[col * i + j] = val;
        }
    }
    return m;
}

float* create_flat_matrix(int row, int col, float val)
{
    float* m = (float*)malloc(row*col*sizeof(float));
    int i, j = 0;
    for (i = 0; i < row; i++) {
        for (j = 0; j < col; j++) {
            m[col * i + j] = val;
        }
    }
    return m;
}


// print matrix row-major-format
void print_flat_matrix(float *m, int row, int col)
{
    int i, j = 0;
    for (i = 0; i < row; i++) {
        for (j = 0; j < col; j++) {
            printf("%.2f ", m[col * i + j]);
        }
        printf("\n");
    }
}

__global__ void mmul_d(float *first, int m, int p, float *second, int q, float *multiply)
{
    int c, d, k = 0;
    float sum = .0f;
    for (c = 0; c < m; c++) {
        for (d = 0; d < q; d++) {
            for (k = 0; k < p; k++) {
                sum = sum + first[c*m+k] * second[k*q+d];
            }
            multiply[c*q+d] = sum;
            sum = 0;
        }
    }
}

__global__ void mmul_d_thread(float *first, int m, int p, float *second, int q, float *multiply)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int d, k = 0;
    float sum = .0f;
    for (d = 0; d < q; d++) {
        for (k = 0; k < p; k++) {
            sum = sum + first[c*m+k] * second[k*q+d];
        }
        multiply[c*q+d] = sum;
        sum = 0;
    }
}

// Compute C = A * B
__global__ void matrixMultiply(float * A, float * B, float * C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns) {
    //@@ Insert code to implement matrix multiplication here
    __shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x, by = blockIdx.y,
       tx = threadIdx.x, ty = threadIdx.y,
       Row = by * TILE_WIDTH + ty,
       Col = bx * TILE_WIDTH + tx;
    float Pvalue = 0;

    for (int m = 0; m < (numAColumns-1)/TILE_WIDTH+1; ++m) {
       if (Row < numARows && m*TILE_WIDTH+tx < numAColumns)
          ds_M[ty][tx] = A[Row*numAColumns + m*TILE_WIDTH+tx];
       else
          ds_M[ty][tx] = 0;
       if (Col < numBColumns && m*TILE_WIDTH+ty < numBRows)
          ds_N[ty][tx] = B[(m*TILE_WIDTH+ty)*numBColumns+Col];
       else
          ds_N[ty][tx] = 0;

       __syncthreads();
       for (int k = 0; k < TILE_WIDTH; ++k)
          Pvalue += ds_M[ty][k] * ds_N[k][tx];
       __syncthreads();
    }
    if (Row < numCRows && Col < numCColumns)
       C[Row*numCColumns+Col] = Pvalue;
}

int main(int argc, char** argv)
{
    if (argc < 6) {
        printf("insufficient args. for A x B = C, required args: [row num A] [col num A/row num B] [col num B] [cuda block size] [reps] [optimized] [compare]\n");
        return EXIT_FAILURE;
    }

    int m, n, p, q = 0;
    m = atoi(argv[1]);
    n = atoi(argv[2]);
    p = n;
    q = atoi(argv[3]);
    int blockSize = atoi(argv[4]);
    int nBlocks = (blockSize > 0) ? (m * n) / blockSize + ((m * n) % blockSize == 0 ? 0 : 1) : 0;
    int reps = atoi(argv[5]);
    // optimized = ignore blockSize and nBlocks
    int optimized = (argc >= 7) ? atoi(argv[6]):0;
    int compare  = (argc >= 8) ? atoi(argv[7]):0;
    //@@ Initialize the optimized grid and block dimensions here
    dim3 dimGrid((q-1)/TILE_WIDTH+1, (m-1)/TILE_WIDTH+1, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    float *first_d, *second_d, *multiply_d;
    float *first, *second, *multiply;

    int i = 0;
    double total_time = 0.0f;
    for (i = 0; i < reps; i++) {
        double exec_time = ((double) clock()) * -1;
        first = create_flat_matrix(m, n, 1);
        second = create_flat_matrix(p, q, 2);
        multiply = create_flat_matrix(m, q, 0);

        cudaMalloc((void **) &first_d, m * n * sizeof(float));
        cudaMalloc((void **) &second_d, p * q * sizeof(float));
        cudaMalloc((void **) &multiply_d, m * q * sizeof(float));

        cudaMemcpy(first_d, first, m * n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(second_d, second, p * q * sizeof(float), cudaMemcpyHostToDevice);

        if (optimized == 1) {
            matrixMultiply<<<dimGrid, dimBlock>>>(first_d, second_d, multiply_d, m, n, p, q, m, q);
        } else {
            mmul_d_thread <<< nBlocks, blockSize >>> (first_d, m, n, second_d, q, multiply_d);
        }

        cudaMemcpy(multiply, multiply_d, m * q * sizeof(float), cudaMemcpyDeviceToHost);

        if (compare == 1) {
            printf("first:\n");
            print_flat_matrix(first, m, n);
            printf("second:\n");
            print_flat_matrix(second, p, q);
            printf("multiply:\n");
            print_flat_matrix(multiply, m, q);
        }

        free(multiply); free(second); free(first);
        cudaFree(first_d); cudaFree(second_d); cudaFree(multiply_d);
        total_time = total_time + ((exec_time + ((double)clock())) / CLOCKS_PER_SEC);
        // printf("%d: %.6f\n", i, ((exec_time + ((double)clock())) / CLOCKS_PER_SEC));
    }
    printf("%d\t%d\t%d\t%d\t%d\t%.6f\n", m, n, q, blockSize, reps, (total_time / reps));
    return EXIT_SUCCESS;
}
