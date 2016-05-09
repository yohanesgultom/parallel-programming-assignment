// author: yohanes.gultom@gmail.com
// Source adapted from: https://raw.githubusercontent.com/sol-prog/cuda_cublas_curand_thrust/master/mmul_1.cu
// Low level matrix multiplication on GPU using CUDA with CURAND and CUBLAS
// C(m,n) = A(m,k) * B(k,n)
#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <cublas_v2.h>
#include <curand.h>
#include <time.h>

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

// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on GPU
void GPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A) {
	// Create a pseudo-random number generator
	curandGenerator_t prng;
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

	// Set the seed for the random number generator using the system clock
	curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

	// Fill the array with random numbers on the device
	curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
}

// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n)
void gpu_blas_mmul(const float *A, const float *B, float *C, const int m, const int k, const int n) {
	int lda=m,ldb=k,ldc=m;
	const float alf = 1;
	const float bet = 0;
	const float *alpha = &alf;
	const float *beta = &bet;

	// Create a handle for CUBLAS
	cublasHandle_t handle;
	cublasCreate(&handle);

	// Do the actual multiplication
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

	// Destroy the handle
	cublasDestroy(handle);
}


//Print matrix A(nr_rows_A, nr_cols_A) storage in column-major format
void print_matrix(const float *A, int nr_rows_A, int nr_cols_A) {

    for(int i = 0; i < nr_rows_A; ++i){
        for(int j = 0; j < nr_cols_A; ++j){
            printf("%.2f ", A[j * nr_rows_A + i]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char** argv) {
	if (argc < 5) {
		printf("unsufficient arguments\n");
		return EXIT_FAILURE;
	}


	// Allocate 3 arrays on CPU
	int nr_rows_A = atoi(argv[1]);
	int nr_cols_A = atoi(argv[2]);
	int nr_rows_B = nr_cols_A;
	int nr_cols_B = atoi(argv[3]);
	int nr_rows_C = nr_cols_A;
	int nr_cols_C = nr_rows_B;
	int reps = atoi(argv[4]);
    int print = (argc >= 6) ? atoi(argv[5]) : 0;

	// // for simplicity we are going to use square arrays
	// nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = 3;

	float *h_A, *h_B, *h_C;
	float *d_A, *d_B, *d_C;
	double total_time = 0.0;
	int i = 0;
	for (i = 0; i < reps; i++) {
		double exec_time = ((double) clock()) * -1;

		h_A = create_flat_matrix(nr_rows_A, nr_cols_A, 1);
		h_B = create_flat_matrix(nr_rows_B, nr_cols_B, 2);
		h_C = create_flat_matrix(nr_rows_C, nr_cols_C, 0);
		// Allocate 3 arrays on GPU
		cudaMalloc(&d_A,nr_rows_A * nr_cols_A * sizeof(float));
		cudaMalloc(&d_B,nr_rows_B * nr_cols_B * sizeof(float));
		cudaMalloc(&d_C,nr_rows_C * nr_cols_C * sizeof(float));
		// If you already have useful values in A and B you can copy them in GPU:
		cudaMemcpy(d_A,h_A,nr_rows_A * nr_cols_A * sizeof(float),cudaMemcpyHostToDevice);
		cudaMemcpy(d_B,h_B,nr_rows_B * nr_cols_B * sizeof(float),cudaMemcpyHostToDevice);
		// Optionally we can copy the data back on CPU and print the arrays
		cudaMemcpy(h_A,d_A,nr_rows_A * nr_cols_A * sizeof(float),cudaMemcpyDeviceToHost);
		cudaMemcpy(h_B,d_B,nr_rows_B * nr_cols_B * sizeof(float),cudaMemcpyDeviceToHost);
		// Multiply A and B on GPU
		gpu_blas_mmul(d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);
		// Copy (and print) the result on host memory
		cudaMemcpy(h_C,d_C,nr_rows_C * nr_cols_C * sizeof(float),cudaMemcpyDeviceToHost);
        if (print == 1) {
            printf("A:\n");
    		print_matrix(h_A, nr_rows_A, nr_cols_A);
    		printf("B:\n");
    		print_matrix(h_B, nr_rows_B, nr_cols_B);
    		printf("C:\n");
    		print_matrix(h_C, nr_rows_C, nr_cols_C);
        }
		//Free GPU memory
		cudaFree(d_A);
		cudaFree(d_B);
		cudaFree(d_C);
		// Free CPU memory
		free(h_A);
		free(h_B);
		free(h_C);

		total_time = total_time + (exec_time + ((double)clock())) / CLOCKS_PER_SEC;
		// printf("%d: %.6f\n", i, ((exec_time + ((double)clock())) / CLOCKS_PER_SEC));
	}
	printf("%d\t%d\t%d\t%d\t%.6f\n", nr_rows_A, nr_cols_A, nr_cols_B, reps, (total_time/reps));
	return EXIT_SUCCESS;
}
