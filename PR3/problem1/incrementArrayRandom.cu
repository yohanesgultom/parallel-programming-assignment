// incrementArray.cu

#include <stdio.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

void incrementArrayOnHost(float *a, int N)
{
    int i;
    for (i=0; i < N; i++) a[i] = a[i]+1.f;
}

__global__ void incrementArrayOnDevice(float *a, int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx<N) a[idx] = a[idx]+1.f;
}

void printarray(float *a, int n)
{
    int i = 0;
    for (i = 0; i < n; i++) printf("%f ", a[i]);
    printf("\n");
}

// http://www.concentric.net/~Ttwang/tech/inthash.htm
unsigned long mix(unsigned long a, unsigned long b, unsigned long c)
{
    a=a-b;  a=a-c;  a=a^(c >> 13);
    b=b-c;  b=b-a;  b=b^(a << 8);
    c=c-a;  c=c-b;  c=c^(b >> 13);
    a=a-b;  a=a-c;  a=a^(c >> 12);
    b=b-c;  b=b-a;  b=b^(a << 16);
    c=c-a;  c=c-b;  c=c^(b >> 5);
    a=a-b;  a=a-c;  a=a^(c >> 3);
    b=b-c;  b=b-a;  b=b^(a << 10);
    c=c-a;  c=c-b;  c=c^(b >> 15);
    return c;
}

int main(int argc, char** argv)
{
    // program args
    if (argc < 2) {
        printf("usage: incrementArrayRandom [max_size] [repetitions]\n");
        return EXIT_SUCCESS;
    }

    int max_size = atoi(argv[1]);
    int repetitions = atoi(argv[2]);

    // randomize within same run
    srand(mix(clock(), time(NULL), getpid()));

    float *a_h, *b_h; // pointers to host memory
    float *a_d; // pointer to device memory
    int i, epoch = 0;
    int N = 0;

    int total_success = 0;
    for (epoch = 0; epoch < repetitions; epoch++) {
        N = rand() % max_size;
        size_t size = N*sizeof(float);

        // allocate arrays on host
        a_h = (float *)malloc(size);
        b_h = (float *)malloc(size);
        // allocate array on device
        cudaMalloc((void **) &a_d, size);
        // initialization of host data
        for (i=0; i<N; i++) a_h[i] = (float)i;
        // copy data from host to device
        cudaMemcpy(a_d, a_h, sizeof(float)*N, cudaMemcpyHostToDevice);
        // do calculation on host
        incrementArrayOnHost(a_h, N);
        // printarray(a_h, N);
        // do calculation on device:
        // Part 1 of 2. Compute execution configuration
        int blockSize = 4;
        int nBlocks = N/blockSize + (N%blockSize == 0?0:1);
        // Part 2 of 2. Call incrementArrayOnDevice kernel
        incrementArrayOnDevice <<< nBlocks, blockSize >>> (a_d, N);
        // Retrieve result from device and store in b_h
        cudaMemcpy(b_h, a_d, sizeof(float)*N, cudaMemcpyDeviceToHost);
        // check results
        // printarray(b_h, N);
        int success = 1;
        for (i=0; i<N; i++) {
            if (a_h[i] != b_h[i]) {
                success = 0;
                break;
            }
        }
        printf("epoch %d a[%d] = %s\n", epoch, N, (success == 1) ? "true" : "false");
        if (success == 1) total_success += 1;
    }
    printf("\nsuccess rate: %f%%\n", total_success / ((float)repetitions) * 100.0);
    // cleanup
    free(a_h); free(b_h); cudaFree(a_d);
    return EXIT_SUCCESS;
}
