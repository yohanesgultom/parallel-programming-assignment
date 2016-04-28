// Multiply two matrices A * B = C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

// #define WA 1024
// #define HA 1024
// #define WB 1024
// #define HB WA
// #define WC WB
// #define HC HA

#define MAX_SOURCE_SIZE (0x100000)

char *oclLoadProgSource(char *fileName, char *comment, size_t *source_size)
{
    /* Load the source code containing the kernel*/
    FILE *fp = fopen(fileName, "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    char *source_str = (char*)malloc(MAX_SOURCE_SIZE);
    *source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);
    return source_str;
}

void checkError(cl_int err, const char *operation)
{
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "Error during operation '%s': %d\n", operation, err);
        exit(1);
    }
}

// Allocates a matrix with random float entries.
void randomInit(float* data, int size)
{
    int i;
    for (i = 0; i < size; ++i)
    data[i] = rand() / (float)RAND_MAX;
}

void init(float* data, int size, float val)
{
    int i;
    for (i = 0; i < size; ++i)
    data[i] = val;
}


/////////////////////////////////////////////////////////
// Program main
/////////////////////////////////////////////////////////

int main(int argc, char** argv)
{
    float t0 = clock();

    int i, HA, WA, HB, WB, HC, WC;
    HA = atoi(argv[1]);
    WA = HA; HB = HA; WB = HA; HC = HA; WC = HA;
    int local_size = atoi(argv[2]);
    int print = (argc >= 4) ? 1 : 0;

    char *kernel_filename = "mmul_opencl.cl";
    char *kernel_comment = "// matrix multiplication\n";

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

    // 4. allocate host memory for the result C
    unsigned int size_C = WC * HC;
    unsigned int mem_size_C = sizeof(float) * size_C;
    float* h_C = (float*) malloc(mem_size_C);

    // 5. Initialize OpenCL
    // OpenCL specific variables
    cl_context clGPUContext;
    cl_command_queue clCommandQue;
    cl_program clProgram;
    cl_kernel clKernel;

    size_t dataBytes;
    size_t kernelLength;
    cl_int errcode;

    // OpenCL device memory for matrices
    cl_mem d_A;
    cl_mem d_B;
    cl_mem d_C;

    /*****************************************/
    /* Initialize OpenCL */
    /*****************************************/
    /* Get Platform and Device Info */

    cl_device_id device_id = NULL;
    cl_platform_id platform_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    errcode = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    checkError(errcode, "clGetPlatformIDs");

    errcode = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);
    checkError(errcode, "clGetDeviceIDs");

    /* Create OpenCL context */
    clGPUContext = clCreateContext(NULL, 1, &device_id, NULL, NULL, &errcode);
    checkError(errcode, "clCreateContext");

    // clGPUContext = clCreateContextFromType(0, CL_DEVICE_TYPE_GPU, NULL, NULL, &errcode);
    // shrCheckError(errcode, CL_SUCCESS);
    checkError(errcode, "clCreateContextFromType");

    // get the list of GPU devices associated
    // with context
    errcode = clGetContextInfo(clGPUContext, CL_CONTEXT_DEVICES, 0, NULL, &dataBytes);
    cl_device_id *clDevices = (cl_device_id *) malloc(dataBytes);
    errcode |= clGetContextInfo(clGPUContext, CL_CONTEXT_DEVICES, dataBytes, clDevices, NULL);
    // shrCheckError(errcode, CL_SUCCESS);

    //Create a command-queue
    clCommandQue = clCreateCommandQueue(clGPUContext, clDevices[0], 0, &errcode);
    // shrCheckError(errcode, CL_SUCCESS);

    // Setup device memory
    d_C = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, mem_size_A, NULL, &errcode);
    d_A = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_A, h_A, &errcode);
    d_B = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_B, h_B, &errcode);

    // 6. Load and build OpenCL kernel
    char *clMatrixMul = oclLoadProgSource(kernel_filename, kernel_comment, &kernelLength);
    // shrCheckError(clMatrixMul != NULL, CL_SUCCESS);

    clProgram = clCreateProgramWithSource(clGPUContext, 1, (const char **)&clMatrixMul, &kernelLength, &errcode);
    // shrCheckError(errcode, CL_SUCCESS);
    checkError(errcode, "clCreateProgramWithSource");

    errcode = clBuildProgram(clProgram, 0, NULL, NULL, NULL, NULL);
    // shrCheckError(errcode, CL_SUCCESS);
    checkError(errcode, "clBuildProgram");

    clKernel = clCreateKernel(clProgram, "matrixMul", &errcode);
    // shrCheckError(errcode, CL_SUCCESS);
    checkError(errcode, "clCreateKernel");

    // 7. Launch OpenCL kernel
    size_t localWorkSize[2], globalWorkSize[2];

    int wA = WA;
    int wC = WC;
    errcode = clSetKernelArg(clKernel, 0, sizeof(cl_mem), (void *)&d_C);
    errcode |= clSetKernelArg(clKernel, 1, sizeof(cl_mem), (void *)&d_A);
    errcode |= clSetKernelArg(clKernel, 2, sizeof(cl_mem), (void *)&d_B);
    errcode |= clSetKernelArg(clKernel, 3, sizeof(int), (void *)&wA);
    errcode |= clSetKernelArg(clKernel, 4, sizeof(int), (void *)&wC);
    // shrCheckError(errcode, CL_SUCCESS);

    // localWorkSize[0] = 16;
    // localWorkSize[1] = 16;
    // globalWorkSize[0] = 1024;
    // globalWorkSize[1] = 1024;
    localWorkSize[0] = local_size;
    localWorkSize[1] = local_size;
    globalWorkSize[0] = WA;
    globalWorkSize[1] = WA;

    errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    // shrCheckError(errcode, CL_SUCCESS);

    // 8. Retrieve result from device
    errcode = clEnqueueReadBuffer(clCommandQue, d_C, CL_TRUE, 0, mem_size_C, h_C, 0, NULL, NULL);
    // shrCheckError(errcode, CL_SUCCESS);

    // print out matrices
    if (print == 1) {
        printf("\n\nMatrix A\n");
        for(i = 0; i < size_A; i++)
        {
            printf("%f ", h_A[i]);
            if(((i + 1) % WA) == 0)
            printf("\n");
        }

        printf("\n\nMatrix B\n");
        for(i = 0; i < size_B; i++)
        {
            printf("%f ", h_B[i]);
            if(((i + 1) % WB) == 0)
            printf("\n");
        }

        printf("\n\nMatrix C (Results)\n");
        for(i = 0; i < size_C; i++)
        {
            printf("%f ", h_C[i]);
            if(((i + 1) % WC) == 0)
            printf("\n");
        }
        printf("\n");
    }


    // 10. clean up memory
    free(h_A);
    free(h_B);
    free(h_C);

    clReleaseMemObject(d_A);
    clReleaseMemObject(d_C);
    clReleaseMemObject(d_B);

    free(clDevices);
    free(clMatrixMul);
    clReleaseContext(clGPUContext);
    clReleaseKernel(clKernel);
    clReleaseProgram(clProgram);
    clReleaseCommandQueue(clCommandQue);

    t0 = ((float)(clock() - t0) / CLOCKS_PER_SEC);
    printf("%d\t%d\t%f\n", WA, local_size, t0);

    return 0;
}
