#include <stdio.h>
#include <stdlib.h>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#define VECTOR_SIZE 1024

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

int main(void) {
    int i;
    char *kernel_filename = "saxpy.cl";
    char *kernel_comment = "// saxpy";
    size_t kernelLength;

    // Allocate space for vectors A, B and C
    float alpha = 2.0;
    float *A = (float*)malloc(sizeof(float)*VECTOR_SIZE);
    float *B = (float*)malloc(sizeof(float)*VECTOR_SIZE);
    float *C = (float*)malloc(sizeof(float)*VECTOR_SIZE);
    for(i = 0; i < VECTOR_SIZE; i++)
    {
        A[i] = i;
        B[i] = VECTOR_SIZE - i;
        C[i] = 0;
    }

    // Get platform and device information
    cl_platform_id * platforms = NULL;
    cl_uint num_platforms;
    cl_device_id *device_list = NULL;
    cl_uint num_devices;
    cl_context context;
    char *kernel_content = NULL;

    //Set up the Platform
    cl_int clStatus = clGetPlatformIDs(0, NULL, &num_platforms);
    platforms = (cl_platform_id *) malloc(sizeof(cl_platform_id)*num_platforms);
    clStatus = clGetPlatformIDs(num_platforms, platforms, NULL);

    //Get the devices list and choose the device you want to run on
    clStatus = clGetDeviceIDs( platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
    device_list = (cl_device_id *) malloc(sizeof(cl_device_id)*num_devices);
    clStatus = clGetDeviceIDs( platforms[0], CL_DEVICE_TYPE_GPU, num_devices, device_list, NULL);

    // Create one OpenCL context for each device in the platform
    context = clCreateContext( NULL, num_devices, device_list, NULL, NULL, &clStatus);

    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_list[0], 0, &clStatus);

    // Create memory buffers on the device for each vector
    cl_mem A_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY, VECTOR_SIZE * sizeof(float), NULL, &clStatus);
    cl_mem B_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY, VECTOR_SIZE * sizeof(float), NULL, &clStatus);
    cl_mem C_clmem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, VECTOR_SIZE * sizeof(float), NULL, &clStatus);

    // Copy the Buffer A and B to the device
    clStatus = clEnqueueWriteBuffer(command_queue, A_clmem, CL_TRUE, 0, VECTOR_SIZE * sizeof(float), A, 0, NULL, NULL);
    clStatus = clEnqueueWriteBuffer(command_queue, B_clmem, CL_TRUE, 0, VECTOR_SIZE * sizeof(float), B, 0, NULL, NULL);

    // Create a program from the kernel source
    kernel_content = oclLoadProgSource(kernel_filename, kernel_comment, &kernelLength);
    // printf("%s\n", kernel_content);
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&kernel_content, NULL, &clStatus);

    // Build the program
    clStatus = clBuildProgram(program, 1, device_list, NULL, NULL, NULL);

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "saxpy_kernel", &clStatus);

    // Set the arguments of the kernel
    clStatus = clSetKernelArg(kernel, 0, sizeof(float), (void *)&alpha);
    clStatus = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&A_clmem);
    clStatus = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&B_clmem);
    clStatus = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&C_clmem);

    // Execute the OpenCL kernel on the list
    size_t global_size = VECTOR_SIZE; // Process the entire lists
    size_t local_size = 64;

    // Process one item at a time
    clStatus = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);

    // Read the cl memory C_clmem on device to the host variable C
    clStatus = clEnqueueReadBuffer(command_queue, C_clmem, CL_TRUE, 0, VECTOR_SIZE * sizeof(float), C, 0, NULL, NULL);

    // Clean up and wait for all the comands to complete.
    clStatus = clFlush(command_queue);
    clStatus = clFinish(command_queue);

    // Display the result to the screen
    for(i = 0; i < VECTOR_SIZE; i++) printf("%f * %f + %f = %f\n", alpha, A[i], B[i], C[i]);

    // Finally release all OpenCL allocated objects and host buffers.
    clStatus = clReleaseKernel(kernel);
    clStatus = clReleaseProgram(program);
    clStatus = clReleaseMemObject(A_clmem);
    clStatus = clReleaseMemObject(B_clmem);
    clStatus = clReleaseMemObject(C_clmem);
    clStatus = clReleaseCommandQueue(command_queue);
    clStatus = clReleaseContext(context);
    free(A);
    free(B);
    free(C);
    free(platforms);
    free(device_list);
    return 0;
}
