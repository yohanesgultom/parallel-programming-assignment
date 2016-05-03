// device_query.c
// yohanes.gultom@gmail.com
// Original source:
// * http://stackoverflow.com/questions/17240071/what-is-the-right-way-to-call-clgetplatforminfo
// * Banger, R, Bhattacharyya .K. "OpenCL Programming by Example". 2013. Packt publishing​. p43

#include <stdio.h>
#include <stdlib.h>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#define NELEMS(x)  (sizeof(x) / sizeof((x)[0]))

const cl_platform_info attributeTypes[5] = {
    CL_PLATFORM_NAME,
    CL_PLATFORM_VENDOR,
    CL_PLATFORM_VERSION,
    CL_PLATFORM_PROFILE,
    CL_PLATFORM_EXTENSIONS
};

const char* const attributeNames[] = {
    "CL_PLATFORM_NAME",
    "CL_PLATFORM_VENDOR",
    "CL_PLATFORM_VERSION",
    "CL_PLATFORM_PROFILE",
    "CL_PLATFORM_EXTENSIONS"
};

void PrintDeviceInfo(cl_device_id device)
{
    char queryBuffer[1024];
    int queryInt;
    cl_int clError;
    clError = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(queryBuffer), &queryBuffer, NULL);
    printf("    CL_DEVICE_NAME: %s\n", queryBuffer);
    queryBuffer[0] = '\0';
    clError = clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(queryBuffer), &queryBuffer, NULL);
    printf("    CL_DEVICE_VENDOR: %s\n", queryBuffer);
    queryBuffer[0] = '\0';
    clError = clGetDeviceInfo(device, CL_DRIVER_VERSION, sizeof(queryBuffer), &queryBuffer, NULL);
    printf("    CL_DRIVER_VERSION: %s\n", queryBuffer);
    queryBuffer[0] = '\0';
    clError = clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(queryBuffer), &queryBuffer, NULL);
    printf("    CL_DEVICE_VERSION: %s\n", queryBuffer);
    queryBuffer[0] = '\0';
    clError = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(int), &queryInt, NULL);
    printf("    CL_DEVICE_MAX_COMPUTE_UNITS: %d\n", queryInt);
}

int main(void) {
    int i, j, k, num_attributes;
    char* info;

    cl_platform_id * platforms = NULL;
    cl_uint num_platforms;
    cl_device_id *device_list = NULL;
    cl_uint num_devices;
    cl_int clStatus;
    size_t infoSize;

    // Get platform and device information
    clStatus = clGetPlatformIDs(0, NULL, &num_platforms);
    platforms = (cl_platform_id *) malloc(sizeof(cl_platform_id) * num_platforms);
    clStatus = clGetPlatformIDs(num_platforms, platforms, NULL);

    // for each platform print all attributes
    num_attributes = NELEMS(attributeTypes);
    // printf("\nAttribute Count = %d ", num_attributes);
    for (i = 0; i < num_platforms; i++) {
        printf("Platform - %d\n", i+1);
        for (j = 0; j < num_attributes; j++) {
            // get platform attribute value size
            clGetPlatformInfo(platforms[i], attributeTypes[j], 0, NULL, &infoSize);
            info = (char*) malloc(infoSize);
            // get platform attribute value
            clGetPlatformInfo(platforms[i], attributeTypes[j], infoSize, info, NULL);
            printf("  %d.%d %-11s: %s\n", i+1, j+1, attributeNames[j], info);
        }
        //Get the devices list and choose the device you want to run on
        clStatus = clGetDeviceIDs( platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
        device_list = (cl_device_id *) malloc(sizeof(cl_device_id)*num_devices);
        clStatus = clGetDeviceIDs( platforms[i], CL_DEVICE_TYPE_GPU, num_devices, device_list, NULL);
        for (k = 0; k < num_devices; k++) {
            printf("  Device - %d:\n", (k+1));
            PrintDeviceInfo(device_list[k]);
        }
    }


    free(platforms);
    // free(device_list);
    return 0;
}
