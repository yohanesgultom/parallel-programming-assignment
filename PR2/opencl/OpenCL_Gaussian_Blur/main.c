#include <gaussian.h>
#include <args.h>
#include <stdbool.h>
#include <stdio.h>
#include <time.h>


int main ( int argc, char *argv[] )
{
    double units;
    uint32_t gaussianSize;
    float gaussianSigma, cpu_t, gpu_t;
    char* imgname;//the name of the image file, taken by the arguments
    time_t start;

    //read in the program's arguments
    if(readArguments(argc,argv,&imgname,&gaussianSize,&gaussianSigma)==false)
        return -1;

    start = clock();

    //perform CPU blurring
    if(pna_blur_cpu(imgname,gaussianSize,gaussianSigma)==false)//time it
        return -2;

    cpu_t = (clock() - start) / (float) CLOCKS_PER_SEC;

    start = clock();

    //perform GPU blurring and then read the timer
    if(pna_blur_gpu(imgname,gaussianSize,gaussianSigma)==false)
        return -3;

    gpu_t = (clock() - start) / (float) CLOCKS_PER_SEC;
    printf("CPU Gaussian blurring took: %f s\n", cpu_t);
    printf("GPU Gaussian blurring took: %f s\n", gpu_t);

    return 0;
}
