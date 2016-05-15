#include <args.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>

char helpStr[] = {
    "Usage:\n\t-h\t Displays this help message\n\t"
    "-i\tThe name of the input image for blurring\n\t"
    "-g\tThe size of the gaussian kernel. Default is 3\n\t"
    "-s\tThe sigma parameter of the gaussian. Default is 0.8\n"
};

char readArguments(int argc,
                   char *argv[],
                   char** imgname,
                   uint32_t* gaussianSize,
                   float* gaussianSigma)
{
    uint32_t i;
    char gSizeYes, gSigmaYes;
    gSizeYes = gSigmaYes = false;
    *imgname=0;
    //read the arguments of the program
    for (i=1 ;i<argc ;i++) {
        if (strcmp(argv[i],"-h") == 0) {
            printf("\n%s\n",helpStr);
            return false;
        }
        //read the image name
        else if(strcmp(argv[i],"-i")==0)
        {
            *imgname = argv[i+1];
            i++;
        }
        //read the gaussian size
        else if(strcmp(argv[i],"-g")==0)
        {
            *gaussianSize = atoi(argv[i+1]);
            i++;
            //only accept odd sized gaussian
            if((*gaussianSize)%2 == 0 && (*gaussianSize) > 1)
            {
                printf("Argument Error: The size of the squared gaussian kernel has to be an odd number greater than one.\n");
                return false;
            }
            gSizeYes = true;
        }
        //read the gaussian sigma parameter
        else if(strcmp(argv[i],"-s")==0)
        {
            *gaussianSigma = atof(argv[i+1]);
            i++;
            gSigmaYes = true;
        }
        else
        {
            printf("Unrecognized program argument given. The correct program usage can be seen below:\n%s",helpStr);
            return false;
        }
    }
    //if arguments are not given take default value
    if(*imgname == 0)
        *imgname = DEFAULT_IMG_NAME;
    if(gSizeYes == false)
        *gaussianSize = DEFAULT_GAUSSIAN_SIZE;
    if(gSigmaYes == false)
        *gaussianSigma = DEFAULT_GAUSSIAN_SIGMA;
    //success
    return true;
}
