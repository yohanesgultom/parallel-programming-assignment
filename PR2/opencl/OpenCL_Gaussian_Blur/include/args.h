#ifndef READ_ARGS_H
#define READ_ARGS_H

#ifndef _MSC_VER
    #define __STDC_FORMAT_MACROS //also request the printf format macros
    #include <inttypes.h>
#else//msvc does not have the C99 standard header so we gotta define them explicitly here, since they do have some similar types
    typedef unsigned __int8 uint8_t;
    typedef __int8  int8_t;
    typedef unsigned __int16 uint16_t;
    typedef __int16 int16_t;
    typedef unsigned __int32 uint32_t;
    typedef __int32 int32_t;
    typedef unsigned __int64 uint64_t;
    typedef __int64 int64_t;
#endif

extern char helpStr[];

//default argument values
#define DEFAULT_IMG_NAME        "image3.BMP"
#define DEFAULT_GAUSSIAN_SIZE   3
#define DEFAULT_GAUSSIAN_SIGMA  0.8

//! @brief Reads in the program's arguments
//!
//! @param argc The number of arguments
//! @param argV The arguments buffer
//! @param imgName The name of the image file argument
//! @param gaussianSize The size of the gaussian kernel argument
//! @param gaussianSigma The gaussian sigma parameter argument
//! @return Returns true for succesfull argument parsing and false if there was a failure
char readArguments(int argc, char *argv[],
                   char** imgName, uint32_t* gaussianSize, 
                   float* gaussianSigma);

#endif
