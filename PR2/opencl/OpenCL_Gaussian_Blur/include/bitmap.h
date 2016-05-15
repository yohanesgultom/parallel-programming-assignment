#ifndef BITMAP_H
#define BITMAP_H

/**Bitmap (BMP) File related structs**/
//make sure that the structure's member are titghly packed
#pragma pack(push,1)
//! The bitmap file header, stores general information about the BMP file
typedef struct BitmapFileHeader
{
    //! The signature of the BitmapFile should be 0x42 0x4D
    unsigned char magicNumber[2];
    //! The size of the file in bytes
    unsigned int fileSize;
    //! Depends on the application that creates the image
    unsigned short reserved1;
    //! Depends on the application that creates the image
    unsigned short reserved2;
    //! The offset to the data of the image
    unsigned int dataOffset;
}BitmapFileHeader;

//! The bitmap Info Header, stores detailed information about the image and contains the pixel format
typedef struct BitmapInfoHeader
{
    //! The size of this header in bytes (should be 40)
    unsigned int    headerSize;
    //! The width of the bitmap image
    int             imgWidth;
    //! The height of the bitmap image
    int             imgHeight;
    //! The number of color planes used
    unsigned short  colorPlanes;
    //! The number of bits per pixel
    unsigned short  bpp;
    //! The compression method being used
    unsigned int    compressionMethod;
    //! The size of the raw image data, could some times be 0 ...
    unsigned int    rawSize;
    //! Horizontal resolution of the image
    int             horResolution;
    //! Vertical resolution of the image
    int             verResolution;
    //! Number of colors in the color palette
    unsigned int    numColors;
    //! Number of important colors (ignored generally)
    unsigned int    numIColors;
}BitmapInfoHeader;

//! Bitmap file color table entry. Is in B-G-R-R(reserved) format
typedef struct  BitmapBGRR{

    unsigned char    blue;
    unsigned char    green;
    unsigned  char    red;
    unsigned  char    reserved;
}BitmapBGRR;
//go back to default packing
#pragma pack(pop)

//type of bitmaps
#define ME_24BIT_BMP    1
#define ME_8BIT_BMP     2
#define ME_4BIT_BMP     3
#define ME_1BIT_BMP     4
/**
** @date 22/12/2010
** @author Lefteris
**
** Represent a BMP image
** and can be used by appropriate functions
** to manipulate it and load it from the disc.
** At the moment only uncompressed BMP images
** with the most common header (BitmapInfoHeader) are supported
**/
typedef struct ME_ImageBMP
{
    //! The color table of the bitmap (only if bpp is 8 and below)
    BitmapBGRR* colorTable;
    //! Bitmap Width and height
    int imgWidth,imgHeight;
    //! Size of the BMP image in bytes
    unsigned int fileSize;
    //! The data of the image
    unsigned char* imgData;
    //! denotes the type of the bitmap image
    unsigned int type;
}ME_ImageBMP;


//! Initializes and returns a bmp image object from a file
//! @param filename The name of the file from which to initialize the image
//! @return The initialized BMP object
ME_ImageBMP* meImageBMP_Create(char* filename);
//! Initializes a bmp image object from a file
//!
//! @param bmp The bmp object to initialize
//! @param filename The name of the file from which to initialize the image
//! @return true for succesfull initialization and false otherwise
char meImageBMP_Init(ME_ImageBMP* bmp,char* filename);

//! @brief A bitmap saving to disk function.
//!
//! Saves the image only as 24 BPP uncompressed bitmap
//! @param bmp The bitmap image
//! @param filename The name of the file to save
int meImageBMP_Save(ME_ImageBMP* bmp,char* filename);


//! @brief Destroys a BMP image made with Create
//!
//! @param img The BMP image object to destroy
void meImageBMP_Destroy(ME_ImageBMP* img);
//! @brief Deinitializes a bmp image created with Init
//!
//! @param img The BMP image object to deinitialize
void meImageBMP_Deinit(ME_ImageBMP* img);


#endif //end of include guards

