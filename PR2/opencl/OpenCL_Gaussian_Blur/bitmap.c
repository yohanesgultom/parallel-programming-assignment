#include <bitmap.h>
#include <math.h> //for the ceiling function


#include <stdlib.h> //for malloc
#include <stdio.h>
#include <stdbool.h>
#include <string.h> //for memcpy

#ifdef _WIN32
#include <Winsock2.h>
#else
#include <arpa/inet.h>
#endif

static void swap_endian_ui(unsigned int *v)
{
   *v = (*v>>24) | ((*v<<8) & 0x00FF0000) |
    ((*v>>8) & 0x0000FF00) | (*v<<24);
}
static void swap_endian_us(unsigned short *v)
{
    *v = (*v>>8)|(*v<<8);
}

static bool is_big_endian()
{
    if (htonl(47) == 47) {
        return true;
    }
    return false;
}

// Initializes and returns a bmp image object from a file
ME_ImageBMP* meImageBMP_Create(char* fileName)
{
    ME_ImageBMP* ret;
    ret = malloc(sizeof(ME_ImageBMP));
    //try to initialize it and if it fails return 0
    if(meImageBMP_Init(ret,fileName) == false)
    {
        return 0;
    }
    return ret;
}//end of Creating a BMP image

//Initializes a bmp image
char meImageBMP_Init(ME_ImageBMP* bmp,char* fileName)
{
    int i,j;

    //the bitmap file header
    BitmapFileHeader bmpH;
    BitmapInfoHeader bmpIH;
    //The number of colors, if using a color table
    unsigned int numColors = 0;

    //open the file (binary mode)
    FILE *f = fopen(fileName,"rb");

    //initialize the color table to 0
    bmp->colorTable = 0;

    //Read the BITMAP file header (14 bytes)
    fread(&bmpH,sizeof(BitmapFileHeader),1,f);
    //0x00 From the BMP HEADER: Read the signature of the file and if it is not the BMP file magic number stop
    if(bmpH.magicNumber[0] != 0x42)
        return false;

    //0x02 From the BMP HEADER: Read the size of the file
    bmp->fileSize = bmpH.fileSize;
    if(is_big_endian())//.bmp is always litle endian so if the system is big endian make sure to change it
    {
        swap_endian_ui(&bmp->fileSize);
        swap_endian_ui(&bmpH.dataOffset);
    }


    //Read the BITMAP Info file header (40 bytes)
    fread(&bmpIH,sizeof(BitmapInfoHeader),1,f);

    if(is_big_endian())//.bmp is always litle endian so if the system is big endian make sure to change it
    {
        swap_endian_ui(&bmpIH.headerSize);
        swap_endian_ui(&bmpIH.compressionMethod);
        swap_endian_ui((unsigned int*) &bmpIH.imgWidth); //casting them since we only do byte level manipulation
        swap_endian_ui((unsigned int*) &bmpIH.imgHeight);
        swap_endian_us(&bmpIH.colorPlanes);
        swap_endian_us(&bmpIH.bpp);
        swap_endian_ui(&bmpIH.rawSize);
        swap_endian_ui((unsigned int*)&bmpIH.horResolution);
        swap_endian_ui((unsigned int*)&bmpIH.verResolution);
        swap_endian_ui(&bmpIH.numIColors);
        swap_endian_ui(&bmpIH.numColors);

    }

    //If size is not 40, then it must be a different version of the info header. It's not supported so return
    if(bmpIH.headerSize != 40)
        return false;

    //no compression method is supported yet, so make sure that it is uncompressed
    if(bmpIH.compressionMethod != 0)
    {
        return false;
    }

    //SAVE the image width
    bmp->imgWidth = bmpIH.imgWidth;
    bmp->imgHeight = bmpIH.imgHeight;
    //if bits per pixel is less than or equal to 8 we do have a color table so read that in too
    if(bmpIH.bpp <=8)
    {
        //The number of colours which is 2^n  , where n is the bits per pixel of the image
        numColors = 1 << bmpIH.bpp;
        //create the color table accordingly
        bmp->colorTable = malloc(sizeof(BitmapBGRR)*numColors);
        fread(bmp->colorTable,sizeof(BitmapBGRR),numColors,f);
    }

    //Calculate the bytes required to store a row of the image (padding included)
    unsigned int imgWidthBytes =ceil((float)(bmp->imgWidth*(float)bmpIH.bpp/32))*4;
    //the img width in bytes without padding
    unsigned int imgWidthBytesNP = ceil((float)bmp->imgWidth*(float)bmpIH.bpp/8);
    //calculate the offset (the difference between the normal imgWidth and the one with padding
    int offset = imgWidthBytes - imgWidthBytesNP;
    //since some times the raw size inside the info header can be zero we obtain the raw size from this calculation
    unsigned int dataSize = bmp->imgHeight*imgWidthBytes;
    //a buffer to hold the data of the bitmap file we are going to read, we don't read it from the header since it might be zero some times there
    unsigned char* dataBuff;
    dataBuff = malloc(dataSize);

    //read the data from the bitmap file
    fread(dataBuff,1,dataSize,f);

    ///Begin reading image data from the point of the file where they should be placed
    fseek(f,bmpH.dataOffset,SEEK_SET);

    //Height being positive means the image is stored normally
    if(bmp->imgHeight>0)
    {
        //according to the color depth
        switch(bmpIH.bpp)
        {
            case 24:
                //allocate the new image
                bmp->imgData = malloc(bmp->imgWidth*bmp->imgHeight*3);
                bmp->type = ME_24BIT_BMP;
                //i is counter for bytes in the image, j is counter for bytes in the bitmap file
                for(i=0,j=0;j<dataSize-3;i+=3,j+=3)
                {
                    //jump over the padding at the start of a new line
                    if((j+offset)%imgWidthBytes==0)
                    {
                        j+=offset;
                    }
                    //load it in RGB instead of the BGR it is saved inside
                    bmp->imgData[i+2] = dataBuff[j];//blue
                    bmp->imgData[i+1] = dataBuff[j+1];//green
                    bmp->imgData[i]   = dataBuff[j+2];//red
                }
            break;

            case 8:
                //allocate the new image
                bmp->imgData = malloc(bmp->imgWidth*bmp->imgHeight*3);
                bmp->type = ME_8BIT_BMP;
                //! @todo implement it
                return false;
            break;

            case 4:
                //allocate the new image
                bmp->imgData = malloc(bmp->imgWidth*bmp->imgHeight*3);

                bmp->type = ME_4BIT_BMP;
                int choffset;
                if(offset!=0)
                    choffset= offset+1;
                else
                    choffset = 0;
                //the 4 low and 4 high order bits
                unsigned char low,high;
                //i is counter for bytes in the image, j is counter for bytes in the bitmap file
                for( i=0,j=0;j<dataSize-3;i+=6,j++)
                {
                    //first let's read the byte
                    //the 4 low order bits
                    low = dataBuff[j] & 0x0F;
                    //the 4 high order bits
                    high = (dataBuff[j] & 0xF0)>>4;

                    bmp->imgData[i] = bmp->colorTable[high].red;
                    bmp->imgData[i+1] = bmp->colorTable[high].green;
                    bmp->imgData[i+2] = bmp->colorTable[high].blue;

                    bmp->imgData[i+3] = bmp->colorTable[low].red;
                    bmp->imgData[i+4] = bmp->colorTable[low].green;
                    bmp->imgData[i+5] = bmp->colorTable[low].blue;

                     //jump over the padding at the start of a new line
                    if((j+choffset)%imgWidthBytes==0)
                    {
                        j+=offset;
                        //jump the pixels we are not going to care about

                       //if the last row pixel uses half the byte, backtrack a bit
                       if(bmp->imgWidth%2!=0)
                            i-=3;
                        //if the addition of offset throws us off the dataSize (which is the for condition)
                        if(j>=dataSize)
                            break;//stop
                    }
                }
            break;

        case 1:
        {
            unsigned int byteIndex = 0;
            unsigned int rowBitIndex = 0;
            unsigned short bitIndex = 0;
            //allocate the new image
            bmp->imgData = calloc(bmp->imgWidth*bmp->imgHeight,1);
            //normally 1 bit bitmap can be a 2 color bitmap.
            bmp->type = ME_1BIT_BMP;
            //for each byte of the data
            for(i=0;i<dataSize-3;i++)
            {
                for(j=7;j>=0;j--)//for each bit of that byte
                {
                    rowBitIndex ++;
                    if(rowBitIndex > bmp->imgWidth) //if we are at the end of a row
                    {
                        i+=offset;
                        rowBitIndex = 0;
                        break;
                    }

                    if(bitIndex==8)//if a byte is full go to the next
                    {
                        bitIndex = 0;
                        byteIndex++;
                    }
                    //if we get here get the bit
                    bmp->imgData[byteIndex] |= (  ((dataBuff[i]>>j)&0x1)<<bitIndex );
                    //increase the bit index
                    bitIndex++;
                }
            }
        }
        break;
        default:
            return false;
        break;
        }//end of bits per pixel switch
    }//end of height being positive
    //then the bitmap is stored top to bottom
    //TEST these, these are not tested
    else
    {
       //according to the color depth
        switch(bmpIH.bpp)
        {
            case 24:
                //allocate the new image
                bmp->imgData = malloc(bmp->imgWidth*bmp->imgHeight*3);
                bmp->type = ME_24BIT_BMP;
                //counting backwards thanks to j
                //i is counter for bytes in the image, j is counter for bytes in the bitmap file
                for(i=0,j=dataSize-3;i<dataSize;i+=3,j-=3)
                {

                    //jump over the padding at the start of a new line
                    if((j-offset)%imgWidthBytes==0)
                        j-=offset;
                    //load it in RGB instead of the BGR it is saved inside
                    bmp->imgData[i+2] = dataBuff[j];//blue
                    bmp->imgData[i+1] = dataBuff[j-1];//green
                    bmp->imgData[i]   = dataBuff[j-2];//red

                }
            break;

            case 8:
                //allocate the new image
                bmp->imgData = malloc(bmp->imgWidth*bmp->imgHeight*3);
                bmp->type = ME_24BIT_BMP;
                //! @todo implement it
                return false;
            break;

            case 4://not tested
                //allocate the new image
                bmp->imgData = malloc(bmp->imgWidth*bmp->imgHeight*3);
                bmp->type = ME_4BIT_BMP;
                int choffset;
                if(offset!=0)
                    choffset= offset+1;
                else
                    choffset = 0;
                //counting backwards thanks to j
                //the 4 low and 4 high order bits
                unsigned char low,high;
                //i is counter for bytes in the image, j is counter for bytes in the bitmap file
                for(i=0,j=dataSize-3;j>0;i+=6,j--)
                {

                    //first let's read the byte

                    //the 4 low order bits
                    low = dataBuff[j] & 0x0F;
                    //the 4 high order bits
                    high = (dataBuff[j] & 0xF0)>>4;


                    bmp->imgData[i] = bmp->colorTable[high].red;
                    bmp->imgData[i+1] = bmp->colorTable[high].green;
                    bmp->imgData[i+2] = bmp->colorTable[high].blue;

                    bmp->imgData[i+3] = bmp->colorTable[low].red;
                    bmp->imgData[i+4] = bmp->colorTable[low].green;
                    bmp->imgData[i+5] = bmp->colorTable[low].blue;

                     //jump over the padding at the start of a new line
                    if((j-choffset)%imgWidthBytes==0)
                    {
                        j-=offset;
                        //jump the pixels we are not going to care about

                       //if the last row pixel uses half the byte, backtrack a bit
                       if(bmp->imgWidth%2!=0)
                            i-=3;
                        //if the addition of offset throws us off the dataSize (which is the for condition)
                        if(j>=0)
                            break;//stop
                    }
                }
            break;

            case 1:
                //allocate the new image
                bmp->imgData = malloc(bmp->imgWidth*bmp->imgHeight*3);
                bmp->type = ME_1BIT_BMP;
                //! @todo implement
                return false;
            break;
            default:
                return false;
            break;
        }//end of bpp switch

    }//end of reverse image else

    //close the file
    fclose(f);
    //free the temporary data buffer
    free(dataBuff);

    //return success
    return true;
}

//A bitmap saving to disk function.
int meImageBMP_Save(ME_ImageBMP* bmp,char* fileName)
{
    //the bitmap file header
    BitmapFileHeader bmpH;
    BitmapInfoHeader bmpIH;
    //The number of colors, if using a color table
    uint32_t i,j;

    //open the file (binary mode)
    FILE *f = fopen(fileName,"wb");

    //Write the BITMAP file header (14 bytes)
    bmpH.magicNumber[0] = 0x42;
    bmpH.magicNumber[1] = 0x4D;
    bmpH.fileSize = bmp->fileSize;
    bmpH.dataOffset = sizeof(BitmapFileHeader )+sizeof(BitmapInfoHeader);
    fwrite(&bmpH,sizeof(BitmapFileHeader),1,f);

    ///since this is just for testing, only one type of bitmap (24 bpp)will be saved
    bmpIH.bpp = 24; //only for 24 bpp
    //Calculate the bytes required to store a row of the image (padding included)
    unsigned int imgWidthBytes =ceil((float)(bmp->imgWidth*(float)bmpIH.bpp/32))*4;
    //the img width in bytes without padding
    unsigned int imgWidthBytesNP = ceil((float)bmp->imgWidth*(float)bmpIH.bpp/8);
    //calculate the offset (the difference between the normal imgWidth and the one with padding
    int offset = imgWidthBytes - imgWidthBytesNP;
    //since some times the raw size inside the info header can be zero we obtain the raw size from this calculation
    unsigned int dataSize = bmp->imgHeight*imgWidthBytes;
    bmpIH.headerSize = 40;
    bmpIH.compressionMethod = 0;
    bmpIH.imgWidth = bmp->imgWidth;
    bmpIH.imgHeight = bmp->imgHeight;
    bmpIH.colorPlanes = 1;
    bmpIH.bpp = 24; //only for 24 bpp
    bmpIH.rawSize = bmp->imgHeight*imgWidthBytes;
    bmpIH.horResolution = 1;
    bmpIH.verResolution = 1;
    bmpIH.numIColors=1;
    bmpIH.numColors=1;

    //Write the BITMAP Info file header (40 bytes)
    fwrite(&bmpIH,sizeof(BitmapInfoHeader),1,f);

    //a buffer to hold the data of the bitmap file we are going to write, we don't read it from the header since it might be zero some times there
    unsigned char* dataBuff;
    dataBuff = malloc(dataSize);
    //i is counter for bytes in the image, j is counter for bytes in the bitmap file
    for(i=0,j=0;j<dataSize-3;i+=3,j+=3)
    {
        //jump over the padding at the start of a new line
        if((j+offset)%imgWidthBytes==0)
        {
            j+=offset;
        }
        //load it in RGB instead of the BGR it is saved inside
        dataBuff[j] = bmp->imgData[i+2]; //blue
        dataBuff[j+1] = bmp->imgData[i+1];//green
        dataBuff[j+2] = bmp->imgData[i];//red
    }
    //write the data
    fwrite(dataBuff,dataSize,1,f);
    fclose(f);
    return 1;
}

// Destroys a BMP image object releasing its memory
void meImageBMP_Destroy(ME_ImageBMP* img)
{
    free(img->imgData);
    //if we have a color table free it
    if(img->colorTable!=0)
        free(img->colorTable);
    //also free the image object itself
    free(img);
}
//Deinitializes a bmp image created with Init
void meImageBMP_Deinit(ME_ImageBMP* img)
{
    free(img->imgData);
    //if we have a color table free it
    if(img->colorTable!=0)
        free(img->colorTable);
}


