const int CHANNELS = 3;

/*
The input image is encoded as unsigned chars [0, 255].
Each pixel is 3 consecutive chars for the 3 channels. (RGB)
*/
__global__
void colorToGrayscaleConversion(unsigned char *Pout, unsigned char *Pin, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        // Get 1D offset for the grayscale image.
        int grayOffset = row * width + col;

        // One can think of the RGB image having CHANNEL.
        // times more columns than the gray scale image.
        int rgbOffset = grayOffset * CHANNELS;
        unsigned char r = Pin[rgbOffset];
        unsigned char g = Pin[rgbOffset+1];
        unsigned char b = Pin[rgbOffset+2];

        // Perform the rescaling and store it.
        // We multiple by floating point constants.
        Pout[grayOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
    }
}
