const int BLUR_SIZE = 1;

/*
Each pixel is average of the pixels in the patch whose size iss (2*BLUR_SIZE+1) x (2*BLUR_SIZE+1).
Note that we don't care about the inconsistency after execution of each thread.
*/
__global__
void blurKernel(unsigned char *in, unsigned char *out, int w, int h) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < w && row < h) {
        int pixVal = 0;
        int numPixels = 0;

        for (int r=-BLUR_SIZE; r<BLUR_SIZE+1; ++r) {
            for (int c=-BLUR_SIZE; c<BLUR_SIZE+1; ++c) {
                int curRow = row + r;
                int curCol = col + c;

                if (curRow >= 0 && curRow < h && curCol >= 0 && curCol < w) {
                    pixVal += in[curRow * w + curCol];  // The flattend array of the given image.
                    ++numPixels;
                }
            }
        }

        out[row * w + col] = (unsigned char) (pixVal / numPixels);
    }
}

