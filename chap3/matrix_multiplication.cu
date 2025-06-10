/*
Each thread takes one row from M and one column from N to produce one cell in the output matrix P.
We assume that the width and height are the same in this case, which removes the necessity of the additional integer Height.
*/
__global__
void matrixMulKernel(float *M, float *N, float* P, int Width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < Width && col < Width) {
        float value = 0.0;
        for (int k=0; k<Width; ++k) {
            value += (M[row*Width+k] * N[k*Width+col]);
        }
        P[row*Width+col] = value;
    }
}
