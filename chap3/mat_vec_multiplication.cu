/*
Matrix B x Vector C multiplication.
Assume that the matrix is square.
*/
__global__
void matVecMultiKernel(float *A, float *B, float *C, int size) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size) {
        float value = 0.0;
        for (int col=0; col<size; ++col) {
            value += (B[row*size + col] * C[col]);
        }
        A[row] = value;
    }
}
