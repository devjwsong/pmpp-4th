#include <stdio.h>
#include <stdlib.h>


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


/*
Each thread produces one output matrix row.
Pros: Efficient & Coalesced memory access for Matrix M.
Cons: Non-Coalesced memory access for Matrix N -> Scattered memory access and reduced memory bandwidth.
*/
__global__
void matrixMulRowKernel(float *M, float *N, float *P, int Width) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;  // Only one dimension needed.

    if (row < Width) {
        for (int col=0; col<Width; ++col) {
            float value = 0.0;
            for (int k=0; k<Width; ++k) {
                value += (M[row*Width+k] * N[k*Width+col]);
            }
            P[row*Width+col] = value;
        }
    }
}


/*
Each thread produces one output matrix column.
Pros: Efficient & Coalesced memory access for Matrix N.
Cons: Non-Coalesced memory access for Matrix M -> Scattered memory access and reduced memory bandwidth.
*/
__global__
void matrixMulColKernel(float *M, float *N, float *P, int Width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Only one dimension needed.

    if (col < Width) {
        for (int row=0; row<Width; ++row) {
            float value = 0.0;
            for (int k=0; k<Width; ++k) {
                value += (M[row*Width+k] * N[k*Width+col]);
            }
            P[row*Width+col] = value;
        }
    }
}


int main() {
    int gridDim, blockDim;
    printf("[Grid size & Block size]:\n");
    scanf("%d %d", &gridDim, &blockDim);

    int Width;
    printf("[Matrix Width]:\n");
    scanf("%d", &Width);
    printf("The size of a matrix is %d x %d.\n", Width, Width);

    // Allocate M, N, C.
    int memsize = Width * Width * sizeof(float);
    float *M_h = (float *) malloc(memsize);
    float *N_h = (float *) malloc(memsize);
    float *P_h = (float *) malloc(memsize);
    float *M_d, *N_d, *P_d;

    // Get the values.
    printf("[Matrix 1]:\n");
    for (int i=0; i<Width * Width; ++i) {
        scanf("%f", &M_h[i]);
    }
    printf("[Matrix 2]:\n");
    for (int i=0; i<Width * Width; ++i) {
        scanf("%f", &N_h[i]);
    }

    // Allocate the device global memory for each matrix.
    cudaMalloc((void **) &M_d, memsize);
    cudaMalloc((void **) &N_d, memsize);
    cudaMalloc((void **) &P_d, memsize);

    // Copy the values in A and B from host to device.
    cudaMemcpy(M_d, M_h, memsize, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N_h, memsize, cudaMemcpyHostToDevice);

    // Execute the kernel function.
    matrixMulKernel<<<dim3(gridDim, gridDim), dim3(blockDim, blockDim)>>>(M_d, N_d, P_d, Width);

    // Copy the values in P from device to host.
    cudaMemcpy(P_h, P_d, memsize, cudaMemcpyDeviceToHost);

    // Free the allocated memory spaces in the device.
    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);

    // Print the output.
    printf("[Output]:\n");
    for (int i=0; i<Width; ++i) {
        for (int j=0; j<Width; ++j) {
            printf("%f ", P_h[i * Width + j]);
            if (j == Width-1) printf("\n");
        }
    }

    return 0;
}
