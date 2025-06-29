#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TILE_WIDTH 2


/*
Use tiling to load the input element from global memory to shared memory
to boost the optimize the global memory access.
*/
__global__ void matrixMulKernel(float *M, float *N, float* P, int M_height, int M_width, int N_width) {

    // Declaration of the arrays in shared memory.
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    // Row and Column to process.
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    // Loop over each phase. The number of phase is WIDTH/TILE_WIDTH.
    float pvalue = 0.0f;
    for (int ph=0; ph<ceil(M_width/ (float) TILE_WIDTH); ++ph) {
        // Boundary check for M.
        if ((row < M_height) && ((ph * TILE_WIDTH + tx) < M_width)) {
            Mds[ty][tx] = M[row * M_width + ph * TILE_WIDTH + tx];
        } else {
            Mds[ty][tx] = 0.0f;
        }

        // Boundary check for N.
        if ((col < N_width) && ((ph * TILE_WIDTH + ty) < M_width)) {
            Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * N_width + col];
        } else {
            Nds[ty][tx] = 0.0f;
        }

        // Sync: True decpendency (read-after-write)
        __syncthreads();

        for (int k=0; k<TILE_WIDTH; ++k) {
            pvalue += (Mds[ty][k] * Nds[k][tx]);
        }

        // Sync: False dependency (write-after-read)
        __syncthreads();
    }

    if ((row < M_height) && (col < N_width)) {
        P[row * N_width + col] = pvalue;
    }
}


int main() {
    int gridDim, blockDim;
    printf("[Grid size & Block size]:\n");
    scanf("%d %d", &gridDim, &blockDim);

    int M_height, M_width, N_width;
    printf("[Matrix 1 Height]:\n");
    scanf("%d", &M_height);
    printf("[Matrix 1 Width / Matrix 2 Height]:\n");
    scanf("%d", &M_width);
    printf("[Matrix 2 Width]:\n");
    scanf("%d", &N_width);
    printf("Performing matrix multiplication between (%d x %d) and (%d x %d)\n", M_height, M_width, M_width, N_width);

    // Allocate M, N, P.
    float *M_h = (float *) malloc(M_height * M_width * sizeof(float));
    float *N_h = (float *) malloc(M_width * N_width * sizeof(float));
    float *P_h = (float *) malloc(M_height * N_width * sizeof(float));
    float *M_d, *N_d, *P_d;

    // Get the values.
    printf("[Matrix 1]:\n");
    for (int i=0; i<M_height * M_width; ++i) {
        scanf("%f", &M_h[i]);
    }
    printf("[Matrix 2]:\n");
    for (int i=0; i<M_width * N_width; ++i) {
        scanf("%f", &N_h[i]);
    }

    // Allocate the device global memory for each matrix.
    cudaMalloc((void **) &M_d, M_height * M_width * sizeof(float));
    cudaMalloc((void **) &N_d, M_width * N_width * sizeof(float));
    cudaMalloc((void **) &P_d, M_height * N_width * sizeof(float));

    // Copy the values in A and B from host to device.
    cudaMemcpy(M_d, M_h, M_height * M_width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N_h, M_width * N_width * sizeof(float), cudaMemcpyHostToDevice);

    // Execute the kernel function.
    matrixMulKernel<<<dim3(gridDim, gridDim), dim3(blockDim, blockDim)>>>(M_d, N_d, P_d, M_height, M_width, N_width);

    // Copy the values in P from device to host.
    cudaMemcpy(P_h, P_d, M_height * N_width * sizeof(float), cudaMemcpyDeviceToHost);

    // Free the allocated memory spaces in the device.
    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);

    // Print the output.
    printf("[Output]:\n");
    for (int i=0; i<M_height; ++i) {
        for (int j=0; j<N_width; ++j) {
            printf("%f ", P_h[i * N_width + j]);
            if (j == N_width-1) printf("\n");
        }
    }

    return 0;
}