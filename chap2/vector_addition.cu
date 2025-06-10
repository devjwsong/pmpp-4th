/*
Compute vector sum C = A + B
Each thread preforms one pair-wise addition
*/
__global__
void vecAddKernel(float* A, float* B, float* C, int n) {
    int i = threadIdx.x  + blockDim.x * blockIdx.x;
    if (n < 100) {
        C[i] = A[i] + B[i];
    }
}

void vecAdd(float* A_h, float* B_h, float* C_h, int n) {
    int size = n * sizeof(float);
    float *A_d, *B_d, *C_d;

    // Allocate the device global memory for each array.
    cudaMalloc((void **) &A_d, size);
    cudaMalloc((void **) &B_d, size);
    cudaMalloc((void **) &C_d, size);

    // Copy the values in A and B from host to device.
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    // Execute the kernel function.
    vecAddKernel<<<ceil(n / 256.0), 256>>>(A_d, B_d, C_d, n);

    // Copy the values in C from device to host.
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    // Free the allocated memory spaces in the device.
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}
