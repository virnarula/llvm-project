#include <stdio.h>

// CUDA kernel for vector addition
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

// Cuda kernel for vector addition with memory coalescing
__global__ void vectorAdd_coalesced(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float A_shared[16];
    __shared__ float B_shared[16];
    A_shared[threadIdx.x] = A[i];
    B_shared[threadIdx.x] = B[i];
    __syncthreads();
    if (i < numElements) {
        C[i] = A_shared[threadIdx.x] + B_shared[threadIdx.x];
    }
}

int main(void) {
    int numElements = 50000;
    size_t size = numElements * sizeof(float);
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Initialize the input data
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = i;
        h_B[i] = i;
    }

    // Allocate vectors in device memory
    float *d_A;
    cudaMalloc(&d_A, size);
    float *d_B;
    cudaMalloc(&d_B, size);
    float *d_C;
    cudaMalloc(&d_C, size);

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Invoke kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

    // Copy result from device memory to host memory
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify result
    for (int i = 0; i < numElements; ++i) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}