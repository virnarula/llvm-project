#include <stdio.h>

// CUDA kernel for vector scalar multiplication
__global__ void vectorScalarMultiply(const float *input, float *output, float scalar, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        output[i] = input[i] * scalar;
    }
}

// CUDA kernel for vector scalar multiplication
__global__ void vectorScalarMultiply_coalesced(const float *input, float *output, float scalar, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float input_shared[16];
    input_shared[threadIdx.x] = input[i];
    __syncthreads();
    if (i < numElements) {
        output[i] = input_shared[threadIdx.x] * scalar;
    }
}

int main(void) {
    int numElements = 50000;
    size_t size = numElements * sizeof(float);
    float *h_input = (float *)malloc(size);
    float *h_output = (float *)malloc(size);

    // Initialize the input data
    for (int i = 0; i < numElements; ++i) {
        h_input[i] = i;
    }

    // Allocate vectors in device memory
    float *d_input;
    cudaMalloc(&d_input, size);
    float *d_output;
    cudaMalloc(&d_output, size);

    // Copy vector from host memory to device memory
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Invoke kernel
    float scalar = 3.0f;
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    vectorScalarMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, scalar, numElements);

    // Copy result from device memory to host memory
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Verify result
    for (int i = 0; i < numElements; ++i) {
        if (fabs(h_input[i] * scalar - h_output[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");

    // Free device and host memory
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
