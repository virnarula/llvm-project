#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for vector squaring
__global__ void vectorSquare(const float *input, float *output, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        output[i] = input[i] * input[i];
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
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    vectorSquare<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, numElements);

    // Copy result from device memory to host memory
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Verify result
    for (int i = 0; i < numElements; ++i) {
        if (fabs(h_input[i] * h_input[i] - h_output[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Free host memory
    free(h_input);
    free(h_output);

    return 0;
}
