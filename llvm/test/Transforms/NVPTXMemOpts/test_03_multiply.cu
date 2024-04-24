#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// CUDA kernel for vector multiplication
__global__ void vectorMultiply(const float *input1, const float *input2, float *output, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        output[i] = input1[i] * input2[i];
    }
}

int main(void) {
    int numElements = 50000;
    size_t size = numElements * sizeof(float);
    float *h_input1 = (float *)malloc(size);
    float *h_input2 = (float *)malloc(size);
    float *h_output = (float *)malloc(size);

    // Initialize the input data
    for (int i = 0; i < numElements; ++i) {
        h_input1[i] = i;
        h_input2[i] = 2 * i;  // Different values for the second vector
    }

    // Allocate vectors in device memory
    float *d_input1, *d_input2, *d_output;
    cudaMalloc(&d_input1, size);
    cudaMalloc(&d_input2, size);
    cudaMalloc(&d_output, size);

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_input1, h_input1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input2, h_input2, size, cudaMemcpyHostToDevice);

    // Invoke kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    vectorMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_input1, d_input2, d_output, numElements);

    // Copy result from device memory to host memory
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Verify result
    for (int i = 0; i < numElements; ++i) {
        if (fabs(h_input1[i] * h_input2[i] - h_output[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");

    // Free device and host memory
    cudaFree(d_input1);
    cudaFree(d_input2);
    cudaFree(d_output);
    free(h_input1);
    free(h_input2);
    free(h_output);

    return 0;
}
