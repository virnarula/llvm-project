#include <stdio.h>
#include <cuda.h>

// CUDA kernel to compute dot product of two vectors
__global__ void vectorDotProduct(const float *A, const float *B, float *C, int numElements) {
    extern __shared__ float temp[]; // shared memory for reduction
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int threadId = threadIdx.x;

    // Each thread computes one element of the block sub-dot product
    if (index < numElements) {
        temp[threadId] = A[index] * B[index];
    } else {
        temp[threadId] = 0.0f;
    }
    
    __syncthreads();  // Synchronize all threads within the block

    // Reduction: sum the block's sub-dot products
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadId < stride) {
            temp[threadId] += temp[threadId + stride];
        }
        __syncthreads();
    }

    // Write the result for this block to global memory
    if (threadId == 0) {
        C[blockIdx.x] = temp[0];
    }
}

int main(void) {
    int numElements = 50000;
    size_t size = numElements * sizeof(float);

    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float h_C = 0.0f;
    float *partial_C;

    // Initialize the input data
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = i;
        h_B[i] = 2.0f;
    }

    // Allocate vectors in device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    int blocksPerGrid = 256;
    cudaMalloc(&d_C, blocksPerGrid * sizeof(float));
    partial_C = (float *)malloc(blocksPerGrid * sizeof(float));

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Set the number of threads per block
    int threadsPerBlock = 256;

    // Invoke kernel
    vectorDotProduct<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_A, d_B, d_C, numElements);

    // Copy the array of partial sums from the device to the host
    cudaMemcpy(partial_C, d_C, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

    // Finish reduction on the host
    for (int i = 0; i < blocksPerGrid; ++i) {
        h_C += partial_C[i];
    }

    // Display the result
    printf("Dot Product is: %f\n", h_C);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(partial_C);

    return 0;
}
