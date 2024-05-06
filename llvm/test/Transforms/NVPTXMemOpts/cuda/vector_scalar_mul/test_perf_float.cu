#include <iostream>
#include <cuda_runtime.h>



// CUDA kernel for vector scalar multiplication
__global__ void vectorScalarMultiply(const float *input, float *output, float scalar, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i+16 < numElements)
        auto tmp = input[i];
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

int main() {
    size_t N = 134217728;
    size_t bytes = N * sizeof(float);
    float *h_A = new float[N];
    float *h_C = new float[N];  
    float *h_C_coalesced = new float[N];  

    // Initialize matrices
    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f; 
    }

    float *d_A, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 32;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds;

    // Naive matrix multiplication
    cudaEventRecord(start);
    vectorScalarMultiply<<<numBlocks, threadsPerBlock>>>(d_A, d_C, 15, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Naive execution time: " << milliseconds << " ms\n";
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Coalesced matrix multiplication
    cudaEventRecord(start);
    vectorScalarMultiply_coalesced<<<numBlocks, threadsPerBlock>>>(d_A, d_C, 15, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Coalesced execution time: " << milliseconds << " ms\n";
    cudaMemcpy(h_C_coalesced, d_C, bytes, cudaMemcpyDeviceToHost);

    // Verify correctness
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (h_C[i] != 15 || h_C_coalesced[i] != 15) {
            std::cerr << "Error: Matrix result is incorrect at index " << i << std::endl;
            correct = false;
            break;
        }
    }
    if (correct) {
        std::cout << "All results are correct." << std::endl;
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_C;
    delete[] h_C_coalesced;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
