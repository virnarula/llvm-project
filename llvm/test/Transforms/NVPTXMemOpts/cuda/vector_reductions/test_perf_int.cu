#include <cuda_runtime.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cstdint> // Needed for int64_t

__global__ void naive_reduction(int *input, int *output, int row, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < cols) {
        int sum = 0.0f;
        for (int i = 0; i < row; i++) {
            sum += input[i * cols + idx];
        }
        output[idx] = sum;
    }
}

__global__ void coalesced(int *input, int *output, int row, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < cols) {
        int sum = 0.0f;

        for (int i = 0; i < row; i += 16) {
            // Shared memory for coalesced access
            __shared__ int shared[16];
            shared[threadIdx.x] = input[i * cols + idx];
            __syncthreads();
            for (int k = 0; k < 16; k++) {
                sum += shared[k];
            }
            __syncthreads();
        }
        output[idx] = sum;
    }
}
__global__ void coalesced_prefetched(int *input, int *output, int row, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < cols) {
        int sum = 0.0f;

        // Shared memory for coalesced access
        int tmp = input[threadIdx.x * cols + idx];
        for (int i = 0; i < row; i += 16) {
            __shared__ int shared[16];
            shared[threadIdx.x] = tmp;
            __syncthreads();

            if (i + 16 < row) {
                tmp = input[(i + 16) * cols + idx];
            }

            for (int k = 0; k < 16; k++) {
                sum += shared[k];
            }
            __syncthreads();
        }
        output[idx] = sum;
    }
}



int main() {
    int N = 131072; // Number of rows
    int M = 8192; // Number of columns

    size_t bytes_input = N * M * sizeof(int);
    size_t bytes_output = M * sizeof(int);

    int *h_input = new int[N * M];
    int *h_output = new int[M];
    int *h_output_coalesced = new int[M];
    int *h_output_coalesced_prefetched = new int[M];
    
    // Initialize the input matrix
    for (int i = 0; i < N * M; i++) {
        h_input[i] = 1.0f; // Example initialization
    }

    int *d_input, *d_output;
    cudaMalloc(&d_input, bytes_input);
    cudaMalloc(&d_output, bytes_output);

    cudaMemcpy(d_input, h_input, bytes_input, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(256);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x);

    // Execute the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds;

    cudaEventRecord(start);
    naive_reduction<<<numBlocks, threadsPerBlock>>>(d_input, d_output, N, M);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Naive execution time: " << milliseconds << " ms\n";
    cudaMemcpy(h_output, d_output, bytes_output, cudaMemcpyDeviceToHost);

    cudaEventRecord(start);
    coalesced<<<numBlocks, threadsPerBlock>>>(d_input, d_output, N, M);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Coalesced execution time: " << milliseconds << " ms\n";
    cudaMemcpy(h_output_coalesced, d_output, bytes_output, cudaMemcpyDeviceToHost);

    cudaEventRecord(start);
    coalesced_prefetched<<<numBlocks, threadsPerBlock>>>(d_input, d_output, N, M);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Coalesced_prefetch execution time: " << milliseconds << " ms\n";
    cudaMemcpy(h_output_coalesced_prefetched, d_output, bytes_output, cudaMemcpyDeviceToHost);


    // Output the results
    for (int i = 0; i < M; i++) {
        if (h_output[i] != N || h_output_coalesced[i] != N || h_output_coalesced_prefetched[i] != N) {
            std::cout << "Error at index " << i << std::endl;
            std::cout << "Naive: " << h_output[i] << std::endl;
            std::cout << "Coalesced: " << h_output_coalesced[i] << std::endl;
            std::cout << "Coalesced Prefetched: " << h_output_coalesced_prefetched[i] << std::endl;
        }
    }

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_output;
    delete[] h_output_coalesced;
    delete[] h_output_coalesced_prefetched;

    return 0;
}