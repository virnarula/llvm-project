#include <stdio.h>

// CUDA kernel to transpose a matrix
__global__ void transposeMatrix(const float *input, float *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int pos = y * width + x;
        int transPos = x * height + y;
        output[transPos] = input[pos];
    }
}

int main(void) {
    int width = 1024;
    int height = 1024;
    size_t size = width * height * sizeof(float);
    float *h_input = (float *)malloc(size);
    float *h_output = (float *)malloc(size);

    // Initialize the input matrix
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            h_input[i * width + j] = i * width + j;
        }
    }

    // Allocate matrices in device memory
    float *d_input;
    cudaMalloc(&d_input, size);
    float *d_output;
    cudaMalloc(&d_output, size);

    // Copy matrix from host memory to device memory
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Set up the execution configuration
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Invoke kernel
    transposeMatrix<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, width, height);

    // Copy result from device memory to host memory
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Verify result
    bool success = true;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            if (h_output[j * height + i] != h_input[i * width + j]) {
                fprintf(stderr, "Result verification failed at element (%d, %d)!\n", i, j);
                success = false;
                break;
            }
        }
        if (!success) {
            break;
        }
    }

    if (success) {
        printf("Test PASSED\n");
    } else {
        printf("Test FAILED\n");
    }

    // Free device and host memory
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
