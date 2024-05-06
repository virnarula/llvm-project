#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(result) { gpuAssert((result), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
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

    // Initialize CUDA
    CUdevice cuDevice;
    CUcontext cuContext;
    CUmodule cuModule;
    CUfunction transposeMatrix;
    cuInit(0);
    cuDeviceGet(&cuDevice, 0);
    cuCtxCreate(&cuContext, 0, cuDevice);
    cuModuleLoad(&cuModule, "test.fatbin");
    cuModuleGetFunction(&transposeMatrix, cuModule, "_Z15transposeMatrixPKfPfii");

    // Allocate matrices in device memory
    CUdeviceptr d_input, d_output;
    cuMemAlloc(&d_input, size);
    cuMemAlloc(&d_output, size);

    // Copy matrix from host memory to device memory
    cuMemcpyHtoD(d_input, (void *)h_input, size);

    // Set up the execution configuration
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Invoke kernel
    void *args[] = { &d_input, &d_output, &width, &height };
    cuLaunchKernel(transposeMatrix, blocksPerGrid.x, blocksPerGrid.y, 1, 
                   threadsPerBlock.x, threadsPerBlock.y, 1, 0, 0, args, 0);

    // Copy result from device memory to host memory
    cuMemcpyDtoH((void *)h_output, d_output, size);

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
    cuMemFree(d_input);
    cuMemFree(d_output);
    free(h_input);
    free(h_output);

    // Cleanup CUDA
    cuCtxDestroy(cuContext);

    return 0;
}
