#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

inline void checkCuda(cudaError_t result, const char *file, int line) {
   if (result != cudaSuccess) {
      fprintf(stderr, "CUDA error: %s %s %d\n", cudaGetErrorString(result), file, line);
      exit(result);
   }
}

int main(void) {
    int N = 16384; // Matrix dimension
    size_t size = N * N * sizeof(float);
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Initialize matrices A and B with random values
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // Initialize CUDA
    CUdevice cuDevice;
    CUcontext cuContext;
    CUmodule cuModule;
    CUfunction matMulKernel;
    cuInit(0);
    cuDeviceGet(&cuDevice, 0);
    cuCtxCreate(&cuContext, 0, cuDevice);
    cuModuleLoad(&cuModule, "test.fatbin");
    cuModuleGetFunction(&matMulKernel, cuModule, "_Z6matmulPKfS0_Pfi");

    // Allocate vectors in device memory
    CUdeviceptr d_A, d_B, d_C;
    cuMemAlloc(&d_A, size);
    cuMemAlloc(&d_B, size);
    cuMemAlloc(&d_C, size);

    // Copy matrices from host memory to device memory
    cuMemcpyHtoD(d_A, (void *)h_A, size);
    cuMemcpyHtoD(d_B, (void *)h_B, size);

    // Prepare kernel launch
    int threadsPerBlock = 16;
    dim3 blocksPerGrid((N + threadsPerBlock - 1) / threadsPerBlock, (N + threadsPerBlock - 1) / threadsPerBlock);
    void *args[] = { &d_A, &d_B, &d_C, &N };

    // Invoke kernel
    cuLaunchKernel(matMulKernel, blocksPerGrid.x, blocksPerGrid.y, 1,
                   threadsPerBlock, threadsPerBlock, 1, 0, 0, args, 0);

    // Copy result from device memory to host memory
    cuMemcpyDtoH((void *)h_C, d_C, size);

    // Free device memory
    cuMemFree(d_A);
    cuMemFree(d_B);
    cuMemFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    // Cleanup CUDA
    cuCtxDestroy(cuContext);

    printf("Matrix multiplication completed successfully\n");

    return 0;
}
