#include <stdio.h>
#include <stdlib.h>
#include <math.h>   // for fabs
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

    // Initialize CUDA
    CUdevice cuDevice;
    CUcontext cuContext;
    CUmodule cuModule;
    CUfunction vectorAdd;
    cuInit(0);
    cuDeviceGet(&cuDevice, 0);
    cuCtxCreate(&cuContext, 0, cuDevice);
    cuModuleLoad(&cuModule, "test.fatbin");
    cuModuleGetFunction(&vectorAdd, cuModule, "_Z9vectorAddPKfS0_Pfi");

    // Allocate vectors in device memory
    CUdeviceptr d_A, d_B, d_C;
    cuMemAlloc(&d_A, size);
    cuMemAlloc(&d_B, size);
    cuMemAlloc(&d_C, size);

    // Copy vectors from host memory to device memory
    cuMemcpyHtoD(d_A, (void *)h_A, size);
    cuMemcpyHtoD(d_B, (void *)h_B, size);

    // Invoke kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    void *args[] = { &d_A, &d_B, &d_C, &numElements };
    cuLaunchKernel(vectorAdd, blocksPerGrid, 1, 1, threadsPerBlock, 1, 1, 0, 0, args, 0);

    // Copy result from device memory to host memory
    cuMemcpyDtoH((void *)h_C, d_C, size);

    // Verify result
    for (int i = 0; i < numElements; ++i) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");

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

    return 0;
}
