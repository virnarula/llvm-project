#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_DRIVER_CHECK(result) { driverGpuAssert((result), __FILE__, __LINE__); }
inline void driverGpuAssert(CUresult code, const char *file, int line, bool abort=true) {
   if (code != CUDA_SUCCESS) {
      const char* errorString;
      cuGetErrorName(code, &errorString);
      fprintf(stderr, "Driver GPUassert: %s %s %d\n", errorString, file, line);
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
        h_A[i] = i + 10;  // Example: A[i] = i + 10
        h_B[i] = i;       // Example: B[i] = i
    }

    // Initialize CUDA
    CUdevice cuDevice;
    CUcontext cuContext;
    CUmodule cuModule;
    CUfunction vectorSubtract;
    cuInit(0);
    CUDA_DRIVER_CHECK(cuDeviceGet(&cuDevice, 0));
    CUDA_DRIVER_CHECK(cuCtxCreate(&cuContext, 0, cuDevice));
    CUDA_DRIVER_CHECK(cuModuleLoad(&cuModule, "test_06_substract.fatbin"));
    CUDA_DRIVER_CHECK(cuModuleGetFunction(&vectorSubtract, cuModule, "_Z14vectorSubtractPKfS0_Pfi"));

    // Allocate vectors in device memory
    CUdeviceptr d_A, d_B, d_C;
    CUDA_DRIVER_CHECK(cuMemAlloc(&d_A, size));
    CUDA_DRIVER_CHECK(cuMemAlloc(&d_B, size));
    CUDA_DRIVER_CHECK(cuMemAlloc(&d_C, size));

    // Copy vectors from host memory to device memory
    CUDA_DRIVER_CHECK(cuMemcpyHtoD(d_A, (void *)h_A, size));
    CUDA_DRIVER_CHECK(cuMemcpyHtoD(d_B, (void *)h_B, size));

    // Set up kernel parameters
    void *args[] = { &d_A, &d_B, &d_C, &numElements };
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    CUDA_DRIVER_CHECK(cuLaunchKernel(vectorSubtract, blocksPerGrid, 1, 1, threadsPerBlock, 1, 1, 0, 0, args, 0));
    CUDA_DRIVER_CHECK(cuCtxSynchronize());

    // Copy result from device memory to host memory
    CUDA_DRIVER_CHECK(cuMemcpyDtoH((void *)h_C, d_C, size));

    // Verify result
    for (int i = 0; i < numElements; ++i) {
        if (fabs((h_A[i] - h_B[i]) - h_C[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");

    // Clean up
    CUDA_DRIVER_CHECK(cuMemFree(d_A));
    CUDA_DRIVER_CHECK(cuMemFree(d_B));
    CUDA_DRIVER_CHECK(cuMemFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
    CUDA_DRIVER_CHECK(cuCtxDestroy(cuContext));

    return 0;
}