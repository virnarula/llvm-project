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
    float h_C = 0.0f;
    float *partial_C;

    // Initialize the input data
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = i;
        h_B[i] = 2.0f;
    }

    // Initialize CUDA
    CUdevice cuDevice;
    CUcontext cuContext;
    CUmodule cuModule;
    CUfunction vectorDotProduct;
    cuInit(0);
    CUDA_DRIVER_CHECK(cuDeviceGet(&cuDevice, 0));
    CUDA_DRIVER_CHECK(cuCtxCreate(&cuContext, 0, cuDevice));
    CUDA_DRIVER_CHECK(cuModuleLoad(&cuModule, "test_07_dot.fatbin"));
    CUDA_DRIVER_CHECK(cuModuleGetFunction(&vectorDotProduct, cuModule, "_Z16vectorDotProductPKfS0_Pfi"));

    // Allocate vectors in device memory
    CUdeviceptr d_A, d_B, d_C;
    CUDA_DRIVER_CHECK(cuMemAlloc(&d_A, size));
    CUDA_DRIVER_CHECK(cuMemAlloc(&d_B, size));
    int blocksPerGrid = 256;
    CUDA_DRIVER_CHECK(cuMemAlloc(&d_C, blocksPerGrid * sizeof(float)));
    partial_C = (float *)malloc(blocksPerGrid * sizeof(float));

    // Copy vectors from host memory to device memory
    CUDA_DRIVER_CHECK(cuMemcpyHtoD(d_A, (void *)h_A, size));
    CUDA_DRIVER_CHECK(cuMemcpyHtoD(d_B, (void *)h_B, size));

    // Set the number of threads per block
    int threadsPerBlock = 256;
    size_t sharedMemSize = threadsPerBlock * sizeof(float);

    // Launch the kernel
    void *args[] = { &d_A, &d_B, &d_C, &numElements };
    CUDA_DRIVER_CHECK(cuLaunchKernel(vectorDotProduct,
                                     blocksPerGrid, 1, 1,
                                     threadsPerBlock, 1, 1,
                                     sharedMemSize, 0, // Correct placement of sharedMemSize and stream
                                     args, nullptr));
    CUDA_DRIVER_CHECK(cuCtxSynchronize());

    // Copy the array of partial sums from the device to the host
    CUDA_DRIVER_CHECK(cuMemcpyDtoH((void *)partial_C, d_C, blocksPerGrid * sizeof(float)));

    // Finish reduction on the host
    for (int i = 0; i < blocksPerGrid; ++i) {
        h_C += partial_C[i];
    }

    // Display the result
    printf("Dot Product is: %f\n", h_C);

    // Clean up
    CUDA_DRIVER_CHECK(cuMemFree(d_A));
    CUDA_DRIVER_CHECK(cuMemFree(d_B));
    CUDA_DRIVER_CHECK(cuMemFree(d_C));
    free(h_A);
    free(h_B);
    free(partial_C);
    CUDA_DRIVER_CHECK(cuCtxDestroy(cuContext));

    return 0;
}
