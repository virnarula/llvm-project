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
    float *h_input = (float *)malloc(size);
    float *h_output = (float *)malloc(size);

    // Initialize the input data
    for (int i = 0; i < numElements; ++i) {
        h_input[i] = i;
    }

    // Initialize CUDA
    CUdevice cuDevice;
    CUcontext cuContext;
    CUmodule cuModule;
    CUfunction vectorSquare;
    cuInit(0);
    CUDA_DRIVER_CHECK(cuDeviceGet(&cuDevice, 0));
    CUDA_DRIVER_CHECK(cuCtxCreate(&cuContext, 0, cuDevice));
    CUDA_DRIVER_CHECK(cuModuleLoad(&cuModule, "test_05_square.fatbin"));
    CUDA_DRIVER_CHECK(cuModuleGetFunction(&vectorSquare, cuModule, "_Z12vectorSquarePKfPfi"));

    // Allocate vectors in device memory
    CUdeviceptr d_input, d_output;
    CUDA_DRIVER_CHECK(cuMemAlloc(&d_input, size));
    CUDA_DRIVER_CHECK(cuMemAlloc(&d_output, size));

    // Copy vector from host memory to device memory
    CUDA_DRIVER_CHECK(cuMemcpyHtoD(d_input, (void *)h_input, size));

    // Set up kernel parameters
    void *args[] = { &d_input, &d_output, &numElements };
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    CUDA_DRIVER_CHECK(cuLaunchKernel(vectorSquare, blocksPerGrid, 1, 1, threadsPerBlock, 1, 1, 0, 0, args, 0));
    CUDA_DRIVER_CHECK(cuCtxSynchronize());

    // Copy result from device memory to host memory
    CUDA_DRIVER_CHECK(cuMemcpyDtoH((void *)h_output, d_output, size));

    // Verify result
    for (int i = 0; i < numElements; ++i) {
        if (fabs(h_input[i] * h_input[i] - h_output[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");

    // Clean up
    CUDA_DRIVER_CHECK(cuMemFree(d_input));
    CUDA_DRIVER_CHECK(cuMemFree(d_output));
    free(h_input);
    free(h_output);
    CUDA_DRIVER_CHECK(cuCtxDestroy(cuContext));

    return 0;
}
