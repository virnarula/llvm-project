#include <stdio.h>
#include <stdlib.h>
#include <math.h>   // for fabs
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(result) { gpuAssert((result), __FILE__, __LINE__); }
inline void gpuAssert(CUresult code, const char *file, int line, bool abort=true) {
   if (code != CUDA_SUCCESS) {
      const char *errorString;
      cuGetErrorName(code, &errorString);
      fprintf(stderr,"GPUassert: %s %s %d\n", errorString, file, line);
      if (abort) exit(code);
   }
}

int main(void) {
    int numElements = 50000;
    size_t size = numElements * sizeof(float);
    float *h_input1 = (float *)malloc(size);
    float *h_input2 = (float *)malloc(size);
    float *h_output = (float *)malloc(size);

    // Initialize CUDA
    CUdevice cuDevice;
    CUcontext cuContext;
    CUmodule cuModule;
    CUfunction vectorMultiply;
    cuInit(0);
    CUDA_CHECK(cuDeviceGet(&cuDevice, 0));
    CUDA_CHECK(cuCtxCreate(&cuContext, 0, cuDevice));
    CUDA_CHECK(cuModuleLoad(&cuModule, "test_03_multiply.fatbin"));
    CUDA_CHECK(cuModuleGetFunction(&vectorMultiply, cuModule, "_Z14vectorMultiplyPKfS0_Pfi"));

    // Allocate vectors in device memory
    CUdeviceptr d_input1, d_input2, d_output;
    CUDA_CHECK(cuMemAlloc(&d_input1, size));
    CUDA_CHECK(cuMemAlloc(&d_input2, size));
    CUDA_CHECK(cuMemAlloc(&d_output, size));

    // Copy vectors from host memory to device memory
    CUDA_CHECK(cuMemcpyHtoD(d_input1, (void *)h_input1, size));
    CUDA_CHECK(cuMemcpyHtoD(d_input2, (void *)h_input2, size));

    // Invoke kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    void *args[] = { &d_input1, &d_input2, &d_output, &numElements };
    CUDA_CHECK(cuLaunchKernel(vectorMultiply, blocksPerGrid, 1, 1, threadsPerBlock, 1, 1, 0, 0, args, 0));

    // Copy result from device memory to host memory
    CUDA_CHECK(cuMemcpyDtoH((void *)h_output, d_output, size));

    // Verify result
    for (int i = 0; i < numElements; ++i) {
        if (fabs(h_input1[i] * h_input2[i] - h_output[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");

    // Free device memory
    cuMemFree(d_input1);
    cuMemFree(d_input2);
    cuMemFree(d_output);

    // Free host memory
    free(h_input1);
    free(h_input2);
    free(h_output);

    // Cleanup CUDA
    cuCtxDestroy(cuContext);

    return 0;
}
