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
    int numElements = 500000000;
    size_t size = numElements * sizeof(float);
    float scalar = 3.0f;
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
    CUfunction vectorScalarMultiply;
    cuInit(0);
    cuDeviceGet(&cuDevice, 0);
    cuCtxCreate(&cuContext, 0, cuDevice);
    cuModuleLoad(&cuModule, "test.fatbin");
    cuModuleGetFunction(&vectorScalarMultiply, cuModule, "_Z30vectorScalarMultiply_coalescedPKfPffi");

    // Allocate vectors in device memory
    CUdeviceptr d_input, d_output;
    cuMemAlloc(&d_input, size);
    cuMemAlloc(&d_output, size);

    // Copy vector from host memory to device memory
    cuMemcpyHtoD(d_input, (void *)h_input, size);

    // Prepare kernel launch
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    void *args[] = { &d_input, &d_output, &scalar, &numElements };

    // Invoke kernel
    cuLaunchKernel(vectorScalarMultiply, blocksPerGrid, 1, 1, threadsPerBlock, 1, 1, 0, 0, args, 0);

    // Copy result from device memory to host memory
    cuMemcpyDtoH((void *)h_output, d_output, size);

    // Verify result
    for (int i = 0; i < numElements; ++i) {
        if (fabs(h_input[i] * scalar - h_output[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");

    // Free device memory
    cuMemFree(d_input);
    cuMemFree(d_output);

    // Free host memory
    free(h_input);
    free(h_output);

    // Cleanup CUDA
    cuCtxDestroy(cuContext);

    return 0;
}
