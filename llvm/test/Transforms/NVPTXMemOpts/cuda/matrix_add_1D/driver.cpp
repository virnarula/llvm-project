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
    int numCols = 4096;
    int numRows = 4096;
    int numElements = numCols * numRows;
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
    cuModuleGetFunction(&vectorAdd, cuModule, "_Z9matrixAddPKfS0_Pfii");

    // Allocate vectors in device memory
    CUdeviceptr d_A, d_B, d_C;
    cuMemAlloc(&d_A, size);
    cuMemAlloc(&d_B, size);
    cuMemAlloc(&d_C, size);

    // Copy vectors from host memory to device memory
    cuMemcpyHtoD(d_A, (void *)h_A, size);
    cuMemcpyHtoD(d_B, (void *)h_B, size);

    // Invoke kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(numCols / threadsPerBlock.x, numRows / threadsPerBlock.y);
    void *args[] = { &d_A, &d_B, &d_C, &numRows, &numCols};
    CUresult res = cuLaunchKernel(vectorAdd, 
        numBlocks.x, numBlocks.y, 1, 
        threadsPerBlock.x, threadsPerBlock.y, 1, 
        0, 0, args, 0);
    
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "cuLaunchKernel failed: %d\n", res);
        exit(EXIT_FAILURE);
    }

    // Copy result from device memory to host memory
    cuMemcpyDtoH((void *)h_C, d_C, size);

    // Verify result
    for (int i = 0; i < numElements; ++i) {
        if (h_A[i] + h_B[i] - h_C[i] != 0) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            fprintf(stderr, "h_A[%d] = %f\n", i, h_A[i]);
            fprintf(stderr, "h_B[%d] = %f\n", i, h_B[i]);
            fprintf(stderr, "h_C[%d] = %f\n", i, h_C[i]);
            fprintf(stderr, h_C[i] - h_A[i] - h_B[i] < 0 ? "Negative error\n" : "Positive error\n");
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
