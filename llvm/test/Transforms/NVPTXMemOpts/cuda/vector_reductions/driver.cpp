#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <cstdint> // Needed for int64_t

#define CUDA_DRIVER_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(CUresult code, const char *file, int line, bool abort=true) {
   if (code != CUDA_SUCCESS) {
      const char* errName;
      const char* errMsg;
      cuGetErrorName(code, &errName);
      cuGetErrorString(code, &errMsg);
      std::cerr << "CUDA Driver API error = " << errName << "(" << errMsg << ") at " << file << ":" << line << std::endl;
      if (abort) exit(code);
   }
}

int main() {
    int N = 131072; // Number of rows
    int M = 8192; // Number of columns

    size_t bytes_input = N * M * sizeof(int);
    size_t bytes_output = M * sizeof(int);

    int *h_input = new int[N * M];
    int *h_output = new int[M];
    
    // Initialize the input matrix
    for (int i = 0; i < N * M; i++) {
        h_input[i] = 1; // Example initialization
    }

    int *d_input, *d_output;
    cudaMalloc(&d_input, bytes_input);
    cudaMalloc(&d_output, bytes_output);

    cudaMemcpy(d_input, h_input, bytes_input, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(256);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x);

    // Initialize CUDA
    CUdevice cuDevice;
    CUcontext cuContext;
    CUmodule cuModule;
    CUfunction vectorReduction;
    CUDA_DRIVER_CHECK(cuInit(0));
    CUDA_DRIVER_CHECK(cuDeviceGet(&cuDevice, 0));
    CUDA_DRIVER_CHECK(cuCtxCreate(&cuContext, 0, cuDevice));
    CUDA_DRIVER_CHECK(cuModuleLoad(&cuModule, "test.fatbin"));
    CUDA_DRIVER_CHECK(cuModuleGetFunction(&vectorReduction, cuModule, "_Z5naivePfS_ii"));

    // Execute the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds;

    cudaEventRecord(start);
    void *args[] = { &d_input, &d_output, &N, &M};
    CUresult res = cuLaunchKernel(vectorReduction, 
        numBlocks.x, 1, 1, 
        threadsPerBlock.x, 1, 1, 
        0, 0, args, 0);
    CUDA_DRIVER_CHECK(res);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaMemcpy(h_output, d_output, bytes_output, cudaMemcpyDeviceToHost);

    bool success = true;
    if (success)
        std::cout << "TEST PASSED" << std::endl;
    else
        std::cout << "TEST FAILED" << std::endl;

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_output;

    return 0;
}
