#include "tb_size_marker.cpp"

__global__ void naiveMM(const float *A, float *C, int w) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    float sum = 0;
    for (int i = 0; i < w; i++) {
        for (int j = 0; j < w; j++) {
            sum += A[i+j] * A[i+j];
        }
    }
    C[idx] = sum;
}