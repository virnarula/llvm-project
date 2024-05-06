#include <tb_size_marker.cpp>

__global__ void stencil1d(int n, float *input, float *output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure we do not access out of bounds memory
    if (idx > 4 && idx < n - 5) {
        float result = 0.0f;

        // Apply the stencil from idx-4 to idx+4
        for (int i = -4; i < 4; i++) {
            result += input[idx + i];
        }

        // Store the average of the neighborhood in the output array
        output[idx] = result / 7;  // Normalize by the number of elements, which is 9
    }
}

__global__ void stencil1d_coalesced(int n, float *input, float *output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure we do not access out of bounds memory
    if (idx > 4 && idx < n - 5) {
        float result = 0.0f;

        // Shared memory for coalescing
        __shared__ float input_shared[8][16];
        for (int i = -4; i < 4; i++) {
            input_shared[i+4][threadIdx.x] = input[idx + i];
        }

        // Apply the stencil from idx-4 to idx+4
        for (int i = -4; i < 4; i++) {
            result += input_shared[i+4][threadIdx.x];
        }

        // Store the average of the neighborhood in the output array
        output[idx] = result / 7;  // Normalize by the number of elements, which is 9
    }
}

