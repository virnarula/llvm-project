extern "C" {

__device__ extern void __tb_size_marker_1D(int x) noexcept;
__device__ extern void __tb_size_marker_2D(int x, int y) noexcept;
__device__ extern void __tb_size_marker_3D(int x, int y, int z) noexcept;

__global__ void dummy_func() {
    __tb_size_marker_1D(1);
    __tb_size_marker_2D(1, 1);
    __tb_size_marker_3D(1, 1, 1);
}

}