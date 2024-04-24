# CS 526 Course Project - GPU Memory Optimizations
Authors: Vir Narula (vnarula2) & Chamika Sudusinghe (chamika2)

## Overview
In this project, we implement two GPU memory optimizations known as memory coalescing and prefetching for the NVIDIA PTX backend.

## Setup
Ensure you have the following dependencies:
1. Cmake
2. Python
3. Ninja
4. gcc & g++
5. ccache (optional)

Follow the same build instructions normally given for LLVM, listed below. Hopefully these steps work for your system:

1. `cd llvm-project`
2. `cmake -G"Ninja" -DCMAKE_C_COMPILER_LAUNCHER="ccache" -DCMAKE_CXX_COMPILER_LAUNCHER="ccache" -DCMAKE_BUILD_TYPE=Debug  -DLLVM_OPTIMIZED_TABLEGEN=ON -DLLVM_ENABLE_PROJECTS="clang" -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" -B build -S llvm`
3. `cd ./build`
4. `ninja`

## Project Details

- Our memory optomization pass is under `./llvm/lib/Target/NVPTX`.
- Our tests are under `./llvm/test/Transforms/NVPTXMemOpts/`
- You can find the NVPTX optimization pipeline under `./llvm/lib/Target/NVPTX/NVPTXTargetMachine.cpp`

### Running the tests
We have not set up automated testing yet. For now, you must
1. `cd ./llvm/test/Transforms/NVPTXMemOpts`
2. `../../../../build/bin/clang -S --cuda-gpu-arch=sm_35 -emit-llvm ./test.cu`.
This produces 2 files: `test.ll` & `test-cuda-nvptx64-nvidia-cuda-sm_25.bc`. We are only concerend with the `.bc` file as this gives us the generated kernel function.
3. `../../../../build/bin/llc -march=nvptx64 -mcpu=sm_35 test-cuda-nvptx64-nvidia-cuda-sm_35.bc -o test.ptx` this will compile the kernel function from llvm to ptx.
4. `nvcc --fatbin test.ptx`, which produces `test.fatbin`
5. `nvcc -lcuda driver.cpp`
6. `./a.out`





