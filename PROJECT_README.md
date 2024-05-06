# CS 526 Course Project - GPU Memory Optimizations
Authors: Vir Narula (vnarula2) & Chamika Sudusinghe (chamika2)

## Overview
In this project, we implement two GPU memory optimizations known as memory coalescing and prefetching for the NVIDIA PTX backend.

## Setup + Build
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

- Our memory optomization pass is under `./llvm/lib/Target/NVPTX/`. The relevant files are `NVPTXMemOpts.{h, cpp}`.
- The marker functions we need for our functions can be found under `./llvm/test/Transforms/NVPTXMemOpts/include`
- Our tests are under `./llvm/test/Transforms/NVPTXMemOpts/`. 
    - `lit` contains llvm-lit tests for correctness of our transformation. You can run the lit tests using `python run_lit.py <test-filename>`. 
    - `cuda` contains many unoptimized cuda kernels we used for our performance testing along with optimized versions for development. See instructios below.
- We have integrated our pass with the NVPTX optimization pipeline under `./llvm/lib/Target/NVPTX/NVPTXTargetMachine.cpp`

### Running non-LIT Tests
Not all the tests we wrote can be compiled into full CUDA programs. The script will let you know when that is the case.
1. `cd ./llvm/test/Transforms/NVPTXMemOpts/cuda`
2. `python run_cuda_program.py <test_folder>`







