import os
import sys
import subprocess

RELATIVE_LLVM_BUILD_DIR = "../../../../../build/bin/"
LLC = "llc"
OPT = "opt" 
LLVM_DIS = "llvm-dis"
FILECHECK = "FileCheck"

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: run_lit.py <test_dir> - do not include the trailing /")
        sys.exit(1)
    else:
        test_dir = sys.argv[1]
    
    # Get this script's directory
    script_dir = "."
    
    test_dirs = [d for d in os.listdir(script_dir) if os.path.isdir(d)]
    # check if test_dir is in test_dirs
    if test_dir not in test_dirs:
        print("Test directory not found")
        sys.exit(1)
    
    
    # Check if the test directory has a test.cuda and a driver.cpp file
    test_files = [f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))]
    if "test-cuda-nvptx64-nvidia-cuda-sm_35.ll" not in test_files or "driver.cpp" not in test_files:
        print("This test is not a runnable cuda test. Please try another test directory")
        print("The test directory should contain a test-cuda-nvptx64-nvidia-cuda-sm_35.ll and a driver.cpp file")
        sys.exit(1)
    
    # Get the test.cu and driver.cpp files
    test_file = os.path.join(test_dir, "test-cuda-nvptx64-nvidia-cuda-sm_35.ll")
    driver_file = os.path.join(test_dir, "driver.cpp")
    
    # Run llc on the test file
    llc_exec_string = script_dir + "/" + RELATIVE_LLVM_BUILD_DIR + LLC + " -march=nvptx64 -mcpu=sm_35 " \
        + test_file + " -o " + test_dir + "/test.ptx"
    # invoke command
    ret = subprocess.run(llc_exec_string, shell=True)
    
    nvcc_fatbin_exec_string = "nvcc --fatbin " + test_dir + "/test.ptx"
    ret = subprocess.run(nvcc_fatbin_exec_string, shell=True)
    
    nvcc_driver = "nvcc -lcuda " + test_dir + "/driver.cpp -o " + test_dir + "/a.out"
    ret = subprocess.run(nvcc_driver, shell=True)
    
    exec_string = "./" + test_dir + "/a.out"
    ret = subprocess.run(exec_string, shell=True)
    
    # print(llc_exec_string)
    # print(nvcc_fatbin_exec_string)
    # print(nvcc_driver)
    # print(exec_string)
    
    
    