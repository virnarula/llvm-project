import os
import sys
import pathlib
import subprocess

RELATIVE_LLVM_BUILD_DIR = "../../../../../build/bin/"
OPT = "opt" 
LLVM_DIS = "llvm-dis"
FILECHECK = "FileCheck"

def run_lit_test(test_file):
    exec_string = script_dir + "/" + RELATIVE_LLVM_BUILD_DIR + OPT + " --nvptx-mem-opts < " + test_file +  " | " + \
        script_dir + "/" + RELATIVE_LLVM_BUILD_DIR + LLVM_DIS + " - | " +  \
        script_dir + "/" + RELATIVE_LLVM_BUILD_DIR + FILECHECK + " " + test_file
        # script_dir + "/" + RELATIVE_LLVM_BUILD_DIR + OPT + " -passes=dce | " +  \
    ret = subprocess.run(exec_string, shell=True)
    if ret.returncode != 0:
        print("Test: " + test_file +" - FAILED")
    else:
        print("Test: " + test_file + " - PASSED")

if __name__ == "__main__":
    # Get this script's directory
    script_dir = "."
    
    if len(sys.argv) == 2:
        run_lit_test(sys.argv[1])
    else:
        print("Usage: run_lit.py <test_file>")
        sys.exit(1)
        
        