# Loop Analyzer
## Setup
Build using `ninja clang loop-analyzer`
## CL Usage
1. Compile with clang, turn on profile training
```
clang++ -
```
1. 

2. Compile with clang, with profiling information
```
clang++ -fsave-optimization-record=bitstream -foptimization-record-passes=loop-extract-analysis [other-flags] <filename> -o <output_filename>
```
2. Extract Remarks
```
dsymutil <output_filename>
```
3. Loop Analysis
```
loop-analyzer <output_filename>.dSYM/Contents/Resources/Remarks/<output_filename>
```

## Program Description

When we run the `loop-extract-analysis` pass, we are emiting a remark for each loop in the program. Our `loop-analyzer`