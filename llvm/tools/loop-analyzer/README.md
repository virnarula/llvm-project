# Loop Analyzer
## Build Project
Build using: `ninja clang loop-analyzer`.
## CL Usage
### Build Codebase to Analyze 

1. Compile once with clang, turn on profiling options
```
clang++ -fprofile-instr-generate <file.c> && a.out
```
2. Run with profile information
```
LLVM_PROFILE_FILE=a.profraw ./a.out
```
3. Process raw profiling data
```
llvm-profdata a.profraw -o a.profdata
```

### Analyze Build
4. Build again using profile information
```
clang -fprofile-instr-use=a.profdata <file.c> -o a.profile.out
```

5. Extract Remarks
```
dsymutil a.profile.out
```
6. Loop Analysis
```
loop-analyzer <output_filename>.dSYM/Contents/Resources/Remarks/<output_filename>
```

## Program Description

<!-- When we run the `loop-extract-analysis` pass, we are emiting a remark for each loop in the program. Our `loop-analyzer` -->