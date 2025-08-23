# Loop Analyzer

Analyze extracted loops from LLVM optimization remarks and summarize properties like known bounds, number of exiting blocks, estimated hotness, and whether loop vectorization succeeded.

## What this is
- A new LLVM module pass `loop-extract-analysis` that clones a module, extracts loops (using `CodeExtractor`), and emits a single optimization remark containing the IR of the extracted loop functions plus extraction stats.
- A standalone tool `loop-analyzer` that reads bitstream optimization remarks, reconstructs the extracted loop module, runs lightweight analyses, and prints per-loop details and a summary.

The included tests under `llvm/test/tools/loop-analyzer` demonstrate expected output.

## Build
Use CMake + Ninja from the monorepo root. To build just the tool and its deps for X86:
```bash
mkdir -p build
cmake -S llvm -B build -G Ninja \
  -DLLVM_TARGETS_TO_BUILD=X86 \
  -DLLVM_BUILD_TOOLS=ON \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=Release
ninja -C build loop-analyzer
```

If you also want `opt` to generate remarks locally:
```bash
ninja -C build opt
```

The `loop-analyzer` binary will be in `build/bin/`.

## Generate remarks (Linux, recommended)
You can run the pass with `opt` and ask it to emit bitstream remarks.

1) Start from LLVM IR (either hand-written or produced by clang):
```bash
clang -O2 -g -S -emit-llvm sample.c -o sample.ll
```

2) Run the loop extraction pass and emit bitstream remarks:
```bash
build/bin/opt -passes=loop-extract-analysis \
  -pass-remarks-format=bitstream \
  -pass-remarks-output=sample.remarks \
  -disable-output \
  sample.ll
```

Notes:
- Hotness is estimated using `BlockFrequencyInfo`. If PGO is available it can improve estimates, but it is not required.
- The pass is registered under the name `loop-extract-analysis` (see `PassRegistry.def`).

## Generate remarks (macOS alternative)
If you prefer to use clang and dsymutil like traditional diagnostics:
```bash
# Optional: build with PGO to improve hotness, otherwise omit these steps.
clang++ -fprofile-instr-generate -O2 -g sample.cpp -o a.out && \
  LLVM_PROFILE_FILE=a.profraw ./a.out && \
  llvm-profdata merge a.profraw -o a.profdata && \
  clang++ -fprofile-instr-use=a.profdata -O2 -g sample.cpp -o a.profile.out

# Extract embedded remarks (requires dsymutil on macOS)
dsymutil a.profile.out
# Remarks will live under: a.profile.out.dSYM/Contents/Resources/Remarks/
```
You can then point `loop-analyzer` to the produced remark file.

## Run loop-analyzer
`loop-analyzer` reads a bitstream remarks file and prints analysis to stderr. It also enters an interactive query mode; to run non-interactively, close stdin.
```bash
# Non-interactive run (stdin closed) and capture output to a file
build/bin/loop-analyzer sample.remarks /dev/null </dev/null 2> out.txt
```
- Arg1: path to the remarks file (bitstream format)
- Arg2: an unused output path (historical, pass `/dev/null`)

Example outputs are in the repo:
- `llvm/test/tools/loop-analyzer/Outputs/basic.result.txt`
- `llvm/test/tools/loop-analyzer/Outputs/complex.result.txt`

They show per-loop sections containing:
- Name
- IR dump of the extracted loop function
- `# Exiting Nodes`
- `Known Bounds` (from ScalarEvolution backedge count)
- `Vectorized` (derived from LoopVectorize diagnostics)

Followed by a short summary with counts and totals.

## Known limitations and rough edges
- Naming mismatch: the pass currently emits a remark named `NotExtractable` but the tool looks for `NotExtracted`; summary counts may under-report failures. Easy fix is to align names or accept both in the tool.
- Vectorization status is derived from diagnostics and may retain the previous state if no remark is emitted for a function; reset per-function before running the vectorizer to avoid stale results.
- `Contained` loops are tracked internally but never incremented; the reported count is always 0.
- Interactive mode has no flag to disable; use `</dev/null` for batch usage. Consider adding a `--no-interactive` or `--once` flag.
- README previously referenced dsymutil/macOS only; Linux/opt usage is the recommended path and documented above.

## Quick demo using the included test inputs
The repo ships remark inputs in `llvm/test/tools/loop-analyzer/Inputs/` that correspond to the outputs in `Outputs/`.
```bash
# Basic
build/bin/loop-analyzer llvm/test/tools/loop-analyzer/Inputs/basic.dwarf /dev/null </dev/null 2> basic.out
# Complex
build/bin/loop-analyzer llvm/test/tools/loop-analyzer/Inputs/complex.dwarf /dev/null </dev/null 2> complex.out
```
Compare `basic.out`/`complex.out` to the expected results under `Outputs/`.

## Next steps to revive/extend
- Align remark names (`NotExtracted` vs `NotExtractable`).
- Reset vectorization result state per function to avoid carry-over.
- Increment and report contained-loop statistics.
- Add a non-interactive flag and optionally JSON output for tooling integration.
- Add a small end-to-end test that runs `opt -passes=loop-extract-analysis` to produce a bitstream remarks file and checks `loop-analyzer` output.