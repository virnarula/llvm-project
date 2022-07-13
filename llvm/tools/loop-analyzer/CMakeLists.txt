set(LLVM_LINK_COMPONENTS
  Core
  IRReader
  Support
  Analysis
  TransformUtils
  Vectorize
  Target
  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos
  AllTargetsDisassemblers
)

add_llvm_tool(loop-analyzer
  loop-analyzer.cpp

  DEPENDS
  intrinsics_gen
  SUPPORT_PLUGINS
  )