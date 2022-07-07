set(LLVM_LINK_COMPONENTS
  Core
  IRReader
  Support
  Analysis
  TransformUtils
)

add_llvm_tool(loop-analyzer
  loop-analyzer.cpp

  DEPENDS
  intrinsics_gen
  SUPPORT_PLUGINS
  )