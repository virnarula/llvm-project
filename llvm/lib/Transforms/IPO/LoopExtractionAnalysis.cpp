#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Transforms/IPO/LoopExtractionAnalysis.h"

using namespace llvm;

#define DEBUG_TYPE "loop-extract-analysis"

namespace {
  struct LoopExtractionAnalyzer {
    explicit LoopExtractionAnalyzer(
      function_ref<DominatorTree &(Function &)> LookupDomTree,
      function_ref<LoopInfo &(Function &)> LookupLoopInfo,
      function_ref<AssumptionCache *(Function &)> LookupAssumptionCache)
      : LookupDomTree(LookupDomTree),
        LookupLoopInfo(LookupLoopInfo),
        LookupAssumptionCache(LookupAssumptionCache) {}
    
    bool runOnModule(Module &M);

private:
  function_ref<DominatorTree &(Function &)> LookupDomTree;
  function_ref<LoopInfo &(Function &)> LookupLoopInfo;
  function_ref<AssumptionCache *(Function &)> LookupAssumptionCache;
};
} // namespace

PreservedAnalyses LoopExtractionAnalysisPass::run(Module &M, ModuleAnalysisManager &AM) {
  auto &FAM = AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  auto LookupDomTree = [&FAM](Function &F) -> DominatorTree & {
    return FAM.getResult<DominatorTreeAnalysis>(F);
  };
  auto LookupLoopInfo = [&FAM](Function &F) -> LoopInfo & {
    return FAM.getResult<LoopAnalysis>(F);
  };
  auto LookupAssumptionCache = [&FAM](Function &F) -> AssumptionCache * {
    return FAM.getCachedResult<AssumptionAnalysis>(F);
  };
  
  LoopExtractionAnalyzer(LookupDomTree, LookupLoopInfo, LookupAssumptionCache).runOnModule(M);

  return PreservedAnalyses::all();
}

bool LoopExtractionAnalyzer::runOnModule(Module &M) {
  return true;
}