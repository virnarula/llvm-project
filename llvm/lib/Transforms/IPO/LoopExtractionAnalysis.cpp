#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"

#include "llvm/IR/Dominators.h"
#include "llvm/IR/LegacyPassManager.h"

#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/CodeExtractor.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Scalar.h"
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
  bool runOnFunction(Function &F);

  Function *ExtractLoop(Loop *L, LoopInfo &LI, DominatorTree &DT);
  SmallVector<Function*, 16> ExtractedLoops;

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

Function *LoopExtractionAnalyzer::ExtractLoop(Loop *L, LoopInfo &LI, DominatorTree &DT) {
  Function &Func = *L->getHeader()->getParent();
  AssumptionCache *AC = LookupAssumptionCache(Func);
  CodeExtractorAnalysisCache CEAC(Func); 
  CodeExtractor Extractor(DT, *L, false, nullptr, nullptr, AC);

  if (Function *ExtractionLoop = Extractor.extractCodeRegion(CEAC)) {
    ExtractedLoops.push_back(ExtractionLoop);
    LI.erase(L);
  }

  return nullptr;
}

bool LoopExtractionAnalyzer::runOnFunction(Function &F) {
  if (F.empty())
    return false;

  LoopInfo &LI = LookupLoopInfo(F);
  DominatorTree &DT = LookupDomTree(F);
  
  if (LI.empty()) {
    return false;
  }

  SmallVector<Loop*, 8> Loops;
  Loops.assign(LI.begin(), LI.end());
  for (Loop *L : Loops) {
    if (L->isLoopSimplifyForm()) {
      ExtractLoop(L, LI, DT);
    }
  }

  return false;
}

bool LoopExtractionAnalyzer::runOnModule(Module &M) {
  std::unique_ptr<Module> ClonedModPtr = CloneModule(M);

  auto End = --ClonedModPtr->end();
  for (auto Iter = ClonedModPtr->begin(); ; ++Iter) {
    runOnFunction(*Iter);
    if (Iter == End) {
      break;
    }
  }
  
  if (ExtractedLoops.empty()) {
    return false;
  }

  SmallVector<SmallVector<BasicBlock*, 16>, 4> GroupOfBBs;
  for (Function *Func : ExtractedLoops) {
    SmallVector<BasicBlock*, 16> BBs;
    for (auto &BB : *Func) {
      BBs.push_back(&BB);
    }
    GroupOfBBs.push_back(BBs);
  }

  legacy::PassManager PM;
  PM.add(createBlockExtractorPass(GroupOfBBs, true));
  PM.add(createStripDeadPrototypesPass());
  PM.run(*ClonedModPtr);
  
  Function *Extracted = *ExtractedLoops.begin();
  OptimizationRemarkEmitter ORE(Extracted);
  ORE.emit([&]() {
    std::string str; raw_string_ostream rso(str);
    ClonedModPtr->print(rso, nullptr);
    
    auto DebugLoc = Extracted->getEntryBlock().getFirstNonPHI()->getDebugLoc();
    return OptimizationRemarkAnalysis(DEBUG_TYPE, "ModuleDump", DebugLoc, &Extracted->getEntryBlock())
    << str;

  });
  
  return false;
}
