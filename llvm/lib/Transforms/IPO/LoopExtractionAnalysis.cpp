#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"

#include "llvm/IR/Dominators.h"
#include "llvm/IR/LegacyPassManager.h"

#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/CodeExtractor.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/StripDeadPrototypes.h"
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
        LookupAssumptionCache(LookupAssumptionCache),
        NumNotSimplified(0),
        NumNotExtracted(0),
        NumExtracted(0) {}
    
    bool runOnModule(Module &M);

private:
  bool runOnFunction(Function &F);

  Function *ExtractLoop(Loop *L, LoopInfo &LI, DominatorTree &DT);
  SmallVector<Function*, 16> ExtractedLoops;

  function_ref<DominatorTree &(Function &)> LookupDomTree;
  function_ref<LoopInfo &(Function &)> LookupLoopInfo;
  function_ref<AssumptionCache *(Function &)> LookupAssumptionCache;
    
  int NumNotSimplified, NumNotExtracted, NumExtracted;
    
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
  CodeExtractorAnalysisCache CEAC(Func); 
  CodeExtractor Extractor(DT, *L, false, nullptr, nullptr, nullptr);

  if (Function *ExtractionLoop = Extractor.extractCodeRegion(CEAC)) {
    LI.erase(L);
    return ExtractionLoop;
  }

  return nullptr;
}

bool LoopExtractionAnalyzer::runOnFunction(Function &F) {
  if (F.empty())
    return false;

  DominatorTree DT;
  DT.recalculate(F);
  
  LoopInfo LI;
  LI.analyze(DT);
  
  if (LI.empty()) {
    return false;
  }

  SmallVector<Loop*, 8> Loops;
  Loops.assign(LI.begin(), LI.end());
  for (Loop *L : Loops) {
    // Check that loop is in simply form and not contained inside another loop
    if (L->isLoopSimplifyForm() &&
        L->getLoopDepth() == 1) {
      if (Function *ExtractedFunc = ExtractLoop(L, LI, DT)) {
        ExtractedLoops.push_back(ExtractedFunc);
        NumExtracted++;
      } else {
        LLVM_DEBUG(errs() << "Loop could not be extracted!!\n");
        NumNotExtracted++;
      }
    } else {
      LLVM_DEBUG(dbgs() << "Loop is not in Loop Simply Form!\n");
      NumNotSimplified++;
    }
  }

  return false;
}

bool LoopExtractionAnalyzer::runOnModule(Module &M) {
  std::unique_ptr<Module> ClonedModPtr = CloneModule(M);

  if (M.empty()) 
    return false;

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
  
  std::vector<GlobalValue *> GVs(ExtractedLoops.begin(), ExtractedLoops.end());

  legacy::PassManager PM;
  PM.add(createGVExtractionPass(GVs));
  PM.add(createStripDeadPrototypesPass());
  PM.run(*ClonedModPtr);
  
  if (ClonedModPtr->empty()) {
    assert(false && "Nothing after cloning!");
  }
 
  Function *Extracted = nullptr;
  for (Function &Func : *ClonedModPtr) {
    if (!Func.empty()) {
      Extracted = &Func;
      break;
    }
  }
  
  if (Extracted == nullptr) {
    assert(false && "Failed to find extracted func!");
  }
  
  OptimizationRemarkEmitter ORE(Extracted);

  // Emit the module to remarks
  ORE.emit([&]() {
    std::string str; raw_string_ostream rso(str);
    ClonedModPtr->print(rso, nullptr);
    
    auto DebugLoc = Extracted->getEntryBlock().getFirstNonPHI()->getDebugLoc();
    return OptimizationRemarkAnalysis(DEBUG_TYPE, "ModuleDump", DebugLoc, &Extracted->getEntryBlock())
    << str;

  });
  
  // Emit extraction stats
  ORE.emit([&]() {
    auto DebugLoc = Extracted->getEntryBlock().getFirstNonPHI()->getDebugLoc();
    return OptimizationRemarkAnalysis(DEBUG_TYPE, "NotSimplified", DebugLoc, &Extracted->getEntryBlock())
    << std::to_string(NumNotSimplified);
  });
  
  ORE.emit([&]() {
    auto DebugLoc = Extracted->getEntryBlock().getFirstNonPHI()->getDebugLoc();
    return OptimizationRemarkAnalysis(DEBUG_TYPE, "NotExtractable", DebugLoc, &Extracted->getEntryBlock())
    << std::to_string(NumNotExtracted);
  });
  
  ORE.emit([&]() {
    auto DebugLoc = Extracted->getEntryBlock().getFirstNonPHI()->getDebugLoc();
    return OptimizationRemarkAnalysis(DEBUG_TYPE, "Extracted", DebugLoc, &Extracted->getEntryBlock())
    << std::to_string(NumExtracted);
  });

  errs() << "Ran Loop Extraction Analysis\n";
  
  return false;
}
