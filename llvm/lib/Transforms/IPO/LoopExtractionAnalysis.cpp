#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/MemorySSAUpdater.h"

#include "llvm/Bitcode/BitcodeWriter.h"

#include "llvm/IR/Dominators.h"
#include "llvm/IR/LegacyPassManager.h"

#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/CodeExtractor.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"

#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/StripDeadPrototypes.h"
#include "llvm/Transforms/IPO/LoopExtractionAnalysis.h"


#include "llvm/Analysis/AliasAnalysis.h"


#include "llvm/ADT/SmallSet.h"
#include <set>

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
        NumContained(0),
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
    
  int NumContained, NumNotSimplified, NumNotExtracted, NumExtracted;
    
};
} // namespace

PreservedAnalyses LoopExtractionAnalysisPass::run(Module &M, ModuleAnalysisManager &AM) {
  // M.dump();
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
    // LI.erase(L);
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
  
  TargetLibraryInfoImpl TLII;
  TargetLibraryInfo TLI(TLII);
  AssumptionCache AC(F);
  ScalarEvolution SE(F, TLI, AC, DT, LI);

  AliasAnalysis AA(TLI);

  MemorySSA MSSA(F, &AA, &DT);
  MemorySSAUpdater MSSAU(&MSSA);

  SmallVector<Loop*, 8> Loops;
  Loops.assign(LI.begin(), LI.end());
  static std::set<Loop*> Contained;
  for (Loop *L : Loops) {
    // Check that loop is in simply form and not contained inside another loop
    for (auto Child : L->getSubLoops()) {
      Contained.insert(Child);
    }

    if (simplifyLoop(L, &DT, &LI, &SE, &AC, &MSSAU, false)) {
      LLVM_DEBUG(dbgs() << "Simplified a loop!\n");
      // dbgs() << "Simplified a loop!\n";
    }
    LLVM_DEBUG(errs() << "Loop dump in func " << L->getHeader()->getParent()->getName() << ":\n");
    LLVM_DEBUG(L->dump());

    if (Contained.find(L) == Contained.end()) {
      if (L->isLoopSimplifyForm()) {
        if (Function *ExtractedFunc = ExtractLoop(L, LI, DT)) {
          LLVM_DEBUG(errs() << "Loop was extracted\n");
          ExtractedLoops.push_back(ExtractedFunc);
          NumExtracted++;
          // NumContained += L->getSubLoops().size();
        } else {
          LLVM_DEBUG(errs() << "Loop could not be extracted!!\n");
          NumNotExtracted++;
        }
      } else {
        LLVM_DEBUG(errs() << "Loop is not in Loop Simply Form!\n"); NumNotSimplified++;
      }
    } else {
      LLVM_DEBUG(errs() << "Loop is contained in another loop!\n");
    }
  }

  return false;
}

bool LoopExtractionAnalyzer::runOnModule(Module &M) {
//  M.dump();
  std::unique_ptr<Module> ClonedModPtr = CloneModule(M);

  if (M.empty()) 
    return false;

  SmallVector<Function*, 16> OriginalFunctions;
  for (auto Iter = ClonedModPtr->begin(); Iter != ClonedModPtr->end(); ++Iter) {
    OriginalFunctions.push_back(&*Iter);
  }

  auto End = OriginalFunctions.end();
  for (auto Iter = OriginalFunctions.begin(); Iter != End; ++Iter) {
    runOnFunction(**Iter);
    // if (Iter == End) {
    //   break;
    // }
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
//    WriteBitcodeToFile(*ClonedModPtr, rso);
//    dbgs() << str << "\n";
    
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

  ORE.emit([&] {
    auto DebugLoc = Extracted->getEntryBlock().getFirstNonPHI()->getDebugLoc();
    return OptimizationRemarkAnalysis(DEBUG_TYPE, "Contained", DebugLoc, &Extracted->getEntryBlock())
    << std::to_string(NumContained);
  });

  // errs() << "Ran Loop Extraction Analysis\n";
  
  return false;
}
