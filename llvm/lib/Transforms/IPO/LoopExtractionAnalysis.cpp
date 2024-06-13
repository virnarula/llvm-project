//===- LoopExtractionAnalysis.cpp - Loop Extraction Analysis ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass extracts loops into remarks for future analysis
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"

#include "llvm/Bitcode/BitcodeWriter.h"

#include "llvm/IR/Dominators.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/PassManager.h"

#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/CodeExtractor.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"

#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/StripDeadPrototypes.h"
#include "llvm/Transforms/IPO/LoopExtractionAnalysis.h"

#include "llvm/ADT/SmallSet.h"
#include <set>

using namespace llvm;

#define DEBUG_TYPE "loop-extract-analysis"

namespace {
/// LoopExtractionAnalyzer contains the methods used to extract the loops of a module into remarks for later analysis.
///
/// This pass works in tandem with the loop-analyzer tool, which will automate the analysis of the remarks
/// This pass is responsible for only the extraction of loops and passing additional information such as hotness for later analysis.
///
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
  BranchProbabilityInfo BPI(Func, LI);
  BlockFrequencyInfo BFI(Func, BPI, LI);
  CodeExtractor Extractor(DT, *L, false, &BFI, &BPI, nullptr);

  if (Function *ExtractionLoop = Extractor.extractCodeRegion(CEAC)) {
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
  
  if (LI.empty())
    return false;
  
  TargetLibraryInfoImpl TLII;
  TargetLibraryInfo TLI(TLII);
  AssumptionCache AC(F);
  ScalarEvolution SE(F, TLI, AC, DT, LI);

  SmallVector<Loop*, 8> Loops;
  Loops.assign(LI.begin(), LI.end());
  SmallPtrSet<Loop*, 16> Contained;
  for (Loop *L : Loops) {
    // Check that loop is in simply form and not contained inside another loop
    for (auto Child : L->getSubLoops()) {
      Contained.insert(Child);
    }

    if (simplifyLoop(L, &DT, &LI, &SE, &AC, nullptr, false)) {
      LLVM_DEBUG(dbgs() << "Simplified a loop!\n");
    }
    LLVM_DEBUG(dbgs() << "Loop dump in func " << L->getHeader()->getParent()->getName() << ":\n");
    LLVM_DEBUG(L->dump());

    if (Contained.find(L) == Contained.end()) {
      if (L->isLoopSimplifyForm()) {
        if (Function *ExtractedFunc = ExtractLoop(L, LI, DT)) {
          LLVM_DEBUG(dbgs() << "Loop was extracted\n");
          ExtractedLoops.push_back(ExtractedFunc);
          NumExtracted++;
        } else {
          LLVM_DEBUG(dbgs() << "Loop could not be extracted!!\n");
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
  if (M.empty())
    return false;
  
  std::unique_ptr<Module> ClonedModPtr = CloneModule(M);

  SmallVector<Function*, 16> OriginalFunctions;
  for (auto Iter = ClonedModPtr->begin(); Iter != ClonedModPtr->end(); ++Iter) {
    OriginalFunctions.push_back(&*Iter);
  }

  auto End = OriginalFunctions.end();
  for (auto Iter = OriginalFunctions.begin(); Iter != End; ++Iter) {
    runOnFunction(**Iter);
  }
  
  if (ExtractedLoops.empty()) {
    return false;
  }
  
  std::vector<GlobalValue *> GVs(ExtractedLoops.begin(), ExtractedLoops.end());
  
  // Create remark emitter with hotness information attached
  legacy::PassManager PM;
  PM.add(createGVExtractionPass(GVs));
  PM.add(createStripDeadPrototypesPass());
  PM.run(*ClonedModPtr);

  M.getContext().setDiagnosticsHotnessRequested(true);
  ClonedModPtr->getContext().setDiagnosticsHotnessRequested(true);
  
  /*
  for (auto ExtractedLoop : ExtractedLoops) {
    std::unique_ptr<Module> ExtractedModule = CloneModule(*ClonedModPtr);
    Function *ToExtract = ExtractedModule->getFunction(ExtractedLoop->getName());
    std::vector<GlobalValue*> GVToExtract = { ToExtract };
    
    legacy::PassManager PM;
    PM.add(createGVExtractionPass(GVs));
    PM.add(createStripDeadPrototypesPass());
    PM.run(*ExtractedModule);
    
    assert(!ExtractedModule->empty() && "Extracted Module is empty!");
    
    BasicBlock *LoopBB = ToExtract->getEntryBlock().getSingleSuccessor();
    
    DominatorTree DT;
    LoopInfo LI;
    DT.recalculate(*ToExtract);
    LI.analyze(DT);
    BranchProbabilityInfo BPI(*ToExtract, LI);
    BlockFrequencyInfo BFI(*ToExtract, BPI, LI);
    OptimizationRemarkEmitter ORE(ToExtract, &BFI);
    
    ORE.emit([&]() {
      std::string str; raw_string_ostream rso(str);
      ClonedModPtr->print(rso, nullptr);
      auto DebugLoc = ToExtract->getEntryBlock().getFirstNonPHI()->getDebugLoc();
      return OptimizationRemarkAnalysis(DEBUG_TYPE, "ModuleDump", DebugLoc, LoopBB)
      << str;
    });
  }
  */
  
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
  
  BasicBlock *LoopBB = Extracted->getEntryBlock().getSingleSuccessor();
  
  DominatorTree DT;
  LoopInfo LI;
  DT.recalculate(*Extracted);
  LI.analyze(DT);
  BranchProbabilityInfo BPI(*Extracted, LI);
  BlockFrequencyInfo BFI(*Extracted, BPI, LI);
  OptimizationRemarkEmitter ORE(Extracted, &BFI);

  // Emit the module to remarks
  ORE.emit([&]() {
    std::string str; raw_string_ostream rso(str);
    ClonedModPtr->print(rso, nullptr);
    auto DebugLoc = Extracted->getEntryBlock().getFirstNonPHI()->getDebugLoc();
    return OptimizationRemarkAnalysis(DEBUG_TYPE, "ModuleDump", DebugLoc, LoopBB)
    << str;
  });
  
  // Emit extraction stats
  ORE.emit([&]() {
    auto DebugLoc = Extracted->getEntryBlock().getFirstNonPHI()->getDebugLoc();
    return OptimizationRemarkAnalysis(DEBUG_TYPE, "NotSimplified", DebugLoc, LoopBB)
    << std::to_string(NumNotSimplified);
  });
  
  ORE.emit([&]() {
    auto DebugLoc = Extracted->getEntryBlock().getFirstNonPHI()->getDebugLoc();
    return OptimizationRemarkAnalysis(DEBUG_TYPE, "NotExtractable", DebugLoc, LoopBB)
    << std::to_string(NumNotExtracted);
  });
  
  ORE.emit([&]() {
    auto DebugLoc = Extracted->getEntryBlock().getFirstNonPHI()->getDebugLoc();
    return OptimizationRemarkAnalysis(DEBUG_TYPE, "Extracted", DebugLoc, LoopBB)
    << std::to_string(NumExtracted);
  });

  ORE.emit([&] {
    auto DebugLoc = Extracted->getEntryBlock().getFirstNonPHI()->getDebugLoc();
    return OptimizationRemarkAnalysis(DEBUG_TYPE, "Contained", DebugLoc, LoopBB)
    << std::to_string(NumContained);
  });
  
  return false;
}

