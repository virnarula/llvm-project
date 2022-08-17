#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"

#include "llvm/Remarks/Remark.h"
#include "llvm/Remarks/RemarkParser.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/TargetSelect.h"

#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"

#include "llvm/Linker/Linker.h"

#include "llvm/Transforms/Vectorize/LoopVectorize.h"
#include "llvm/Transforms/Vectorize.h"

#include "llvm/MC/TargetRegistry.h"
#include "llvm/MC/SubtargetFeature.h"

#include "llvm/IRReader/IRReader.h"

#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Target/CodeGenCWrappers.h"

#include <iostream>
#include <fstream>
#include <cstdio>
#include <vector>

#include "llvm/Remarks/Remark.h"

using namespace llvm;

static cl::OptionCategory SizeDiffCategory("llvm-remark-size-diff options");
static cl::opt<std::string> InputFileName(cl::Positional, cl::Required,
                                           cl::cat(SizeDiffCategory),
                                           cl::desc("remarks_file"));

static cl::opt<std::string> OutputFileName(cl::Positional, cl::Required,
                                           cl::cat(SizeDiffCategory),
                                           cl::desc("output remark file"));

static int Extracted, NotExtracted, NotSimplified, Contained, Dumps, NumVectorized, NumDeterminableBounds;



// static int OptRemarks, MissedRemarks, FailedRemarks, AnalysisRemarks;

static enum VectorizationResult {
  Success,
  NotProfitable,
  ContainsCallInstruction,
  Other
} VectorizeRes;

// struct to contain all information about a loop
typedef struct LoopDetails {
  Function *F;
  VectorizationResult VecRes;
  bool KnownBounds;
  unsigned ExitNodes;
  unsigned Hotness;
} LoopDetails;

// Summary of vectorization results
typedef struct VectorizationSummary {
  unsigned NumSuccess;
  unsigned NumNotProfitable;
  unsigned NumContainsCallInstruction;
  unsigned NumOther;
  
  VectorizationSummary() :  NumSuccess(0), NumNotProfitable(0),
                            NumContainsCallInstruction(0),
                            NumOther(0) { }
} VectorizationSummary;

static VectorizationSummary VectorizeCounter;

class MyDiagnosticHandler final : public DiagnosticHandler {
public:
  MyDiagnosticHandler() : VectorizePassName("LoopVectorize") { }
  
  bool handleDiagnostics(const DiagnosticInfo &DI) override {
    std::string Msg; raw_string_ostream RSO(Msg);
    DiagnosticPrinterRawOStream DP(RSO);
    DI.print(DP);
    
    const DiagnosticInfoOptimizationBase *DIOB =
      cast<DiagnosticInfoOptimizationBase>(&DI);
    if (DIOB) {
      const Optional<uint64_t> Hotness = DIOB->getHotness();
      if (Hotness)
        dbgs() << "Hotness: " << Hotness.value() << "\n";
    }
    
    std::string ToMatch;
    switch (DI.getKind()) {
      case DK_OptimizationRemarkAnalysis:
        ToMatch = "call instruction cannot be vectorized";
        if (Msg.find(ToMatch) != std::string::npos) {
          VectorizeRes = VectorizationResult::ContainsCallInstruction;
          VectorizeCounter.NumContainsCallInstruction++;
          return true;
        } else {
          VectorizeRes = VectorizationResult::Other;
          VectorizeCounter.NumOther++;
          return true;
        }
        // This is where we would add more checks
        
      case DK_OptimizationRemark:
        ToMatch = "vectorized loop";
        if (Msg.find(ToMatch) != std::string::npos) {
          VectorizeRes = VectorizationResult::Success;
          VectorizeCounter.NumSuccess++;
        }
        return true;
    }
    return false;
  }

  bool isAnalysisRemarkEnabled(StringRef PassName) const override {
    return VectorizePassName.equals(PassName);
  }
  bool isMissedOptRemarkEnabled(StringRef PassName) const override {
    return VectorizePassName.equals(PassName);
  }
  bool isPassedOptRemarkEnabled(StringRef PassName) const override {
    return VectorizePassName.equals(PassName);
  }

  bool isAnyRemarkEnabled() const override {
    return true;
  }
  
private:
  StringRef VectorizePassName;
};


legacy::FunctionPassManager createFPM(Module &M) {
  legacy::FunctionPassManager FPM(&M);

  std::string Msg;
  const Target *T = TargetRegistry::lookupTarget(M.getTargetTriple(), Msg);
  SubtargetFeatures Features;
  Features.getDefaultSubtargetFeatures(Triple(M.getTargetTriple()));

  const TargetMachine *TM = T->createTargetMachine(M.getTargetTriple(), "", Features.getString(), TargetOptions(), Reloc::Static, M.getCodeModel(), CodeGenOpt::Default);

  FPM.add(createTargetTransformInfoWrapperPass(TM->getTargetIRAnalysis()));
  FPM.add(createLoopVectorizePass());

  return FPM;
}

void printLoopSummaries(Module &M) {
  auto FPM = createFPM(M);
  dbgs() << "Loops:\n";
  for (Function &F : M) {
    if (F.empty())
      continue;
    
    DominatorTree DT;
    DT.recalculate(F);
    
    LoopInfo LI;
    LI.analyze(DT);
    
    const Loop *L = *LI.begin();
    
    if (!L) {
      errs() << "Extracted function didn't have a loop?\n";
      exit(1);
    }
    
    dbgs() << "_______________________\n";
    dbgs() << "Name: " << F.getName() << "\n";
    dbgs() << "IR: \n";
    F.dump();
    
    SmallVector<BasicBlock*, 16> ExitingBlocks;
    L->getExitingBlocks(ExitingBlocks);
    dbgs() << "# Exiting Nodes: " << ExitingBlocks.size() << "\n";
    
    
    TargetLibraryInfoImpl TLII;
    TargetLibraryInfo TLI(TLII);
    AssumptionCache AC(F);
    ScalarEvolution SE(F, TLI, AC, DT, LI);
    if (isa<SCEVCouldNotCompute>(SE.getBackedgeTakenCount(L))) {
      dbgs() << "Known Bounds: false\n";
    } else {
      dbgs() << "Known Bounds: true\n";
      NumDeterminableBounds++;
    }
    
    FPM.run(F);
    dbgs() << "Vectorized: ";
    switch (VectorizeRes) {
      case VectorizationResult::Success:
        dbgs() << "true\n";
        break;
        
      case VectorizationResult::ContainsCallInstruction:
        dbgs() << "false: contains call instruction\n";
        break;
        
      default:
        dbgs() << "false\n";
        break;
    }
  }
}

LLVMContext Context;
std::unique_ptr<Module> ModulePointer;
//std::vector<std::unique_ptr<Module>> Modules;
std::vector<LoopDetails> LoopsVec;
//MemoryBuffer *MB;

LoopDetails AnalyzeLoops(Function &F, legacy::FunctionPassManager &FPM) {
  LoopDetails Details;
  Details.F = &F;
  
  DominatorTree DT;
  LoopInfo LI;
  DT.recalculate(F);
  LI.analyze(DT);
  
  const Loop *L = *LI.begin();
  
  if (!L) {
    errs() << "Extracted function didn't have a loop\n";
    exit(1);
  }
  
  // Calculate number of exit blocks
  SmallVector<BasicBlock*, 16> ExitingBlocks;
  L->getExitingBlocks(ExitingBlocks);
  Details.ExitNodes = ExitingBlocks.size();
  
  // Find if loop as known bounds
  TargetLibraryInfoImpl TLII;
  TargetLibraryInfo TLI(TLII);
  AssumptionCache AC(F);
  ScalarEvolution SE(F, TLI, AC, DT, LI);
  Details.KnownBounds = !isa<SCEVCouldNotCompute>(SE.getBackedgeTakenCount(L));
  NumDeterminableBounds += Details.KnownBounds;
  
  // Get Hotness information
  BranchProbabilityInfo BPI(F, LI);
  BlockFrequencyInfo BFI(F, BPI, LI);
  BasicBlock *Incoming, *Backedge;
  if (L->getIncomingAndBackEdge(Incoming, Backedge)) {
    BlockFrequency BlockFreq = BFI.getBlockFreq(Backedge);
    Details.Hotness = BlockFreq.getFrequency();
  }
  
  // Find it it has been vectorized of not
  FPM.run(F);
  Details.VecRes = VectorizeRes;
  
  return Details;
  }

void AnalyzeModules() {
  auto FPM = createFPM(*ModulePointer);
  for (Function &F : *ModulePointer) {
    if (F.empty())
      continue;
    
    LoopDetails Details = AnalyzeLoops(F, FPM);
    LoopsVec.push_back(Details);
  }
  
}

Error ProcessRemarks(StringRef InputFileName) {
  std::remove(OutputFileName.c_str());
  auto Buf = MemoryBuffer::getFile(InputFileName);
  if (auto EC = Buf.getError()) {
    errs() << "Buffer error!\n";
    exit(1);
  }
  
  auto MaybeParser = remarks::createRemarkParserFromMeta(remarks::Format::Bitstream, (*Buf)->getBuffer());
  if (!MaybeParser)
    return MaybeParser.takeError();
  
  auto &Parser = **MaybeParser;
  auto MaybeRemark = Parser.next();
  
  while (MaybeRemark) {
    const remarks::Remark &Remark = **MaybeRemark;
    const auto &PassName = Remark.PassName;
    
    if (auto Hotness = Remark.Hotness) {
      dbgs() << "Remark: " << Remark.getArgsAsMsg() << "\n";
      dbgs() << "Hotness: " << Hotness.value();
    }
    if (PassName.equals("loop-extract-analysis")) {
      const auto &RemarkName = Remark.RemarkName;
      if (RemarkName.equals("ModuleDump")) {
        std::string ModuleString = Remark.getArgsAsMsg();

        auto MB = MemoryBuffer::getMemBuffer(ModuleString);
        SMDiagnostic Err;
        ModulePointer = parseIR(MB->getMemBufferRef(), Err, Context);
        Dumps++;
      } else if (RemarkName.equals("Extracted")) {
        Extracted += std::stoi(Remark.getArgsAsMsg());
      } else if (RemarkName.equals("NotExtracted")) {
        NotExtracted += std::stoi(Remark.getArgsAsMsg());
      } else if (RemarkName.equals("NotSimplified")) {
        NotSimplified += std::stoi(Remark.getArgsAsMsg());
      } else if (RemarkName.equals("Contained")) {
        Contained += std::stoi(Remark.getArgsAsMsg());
      }
      MaybeRemark = Parser.next();
    }
  }
  
  auto E = MaybeRemark.takeError();
  if (!E.isA<remarks::EndOfFileError>())
    return E;

  consumeError(std::move(E));
  
  return Error::success();
}

std::unique_ptr<Module> getLoopModule(LLVMContext &Context) {
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseIRFile(OutputFileName, Err, Context);
  if (!M) {
    errs() << "Unable to parse IR file!\n";
    Err.print("yes", errs());
    exit(1);
  }
  return M;
}

void printFunctionNames() {
  dbgs() << "Functions:\n";
  for (LoopDetails &Details : LoopsVec) {
    dbgs() << Details.F->getName() << "\n";
  }
  dbgs() << "\n";
}

void printSummaryStats() {
  dbgs() << "\n_______________\n";
  dbgs() << "Summary Stats: \n";
  dbgs() << "# of module dumps: " << Dumps << "\n\n";

  // dbgs() << "Not extracted because not in Simplified Form: " << NotSimplified << "\n";
  // dbgs() << "Simplied form but still unable to extract: " << NotExtracted << "\n";
  dbgs() << "Extracted: " << Extracted << "\n";
  dbgs() << "Contained in extracted loops: " << Contained << "\n\n";
  
  dbgs() << "Total known bounds: " << NumDeterminableBounds << "\n";
  dbgs() << "Total Vectorized: " << VectorizeCounter.NumSuccess << "\n";
  dbgs() << "# Cannot vectorize - contains call instruction: " << VectorizeCounter.NumContainsCallInstruction << "\n";
  dbgs() << "\n";
}

std::string VectorizeResToString(VectorizationResult Res) {
  switch (Res) {
    case VectorizationResult::Success:
      return "Success!";
    case VectorizationResult::ContainsCallInstruction:
      return "Failed - contains call instruction";
    case VectorizationResult::NotProfitable:
      return "Failed - not profitable";
    default:
      return "Failed - other";
  }
}

void printLoopSummary(LoopDetails Details) {
  dbgs() << "IR Dump: \n";
  dbgs() << *Details.F << "\n";
  dbgs() << "Name: " << Details.F->getName() << "\n";
  dbgs() << "Known Bounds: " << Details.KnownBounds << "\n";
  dbgs() << "Exit Nodes: " << Details.ExitNodes << "\n";
  dbgs() << "Vectorization Status: " <<
    VectorizeResToString(Details.VecRes) << "\n";
  dbgs() << "Hotness: " << Details.Hotness << "\n";
  
}

void beginQuerying() {
  while (true) {
    dbgs() << ">> ";
    std::string Query;
    std::cin >> Query;
    if (Query.empty()) {
      return;
    }
    Function *F = ModulePointer->getFunction(Query);
    if (!F) {
      dbgs() << "Not a valid function!\n";
      continue;
    }
    
    for (LoopDetails Loop : LoopsVec) {
      if (F == Loop.F) {
        printLoopSummary(Loop);
        break;
      }
    }
  }
}

void sortLoopsByHotness() {
  std::sort(LoopsVec.begin(), LoopsVec.end(), [](LoopDetails Left, LoopDetails Right){
    return Left.Hotness > Right.Hotness;
  });
}

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);

  cl::ParseCommandLineOptions(argc, argv,
                              "Extract module from remark and put it into a file\n");

  InitializeAllTargetInfos();
  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmParsers();
  InitializeAllAsmPrinters();
  InitializeAllDisassemblers();
  
  Context.setDiagnosticHandler(std::make_unique<MyDiagnosticHandler>());
  
  Error E = ProcessRemarks(InputFileName);
  
  if (E) {
    errs() << "Error parsing Remarks!\n";
    exit(1);
  }
  
//  if (Modules.empty()) {
//    errs() << "Could not find any loop remarks!\n";
//    exit(1);
//  }
  
  AnalyzeModules();
  
  sortLoopsByHotness();
  printFunctionNames();
  printSummaryStats();
  
  handleAllErrors(std::move(E), [&](const ErrorInfoBase &PE) {
        PE.log(WithColor::error());
        errs() << '\n';
      });
  
  beginQuerying();
  
  return 0;
}
