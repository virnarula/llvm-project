#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"

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

using namespace llvm;

static cl::OptionCategory SizeDiffCategory("llvm-remark-size-diff options");
static cl::opt<std::string> InputFileName(cl::Positional, cl::Required,
                                           cl::cat(SizeDiffCategory),
                                           cl::desc("remarks_file"));

static cl::opt<std::string> OutputFileName(cl::Positional, cl::Required,
                                           cl::cat(SizeDiffCategory),
                                           cl::desc("output remark file"));

static int Extracted, NotExtracted, NotSimplified, Dumps, NumVectorized;
static bool Vectorized;

class MyDiagnosticHandler final : public DiagnosticHandler {
public:
  MyDiagnosticHandler() { VectorizePassName = "LoopVectorize"; }
  
  bool handleDiagnostics(const DiagnosticInfo &DI) override {
    std::string Msg; raw_string_ostream RSO(Msg);
    DiagnosticPrinterRawOStream DP(RSO);
    DI.print(DP); 
    
    std::string ToMatch("vectorized loop");
    if (Msg.find(ToMatch) != std::string::npos) {
      Vectorized = true;
      NumVectorized++;
    }
    
    return true;
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

Error ProcessRemarks(StringRef InputFileName) {
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
    if (PassName.equals("loop-extract-analysis")) {
      const auto &RemarkName = Remark.RemarkName;
      if (RemarkName.equals("ModuleDump")) { // it just rewrites over the moduledumps
        std::string ModuleString = Remark.getArgsAsMsg();
        std::ofstream out(OutputFileName);
        out << ModuleString;
        out.close();
        Dumps++;
      } else if (RemarkName.equals("Extracted")) {
        Extracted += std::stoi(Remark.getArgsAsMsg());
      } else if (RemarkName.equals("NotExtracted")) {
        NotExtracted += std::stoi(Remark.getArgsAsMsg());
      } else if (RemarkName.equals("NotSimplified")) {
        NotSimplified += std::stoi(Remark.getArgsAsMsg());
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

void printLoopSummaries(Module &M) {
  legacy::FunctionPassManager FPM(&M);

  std::string Msg;
  const Target *T = TargetRegistry::lookupTarget(M.getTargetTriple(), Msg);
  SubtargetFeatures Features;
  Features.getDefaultSubtargetFeatures(Triple(M.getTargetTriple()));
  const TargetMachine *TM = T->createTargetMachine(M.getTargetTriple(), "", Features.getString(), TargetOptions(), Reloc::Static, M.getCodeModel(), CodeGenOpt::Default);

  // construct the target, targetmachine 
  FPM.add(createTargetTransformInfoWrapperPass(TM->getTargetIRAnalysis()));

  FPM.add(createLoopVectorizePass());
  dbgs() << "Loops:";
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
      dbgs() << "Determinable Bounds: false\n";
    } else {
      dbgs() << "Determinable Bounds: true\n";
    }

    Vectorized = false;
    FPM.run(F);
    dbgs() << "Vectorized: ";
    if (Vectorized)
      dbgs() << "true\n";
    else
      dbgs() << "false\n";
  }
}

void printSummaryStats() {
  dbgs() << "\n_______________\n";
  dbgs() << "Summary Stats: \n";
  dbgs() << "Extracted: " << Extracted << "\n";
  dbgs() << "Not able to Extract: " << NotExtracted << "\n";
  dbgs() << "Not in Simplified Form: " << NotSimplified << "\n";
  dbgs() << "\n";
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
  
  Error E = ProcessRemarks(InputFileName);
  
  if (E) {
    errs() << "Error parsing Remarks!\n";
    exit(1);
  }

  printSummaryStats();
  
  LLVMContext Context;
  Context.setDiagnosticHandler(std::make_unique<MyDiagnosticHandler>());
  auto M = getLoopModule(Context);
  printLoopSummaries(*M);
  
  handleAllErrors(std::move(E), [&](const ErrorInfoBase &PE) {
        PE.log(WithColor::error());
        errs() << '\n';
      });
  
  return 0;
}
