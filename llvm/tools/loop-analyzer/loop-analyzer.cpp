#include "llvm/Remarks/Remark.h"
#include "llvm/Remarks/RemarkParser.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/WithColor.h"

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
//      errs() << Remark.getArgsAsMsg() << "\n";
      std::string ModuleString = Remark.getArgsAsMsg();
      std::ofstream out(OutputFileName);
      out << ModuleString;
      out.close();
    }
    MaybeRemark = Parser.next();
  }
  
  auto E = MaybeRemark.takeError();
  if (!E.isA<remarks::EndOfFileError>())
    return E;

  consumeError(std::move(E));
  
  return Error::success();
}

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);
  
  cl::ParseCommandLineOptions(argc, argv,
                                "Extract module from remark and put it into a file\n");
  
  Error E = ProcessRemarks(InputFileName);
  
  
  handleAllErrors(std::move(E), [&](const ErrorInfoBase &PE) {
        PE.log(WithColor::error());
        errs() << '\n';
      });
  
  return 0;
}

