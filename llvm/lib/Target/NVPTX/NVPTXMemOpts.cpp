//===- NVPTXMemOpts.cpp - ------------------------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file
// Implements the memory coalescing and prefetching through a co-optimization pass.
//
//===----------------------------------------------------------------------===//

#include "NVPTXMemOpts.h"
#include "NVPTX.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#define DEBUG_TYPE "nvptx-mem-opts"

using namespace llvm;

namespace {
  struct NVPTXMemOpts : public FunctionPass {
    static char ID;

    NVPTXMemOpts() : FunctionPass(ID) {}

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.setPreservesCFG();
    }

    bool runOnFunction(Function &F) override;

    StringRef getPassName() const override {
      return "Memory coalescing and prefetching";
    }
  };

char NVPTXMemOpts::ID = 0;

bool NVPTXMemOpts::runOnFunction(Function &F) {
  return false;
}

} // end anonymous namespace

namespace llvm {
void initializeNVPTXMemOptsPass(PassRegistry &);
}

INITIALIZE_PASS(NVPTXMemOpts, "nvptx-mem-opts",
                "Memory coalescing and prefetching",
                true, false)

FunctionPass *llvm::createNVPTXMemOptsPass() {
  return new NVPTXMemOpts();
}