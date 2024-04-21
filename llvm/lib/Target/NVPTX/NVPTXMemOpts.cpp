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
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"

#define DEBUG_TYPE "nvptx-mem-opts"

using namespace llvm;

namespace {
  struct NVPTXMemOpts : public FunctionPass {
    static char ID;

    NVPTXMemOpts() : FunctionPass(ID) {}

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.setPreservesCFG();
      // Added for analysis for prefetching
      AU.addRequired<LoopInfoWrapperPass>();
      AU.addRequired<ScalarEvolutionWrapperPass>();
    }

    bool runOnFunction(Function &F) override;

    StringRef getPassName() const override {
      return "Memory coalescing and prefetching";
    }

    static std::string SYNC_THREADS_INTRINSIC_NAME;
    static std::string NVVM_READ_SREG_INTRINSIC_NAME;

    enum IndexType {
      CONSTANT,
      ABSOLUTE_THREAD_ID,
      LOOP_INDUCTION
    };
  private:

    // Helper functions
    void CoalesceMemCalls(LoadInst *LI, std::vector<IndexType> &indexValues);
    bool isCallCoalescable(LoadInst *LI, std::vector<IndexType> &indexValues);
    bool canPrefetch(LoadInst *LI, LoopInfo &LIInfo, ScalarEvolution &SE);
    void prefetchDataToCache(IRBuilder<> &Builder, Value *Address);
    
    std::vector<IndexType> isLoadingFromArray(LoadInst *LI);

    Module *M;
    };
  };

char NVPTXMemOpts::ID = 0;
std::string NVPTXMemOpts::SYNC_THREADS_INTRINSIC_NAME = "llvm.nvvm.barrier0";
std::string NVPTXMemOpts::NVVM_READ_SREG_INTRINSIC_NAME = "llvm.nvvm.read.ptx.sreg";

// A common pattern to calculate the abosolute index of a thread is:
// idx = tid + ctaid * ntid
// This function will check if an index is calculated in this way
bool isAbsoluteThreadIndex(Value *idx) {
  auto sext = dyn_cast<SExtInst>(idx);
  if (!sext) { return false; }
  
  auto val = sext->getOperand(0);
  auto add = dyn_cast<BinaryOperator>(val);

  if (!add || add->getOpcode() != Instruction::Add) { return false; }

  auto mul = dyn_cast<BinaryOperator>(add->getOperand(0));
  if (!mul || mul->getOpcode() != Instruction::Mul) { return false; }

  auto tid = dyn_cast<CallInst>(mul->getOperand(0));
  auto ntid = dyn_cast<CallInst>(mul->getOperand(1));
  auto ctaid = dyn_cast<CallInst>(add->getOperand(1));

  if (!tid || !ntid || !ctaid) { return false; }

  return true;
}

/*
This function is quite complicated because we are trying to convert
a single GEP instruction into a vector representing the index element.
This will require traversing backwards to find the initial values being used as indexes

TODO: some arrays are two dimensional but represented as a single index. 
We need to handle this case next.
*/
void getIndexValues(GetElementPtrInst *GEP, std::vector<NVPTXMemOpts::IndexType> &indexValues) {
  // get first index value. There should be exactly one
  auto index_value = GEP->idx_begin();
  if (isa<ConstantInt>(index_value)) {
    indexValues.push_back(NVPTXMemOpts::IndexType::CONSTANT);
    return;
  } else if (isAbsoluteThreadIndex(cast<Value>(index_value))) {
      indexValues.push_back(NVPTXMemOpts::IndexType::ABSOLUTE_THREAD_ID);
    return;
  }


  return;
}

// This function will check if the load instruction is loading from an array
// If it is, it will return the index value types used to access the array
// If not, it will return an empty vector
std::vector<NVPTXMemOpts::IndexType> NVPTXMemOpts::isLoadingFromArray(LoadInst *LI) {

  std::vector<NVPTXMemOpts::IndexType> indexValues;
  assert(LI && "LI is null");
  auto GEP = dyn_cast<GetElementPtrInst>(LI->getPointerOperand());
  if (!GEP) { return indexValues; }
  
  auto ptr = GEP->getPointerOperand();
  auto ptrGEP = dyn_cast<GetElementPtrInst>(ptr);
  assert(!ptrGEP && "Nested GEP not supported");

  // get index value. There should be exactly one
  auto idx = GEP->idx_begin();
  assert(idx != GEP->idx_end() && "No index found");

  getIndexValues(GEP, indexValues);
  return indexValues;
}

/*
Rules regarding coalescing:
- if the index is a constant for all threads in a warp, it cannot be coalesced
- if the index is a constant for one thread but contiguous across a warp, it can be coalesced 
- if the index is a loop induction variable, it can be coalesced

Other memory accesses will be ignored for now
*/
bool NVPTXMemOpts::isCallCoalescable(LoadInst *LI, std::vector<IndexType> &indexValues) {
  auto GEP = dyn_cast<GetElementPtrInst>(LI->getPointerOperand());
  assert(GEP && "GEP is null");
  auto ptr = GEP->getPointerOperand();
  auto ptrGEP = dyn_cast<GetElementPtrInst>(ptr);
  assert(!ptrGEP && "Nested GEP not supported");

  // We only consider loads from global memory. Filters out already coalesced loads
  if (GEP->getPointerOperand()->getType()->getPointerAddressSpace() != 1) {
    return false;
  }

  // If the load is being stored to shared memory, it cannot be coalesced
  // It is probably already coalesced
  auto storeInst = dyn_cast<StoreInst>(LI->user_back());
  if (storeInst && storeInst->getPointerAddressSpace() == 3) {
    return false;
  }

  // TODO:: there will be other considerations
  // otherwise, we assume  the call is coalescable
  return true;
}

void NVPTXMemOpts::CoalesceMemCalls(LoadInst *LI, std::vector<IndexType> &indexValues) {
  assert (LI && "LI is null");
  assert (indexValues.size() > 0 && "indexValues is empty");
  auto GEP = dyn_cast<GetElementPtrInst>(LI->getPointerOperand());
  assert (GEP && "GEP is null");

  // First, we need to create a shared memory buffer to store the data
  // We will use the same type as the original array
  auto arrayType = GEP->getSourceElementType();
  // TODO:: for now, we are assuming type is int64 or float64;
  // we need 8 bytes per element, 64 bytes per warp
  int arraySize = 16;
  // Create the array type
  auto sharedArrayType = ArrayType::get(arrayType, arraySize);
  auto arrayInitVal = UndefValue::get(sharedArrayType); // TODO:: see why this is not working as the array initializer below
  auto sharedArray = new GlobalVariable(*M, sharedArrayType, 
    false, GlobalValue::InternalLinkage, 
    arrayInitVal, "sharedArray", nullptr, 
    GlobalValue::NotThreadLocal, 3, false);
  sharedArray->setAlignment(MaybeAlign(4));

  IRBuilder<> Builder(GEP->getNextNode());
  // First, we need to load the value from the original array.
  // This will be loaded into shared memory.
  auto LoadInst = Builder.CreateLoad(GEP->getSourceElementType(), GEP);
  // Next, we need to calculate the index for the shared memory array
  // The original index is the absolute thread id. we need to convert this to tid
  // first, get the thread id. Find the instrinsic call that is already in the function
  auto TidInstrinsic = Intrinsic::getDeclaration(M, Intrinsic::nvvm_read_ptx_sreg_tid_x);
  // get the register that reads the thread id
  auto TidVal = Builder.CreateCall(TidInstrinsic, {});

  // Next, we need to calculate the index for the shared memory array
  // first, zero extend the tid value
  auto TidZeroExt = Builder.CreateZExt(TidVal, Type::getInt64Ty(M->getContext()));
  // Next, create a GEP to calculate the index of shared memory
  auto ZeroVal = ConstantInt::get(Type::getInt64Ty(M->getContext()), 0);
  auto SharedGEP = Builder.CreateGEP(sharedArrayType, sharedArray, std::vector<Value*>{ZeroVal, TidZeroExt});

  // store the value from the original array to the shared memory array
  Builder.CreateStore(LoadInst, SharedGEP);

  // We need to insert __syncthreads() before the load instruction
  // This is to ensure that all threads have written to shared memory before we read from it
  auto syncThreads = Intrinsic::getDeclaration(M, Intrinsic::nvvm_barrier0);
  Builder.CreateCall(syncThreads, {});

  // Finally, replace the load location with the shared memory location
  Builder.SetInsertPoint(LI);
  auto SharedLoad = Builder.CreateLoad(GEP->getSourceElementType(), SharedGEP);
  LI->replaceAllUsesWith(SharedLoad);

}

// Logic to decide whether to prefetch or not
bool NVPTXMemOpts::canPrefetch(LoadInst *LI, LoopInfo &LIInfo, ScalarEvolution &SE) {
    return false;
}

// Prefetcing logic
void NVPTXMemOpts::prefetchDataToCache(IRBuilder<> &Builder, Value *Address) {
}

bool NVPTXMemOpts::runOnFunction(Function &F) {
  M = F.getParent();
  //Analysis for prefetching: 
  LoopInfo &LIInfo = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  ScalarEvolution &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();

  errs() << "Hello from NVPTXMemOpts\n";
  std::vector<LoadInst*> toDelete;
  for (auto &BB : F) {
    for (auto I = BB.begin(); I != BB.end(); ++I){
      if (auto *LI = dyn_cast<LoadInst>(&*I)) {
        auto indexValues = isLoadingFromArray(LI);
        if (indexValues.empty()) 
          continue;
        if (isCallCoalescable(LI, indexValues)) {
          errs() << "Found a candidate instruction: " << *LI << "\n";
          CoalesceMemCalls(LI, indexValues);
          toDelete.push_back(LI);
        }
      }
    }
    for (auto LI : toDelete) {
      // assert that the LI has no uses
      assert(LI->use_empty());
      LI->eraseFromParent();
    }
    toDelete.clear();
  }
  return false;
}

// } // end anonymous namespace

namespace llvm {
void initializeNVPTXMemOptsPass(PassRegistry &);
}

INITIALIZE_PASS(NVPTXMemOpts, "nvptx-mem-opts",
                "Memory coalescing and prefetching",
                true, false)

FunctionPass *llvm::createNVPTXMemOptsPass() {
  return new NVPTXMemOpts();
}