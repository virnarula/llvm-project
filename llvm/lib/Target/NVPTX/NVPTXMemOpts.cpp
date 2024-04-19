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
    
    std::vector<IndexType> isLoadingFromArray(LoadInst *LI);

    Module *M;
    };
  };

char NVPTXMemOpts::ID = 0;
std::string NVPTXMemOpts::SYNC_THREADS_INTRINSIC_NAME = "llvm.nvvm.barrier0";
std::string NVPTXMemOpts::NVVM_READ_SREG_INTRINSIC_NAME = "llvm.nvvm.read.ptx.sreg";

// A common pattern to calculate the abosolute index of a thread is:
// idx = tid + ctaid * ntid
// This function will check if the index is calculated in this way
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

// Return dimension's indexes for an array load instruction
// return 0 if the value is not an array
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
  
  // TODO:: if more than one index, it is probably coalesced already
  // if (++idx != GEP->idx_end()) {
  //   return indexValues;
  // }

  getIndexValues(GEP, indexValues);
  return indexValues;
}

int isStoringToArray(StoreInst *SI) {
  assert(SI && "SI is null");
  auto GEP = dyn_cast<GetElementPtrInst>(SI->getPointerOperand());
  if (!GEP) { return 0; }

  return 0;
}

// Check if the index is a constant
bool isIndexConstant(Value *idx) {
  return isa<ConstantInt>(idx);
}

// Check if the index is a thread constant.
// ie. the thread id. this is not a constant for all threads in a warp
bool isIndexThreadConstant(Value *idx) {
  return false;
}


bool NVPTXMemOpts::isCallCoalescable(LoadInst *LI, std::vector<IndexType> &indexValues) {
  // Check if the call is already coalesced
  // We can do this by seeing if the call is already a load from shared memory
  // If it is, we can skip this call
  auto GEP = dyn_cast<GetElementPtrInst>(LI->getPointerOperand());
  assert(GEP && "GEP is null");
  auto ptr = GEP->getPointerOperand();
  auto ptrGEP = dyn_cast<GetElementPtrInst>(ptr);
  assert(!ptrGEP && "Nested GEP not supported");

  // check if the loaded float is being used by a store into shared memory (addressspace 3)
  // if it is, we can skip this call

  // check if the gep is loading from global memory
  if (GEP->getPointerOperand()->getType()->getPointerAddressSpace() != 1) {
    return false;
  }

  auto storeInst = dyn_cast<StoreInst>(LI->user_back());
  if (storeInst && storeInst->getPointerAddressSpace() == 3) {
    return false;
  }

  // TODO:: otherwise, for now, we will assume that the call is coalescable
  return true;
}

/*
This function will coalesce memory calls.
Example:

  %0 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %1 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %mul = mul i32 %0, %1
  %2 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %add = add i32 %mul, %2
  %3 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
  %4 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %mul5 = mul i32 %3, %4
  %5 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %add7 = add i32 %mul5, %5
  %idxprom = sext i32 %add7 to i64
  %arrayidx = getelementptr inbounds ptr, ptr %A2, i64 %idxprom
  %6 = load ptr, ptr %arrayidx, align 8
  %idxprom8 = sext i32 %i.0 to i64
  %arrayidx9 = getelementptr inbounds float, ptr %6, i64 %idxprom8
  %7 = load float, ptr %arrayidx9, align 4

Will give the following parameters:
  LI = %6
  indexValues = { %add7, %add5, %add, %mul, %add7, %add, %mul }

Will be coalesced to:




*/

/*
Rules regarding coalescing:
- if the index is a constant for all threads in a warp, it cannot be coalesced
- if the index is a constant for one thread but contiguous across a warp, it can be coalesced 
- if the index is a loop induction variable, it can be coalesced

Other memory accesses will be ignored for now
*/
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
  // LI->eraseFromParent();

}

bool NVPTXMemOpts::runOnFunction(Function &F) {
  M = F.getParent();

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
      if (auto *SI = dyn_cast<StoreInst>(&*I)) {
        if (isStoringToArray(SI) > 0) {
          errs() << "Found a store instruction: " << *SI << "\n";
        }
      }
    }
    for (auto LI : toDelete) {
      // asser that the LI has no uses
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