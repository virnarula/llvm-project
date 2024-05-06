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
#include "llvm/IR/DataLayout.h"
#include "llvm/Analysis/IVDescriptors.h"

#define DEBUG_TYPE "nvptx-mem-opts"

using namespace llvm;

namespace {
  struct NVPTXMemOpts : public ModulePass {
    static char ID;

    NVPTXMemOpts() : ModulePass(ID) {}

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.setPreservesCFG();
      // Added for analysis for prefetching
      AU.addRequired<LoopInfoWrapperPass>();
      AU.addRequired<ScalarEvolutionWrapperPass>();
    }

    bool runOnModule(Module &M) override;
    bool runOnFunction(Function &F);

    StringRef getPassName() const override {
      return "Memory coalescing and prefetching";
    }

    static std::string SYNC_THREADS_INTRINSIC_NAME;
    static std::string NVVM_READ_SREG_INTRINSIC_NAME;
    static std::string THREAD_BLOCK_SIZE_MARKER_1D;
    static std::string THREAD_BLOCK_SIZE_MARKER_2D;
    static std::string THREAD_BLOCK_SIZE_MARKER_3D;

  private:

    bool CoalesceBasicBlocks(Function &F);
    bool CoalesceLoops(LoopInfo &LIInfo, ScalarEvolution &SE, const DataLayout &DL);
    Instruction *CoalescePattern1(LoadInst *LI, std::vector<std::pair<LoadInst*, GetElementPtrInst*>> &LoadsToReplace);
    void CoalescePattern2(LoadInst *LI, PHINode *IndVar, Loop *L, ScalarEvolution &SE);
    void CoalescePattern3(LoadInst *LI, PHINode *IndVar, Loop *L, ScalarEvolution &SE);

    // Helper functions
    void GetThreadBlockSize(Function &F);
    ArrayType *getSharedArrayType(GetElementPtrInst *GEP);
    bool isCallCoalescable(LoadInst *LI);
    
    std::vector<int> threadBlockSize;
    };
  };

char NVPTXMemOpts::ID = 0;
std::string NVPTXMemOpts::SYNC_THREADS_INTRINSIC_NAME = "llvm.nvvm.barrier0";
std::string NVPTXMemOpts::NVVM_READ_SREG_INTRINSIC_NAME = "llvm.nvvm.read.ptx.sreg";
std::string NVPTXMemOpts::THREAD_BLOCK_SIZE_MARKER_1D = "__tb_size_marker_1D";
std::string NVPTXMemOpts::THREAD_BLOCK_SIZE_MARKER_2D = "__tb_size_marker_2D";
std::string NVPTXMemOpts::THREAD_BLOCK_SIZE_MARKER_3D = "__tb_size_marker_3D";

// Convert a llvm::Value to an int. must be a ConstantInt
int Value2Int(Value *V) {
  auto CI = dyn_cast<ConstantInt>(V);
  assert(CI && "Value is not a ConstantInt");
  return CI->getZExtValue();
}

// ================================================================
// ================== Pattern matching functions ==================
// ================================================================

// A common pattern to calculate the abosolute index of a thread is:
// idx = tid + ctaid * ntid
// This function will check if a value is calculated in this way
// isX is true if the value is threadIdx.x. Otherwise, it is threadIdx.y or threadIdx.z
bool isAbsoluteThreadIdxHelper(Instruction *I, bool isX = true) {
  auto add = dyn_cast<BinaryOperator>(I);
  if (!add || add->getOpcode() != Instruction::Add) { return false; }

  auto mul = dyn_cast<BinaryOperator>(add->getOperand(0));
  if (!mul || mul->getOpcode() != Instruction::Mul) { return false; }

  // The operands can be in any order. This will assign them correctly.
  CallInst *ctaid = nullptr;
  CallInst *ntid = nullptr;
  CallInst *LHS = dyn_cast<CallInst>(mul->getOperand(0));
  CallInst *RHS = dyn_cast<CallInst>(mul->getOperand(1));
  if (!LHS || !RHS) { return false; }
  if (LHS->getCalledFunction()->getName().starts_with(NVPTXMemOpts::NVVM_READ_SREG_INTRINSIC_NAME + ".ntid.")) {
    ntid = dyn_cast<CallInst>(mul->getOperand(0));
    ctaid = dyn_cast<CallInst>(mul->getOperand(1));
  } else if (RHS->getCalledFunction()->getName().starts_with(NVPTXMemOpts::NVVM_READ_SREG_INTRINSIC_NAME + ".ntid.")) {
    ntid = dyn_cast<CallInst>(mul->getOperand(1));
    ctaid = dyn_cast<CallInst>(mul->getOperand(0));
  } else {
    return false;
  }

  auto tid = dyn_cast<CallInst>(add->getOperand(1));

  if (!tid || !ntid || !ctaid) { return false; }

  // Check that the operands are the correct intrinsics
  if (!tid->getCalledFunction()->getName().starts_with(NVPTXMemOpts::NVVM_READ_SREG_INTRINSIC_NAME) ||
      !ntid->getCalledFunction()->getName().starts_with(NVPTXMemOpts::NVVM_READ_SREG_INTRINSIC_NAME) ||
      !ctaid->getCalledFunction()->getName().starts_with(NVPTXMemOpts::NVVM_READ_SREG_INTRINSIC_NAME)) { 
        return false; 
  }

  if (isX && !(tid->getCalledFunction()->getName().ends_with(".tid.x") &&
               ntid->getCalledFunction()->getName().ends_with(".ntid.x") &&
               ctaid->getCalledFunction()->getName().ends_with(".ctaid.x"))) {
    return false;
  } else if (!isX && !(tid->getCalledFunction()->getName().ends_with(".tid.y") &&
               ntid->getCalledFunction()->getName().ends_with(".ntid.y") &&
               ctaid->getCalledFunction()->getName().ends_with(".ctaid.y") ||
               tid->getCalledFunction()->getName().ends_with(".tid.z") &&
               ntid->getCalledFunction()->getName().ends_with(".ntid.z") &&
               ctaid->getCalledFunction()->getName().ends_with(".ctaid.z"))){
    return false;
  }

  return true;
}

// Returns true if a value is a constant for all threads in a half warp
// If this is true, all inputs used to calculate the value must be either:
//   - Constant
//   - Kernel Argument
//   - Predefined constants not dependent on threadIdx.x
//   - A UnaryOp or BinaryOp with inputs as above
// Recursion is expensive but necessary for flexibility
bool isConstantForHalfWarp(Value *V) {
  if (auto *CI = dyn_cast<ConstantInt>(V)) {
    return true;
  } else if (auto *Arg = dyn_cast<Argument>(V)) {
    return true;  // if it is a function argument, it must be constant for half warp
  }
  
  // V must be an instruction then
  Instruction *I = dyn_cast<Instruction>(V);
  if (!I) { 
    return false;
  } else if (isAbsoluteThreadIdxHelper(I, false)) {
    return true;
  } else if (auto *UI = dyn_cast<UnaryOperator>(V)){
    return isConstantForHalfWarp(UI->getOperand(0));
  } else if (auto *BI = dyn_cast<BinaryOperator>(V)) {
    return isConstantForHalfWarp(BI->getOperand(0)) && isConstantForHalfWarp(BI->getOperand(1));
  }
  
  return false;
}

/*
  Returns true if the load instruction is of the form:
  <Constant for half-warp> + idx
*/
bool PatternMatch1(LoadInst *LI) {
  auto *GEP = dyn_cast<GetElementPtrInst>(LI->getPointerOperand());
  if (!GEP) { return false; }
  auto *SextIndex = dyn_cast<SExtInst>(GEP->getOperand(1));
  if (!SextIndex) { return false; }
  auto *Index = dyn_cast<BinaryOperator>(SextIndex->getOperand(0));
  if (!Index) { return false; }
  if (isAbsoluteThreadIdxHelper(Index)) { return true; }

  if (Index->getOpcode() != Instruction::Add) { return false; }

  auto *Index1 = dyn_cast<Instruction>(Index->getOperand(0));
  auto *Index2 = dyn_cast<Instruction>(Index->getOperand(1));
  if (!Index1 || !Index2) { return false; }
  if (isConstantForHalfWarp(Index1)) {
    return isAbsoluteThreadIdxHelper(Index2);
  } else if (isConstantForHalfWarp(Index2)) {
    return isAbsoluteThreadIdxHelper(Index1);
  }
  return false;
}

/*
  Returns true if the load instruction is of the form:
  <Constant for half-warp> + <loop induction variable>
*/
bool PatternMatch2(LoadInst *LI, PHINode *IndVar) {
  auto *GEP = dyn_cast<GetElementPtrInst>(LI->getPointerOperand());
  if (!GEP) { return false; }
  auto *SextIndex = dyn_cast<SExtInst>(GEP->getOperand(1));
  if (!SextIndex) { return false; }
  auto *Index = dyn_cast<BinaryOperator>(SextIndex->getOperand(0));
  if (!Index) { return false; }
  if (Index->getOpcode() != Instruction::Add) { return false; }
  auto *Index1 = dyn_cast<Instruction>(Index->getOperand(0));
  auto *Index2 = dyn_cast<Instruction>(Index->getOperand(1));
  if (!Index1 || !Index2) { return false; }
  if (Index1 != IndVar && Index2 != IndVar) { return false; }
  // one of the operands is the loop induction variable
  // The other operand should be a constant for the half-warp
  auto *LHS = Index1 == IndVar ? Index2 : Index1;
  return isConstantForHalfWarp(LHS);
}

/*
  Returns true if the load instruction is of the form:
  <Constant for half-warp> + <K> + idx + <loop induction variable>
*/
bool PatternMatch3(LoadInst *LI, PHINode *IndVar){
  auto *GEP = dyn_cast<GetElementPtrInst>(LI->getPointerOperand());
  if (!GEP) { return false; }
  auto *SextIndex = dyn_cast<SExtInst>(GEP->getOperand(1));
  if (!SextIndex) { return false; }
  auto *Index = dyn_cast<BinaryOperator>(SextIndex->getOperand(0));
  if (!Index) { return false; }
  if (Index->getOpcode() != Instruction::Add) { return false; }
  auto *Index1 = dyn_cast<Instruction>(Index->getOperand(0));
  auto *Index2 = dyn_cast<Instruction>(Index->getOperand(1));
  if (!Index1 || !Index2) { return false; }
  if (Index1 != IndVar && Index2 != IndVar) { return false; }
  
  // one of the operands is the loop induction variable
  // The other operand should be a constant for the half-warp
  auto *ConsantPlusThreadIdx = Index1 == IndVar ? Index2 : Index1;

  // the none loop induction variable should be:
  // <Constant for half-warp> + <K> + threadIdx.x
  auto *Add = dyn_cast<BinaryOperator>(ConsantPlusThreadIdx);
  if (!Add || Add->getOpcode() != Instruction::Add) { return false; }
  BinaryOperator *KTimesThreadIdx = nullptr;
  if (isConstantForHalfWarp(Add->getOperand(0))) {
    KTimesThreadIdx = dyn_cast<BinaryOperator>(Add->getOperand(1));
  } else if (isConstantForHalfWarp(Add->getOperand(1))) {
    KTimesThreadIdx = dyn_cast<BinaryOperator>(Add->getOperand(0));
  }

  // KTimesThreadIdx should be of the form K * threadIdx.x
  if (!KTimesThreadIdx || KTimesThreadIdx->getOpcode() != Instruction::Mul) { return false; }
  auto *K = dyn_cast<Instruction>(KTimesThreadIdx->getOperand(0));
  auto *ThreadIdx = dyn_cast<Instruction>(KTimesThreadIdx->getOperand(1));
  if (!K || !ThreadIdx) { return false; }
  if (!isConstantForHalfWarp(K) || !isAbsoluteThreadIdxHelper(ThreadIdx)) { return false; }

  return true;
}


// ================================================================
// ============= Transformation Helper Functions ==================
// ================================================================

/*
Checks if the load satisfies the following conditions for coalescing:
- The load is from global memory
- The load is not already coalesced (i.e. stored to shared memory)
*/
bool NVPTXMemOpts::isCallCoalescable(LoadInst *LI) {
  auto GEP = dyn_cast<GetElementPtrInst>(LI->getPointerOperand());
  assert(GEP && "GEP is null");
  auto ptr = GEP->getPointerOperand();
  auto ptrGEP = dyn_cast<GetElementPtrInst>(ptr);
  assert(!ptrGEP && "Nested GEP not supported");

  // We only consider loads from global memory. Filters out already coalesced loads
  if (GEP->getPointerOperand()->getType()->getPointerAddressSpace() != 1) {
    return false;
  }

  // If the load is being stored to shared memory, its probably already coalesced
  auto storeInst = dyn_cast<StoreInst>(LI->user_back());
  if (storeInst && storeInst->getPointerAddressSpace() == 3) {
    return false;
  }

  // otherwise, we assume  the call is coalescable
  return true;
}

// This will find the threadIdx intrinsic for the given dimension
// 0 = threadIdx.x, 1 = threadIdx.y, 2 = threadIdx.z
// If the intrinsic is not found, it will return nullptr
CallInst *getThreadIdx(Function *F, int dim) {
  std::string suffix;
  if (dim == 0) {
    suffix = ".tid.x";
  } else if (dim == 1) {
    suffix = ".tid.y";
  } else if (dim == 2) {
    suffix = ".tid.z";
  } else {
    assert(false && "Invalid dimension");
  }

  for (auto &BB : *F) {
    for (auto &I : BB) {
      if (auto *CI = dyn_cast<CallInst>(&I)) {
        if (CI->getCalledFunction()->getName().starts_with(NVPTXMemOpts::NVVM_READ_SREG_INTRINSIC_NAME) &&
            CI->getCalledFunction()->getName().ends_with(suffix)) {
          return CI;
        }
      }
    }
  }
  return nullptr;
}

// Get the thread block size from the marker functions
// Remove the marker function calls at the end
void NVPTXMemOpts::GetThreadBlockSize(Function &F) {
  auto M = F.getParent();
  Function *F1 = M->getFunction(NVPTXMemOpts::THREAD_BLOCK_SIZE_MARKER_1D);
  Function *F2 = M->getFunction(NVPTXMemOpts::THREAD_BLOCK_SIZE_MARKER_2D);
  Function *F3 = M->getFunction(NVPTXMemOpts::THREAD_BLOCK_SIZE_MARKER_3D);
  if (!F1 && !F2 && !F3) {
    assert(false && "Thread block size marker not found");
  }
  std::vector<CallInst*> toDelete;
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (auto *CI = dyn_cast<CallInst>(&I)) {
        if (CI->getCalledFunction() == F1) {
          // get the thread block size
          threadBlockSize.push_back(Value2Int(CI->getArgOperand(0)));
          toDelete.push_back(CI);
        }
        else if (CI->getCalledFunction() == F2) {
          // get the thread block size
          threadBlockSize.push_back(Value2Int(CI->getArgOperand(0)));
          threadBlockSize.push_back(Value2Int(CI->getArgOperand(1)));
          toDelete.push_back(CI);
        }
        else if (CI->getCalledFunction() == F3) {
          // get the thread block size
          threadBlockSize.push_back(Value2Int(CI->getArgOperand(0)));
          threadBlockSize.push_back(Value2Int(CI->getArgOperand(1)));
          threadBlockSize.push_back(Value2Int(CI->getArgOperand(2)));
          toDelete.push_back(CI);
        }
      }
    }
  }
  for (auto CI : toDelete) {
    CI->eraseFromParent();
  }
}

// Returns the shared array type for the given GEP
// The array size is the thread block size
ArrayType *NVPTXMemOpts::getSharedArrayType(GetElementPtrInst *GEP) {
  auto arrayType = GEP->getSourceElementType();
  auto arr_1D = ArrayType::get(arrayType, threadBlockSize[0]);
  if (threadBlockSize.size() == 1) {
    return arr_1D;
  }
  auto arr_2D = ArrayType::get(arr_1D, threadBlockSize[1]);
  if (threadBlockSize.size() == 2) {
    return arr_2D;
  }
  auto arr_3D = ArrayType::get(arr_2D, threadBlockSize[2]);
  return arr_3D;
}

// ================================================================
// =================== Transformation Functions ===================
// ================================================================

// Helper function to insert a memory barrier
void InsertBarrier(Instruction *InsertionPoint) {
  auto *F = InsertionPoint->getParent()->getParent();
  auto *M = F->getParent();
  auto *syncThreads = Intrinsic::getDeclaration(M, Intrinsic::nvvm_barrier0);
  IRBuilder<> Builder(InsertionPoint);
  Builder.CreateCall(syncThreads, {});
}

// Creates a shared memory array and returns the handle to it
GlobalVariable *CreateSharedArray(Type *SharedArrayType, LoadInst *LI) {
  auto FuncName = LI->getParent()->getParent()->getName().str() + "input_shared";
  auto arrayInitVal = UndefValue::get(SharedArrayType);
  auto M = LI->getParent()->getParent()->getParent();
  auto sharedArray = new GlobalVariable(*M, SharedArrayType, 
    false, GlobalValue::InternalLinkage, 
    arrayInitVal, FuncName, nullptr, 
    GlobalValue::NotThreadLocal, 3, false);
  sharedArray->setAlignment(MaybeAlign(4));
  return sharedArray;
}

// Returns the new base address for the shared memory array (without threadIdx.x)
Value *LoadStoreToSharedMemory(BasicBlock *SharedMemoryBlock, LoadInst *LI, GetElementPtrInst *GEP, 
                             GlobalVariable *sharedArray, Type *sharedArrayType, Value *ConstantForHW, 
                             Value *NewIndVarPHI, IRBuilder<> &Builder) {
  auto Func = LI->getParent()->getParent();
  auto M = Func->getParent(); 

  Builder.SetInsertPoint(SharedMemoryBlock->getTerminator());
  auto NewBaseAddress = Builder.CreateAdd(ConstantForHW, NewIndVarPHI, "new_add");
  // add this with threadIdx.x
  auto NewBaseAddressThreadIdx = Builder.CreateAdd(NewBaseAddress, getThreadIdx(Func, 0), "new_add_threadidx");
  auto NewSext = Builder.CreateSExt(NewBaseAddressThreadIdx, Type::getInt64Ty(M->getContext()));
  auto ZeroVal = ConstantInt::get(Type::getInt64Ty(M->getContext()), 0);
  auto NewGEP = Builder.CreateInBoundsGEP(GEP->getSourceElementType(), GEP->getPointerOperand(), std::vector<Value*>{NewSext});
  auto NewLoad = Builder.CreateLoad(GEP->getSourceElementType(), NewGEP);
  // store the value from the original array to the shared memory array
  auto Sext = Builder.CreateSExt(getThreadIdx(Func, 0), Type::getInt64Ty(M->getContext()));
  auto Zero = ConstantInt::get(Type::getInt64Ty(M->getContext()), 0);
  auto SharedGEP = Builder.CreateInBoundsGEP(sharedArrayType, sharedArray, std::vector<Value*>{Zero, Sext});
  auto LastStore = Builder.CreateStore(NewLoad, SharedGEP);
  return NewBaseAddress;
}

// Inserts a prefetching instructions into the loop
// Creates an initial prefetch outside the loop and updates the prefetch inside
void Prefetch(BasicBlock *OuterPreheader, BasicBlock* OuterLatch, BasicBlock *OuterHeader,
              BasicBlock *SharedMemoryBlock, BasicBlock *PrefetchBlock, BasicBlock *InnerHeader, 
              BasicBlock *InnerLatch, GetElementPtrInst *GEP, Value *ConstantForHW, 
              CallInst *ThreadIdxCall, Value *NewBaseAddress, Value *NewStepValue, Value *UpperBound,
              IRBuilder<> &Builder) {
  auto Func = OuterPreheader->getParent();
  auto M = Func->getParent();

  // Outside the outer loop, compute the address to prefetch
  Builder.SetInsertPoint(OuterPreheader->getTerminator());
  auto InitalPrefetchAdd = Builder.CreateAdd(ConstantForHW, getThreadIdx(Func, 0), "prefetch_add");
  auto InitalPrefetchSext = Builder.CreateSExt(InitalPrefetchAdd, Type::getInt64Ty(M->getContext()));
  auto InitalPrefetchGEP = Builder.CreateInBoundsGEP(GEP->getSourceElementType(), GEP->getPointerOperand(), std::vector<Value*>{InitalPrefetchSext});
  auto InitalPrefetchLoad = Builder.CreateLoad(GEP->getSourceElementType(), InitalPrefetchGEP);

  // next, we need to prefetch the next data in the loop
  Builder.SetInsertPoint(SharedMemoryBlock->getTerminator());
  // Check that the prefetch address is within bounds
  auto NextPrefetchAdd = Builder.CreateAdd(NewBaseAddress, NewStepValue, "next_prefetch_add");
  auto NextPrefetchSLT = Builder.CreateICmpSLT(NextPrefetchAdd, UpperBound, "next_prefetch_slt");
  // Remove the old branch and replace with a new conditional branch
  // if condition is true, go to prefetch block. Otherwise, go to inner header
  auto NextPrefetchBranch = Builder.CreateCondBr(NextPrefetchSLT, PrefetchBlock, InnerHeader);
  SharedMemoryBlock->getTerminator()->eraseFromParent();

  Builder.SetInsertPoint(PrefetchBlock);
  auto NextPrefetchAddThreadidx = Builder.CreateAdd(NextPrefetchAdd, getThreadIdx(Func, 0), "next_prefetch_add_threadidx");
  // Add with the constant for half warp
  auto NextPrefetchAddress = Builder.CreateAdd(NextPrefetchAddThreadidx, ConstantForHW, "next_prefetch_add_threadidx");
  auto NextPrefetchSext = Builder.CreateSExt(NextPrefetchAddress, Type::getInt64Ty(M->getContext()));
  auto NextPrefetchGEP = Builder.CreateInBoundsGEP(GEP->getSourceElementType(), GEP->getPointerOperand(), std::vector<Value*>{NextPrefetchSext});
  auto NextPrefetchLoad = Builder.CreateLoad(GEP->getSourceElementType(), NextPrefetchGEP);
  Builder.CreateBr(InnerHeader);

  // Create a PHI node in the outer header to select between the next and initial prefetch
  Builder.SetInsertPoint(OuterHeader->getFirstNonPHI());
  auto PrefetchPHI = Builder.CreatePHI(GEP->getSourceElementType(), 2, "prefetch_phi");
  PrefetchPHI->addIncoming(InitalPrefetchLoad, OuterPreheader);

  // Create a phi node in case the prefetch block is not taken
  Builder.SetInsertPoint(InnerHeader->getFirstNonPHI());
  auto PrefetchNextPHI = Builder.CreatePHI(GEP->getSourceElementType(), 2, "prefetch_phi_update");
  PrefetchNextPHI->addIncoming(PrefetchPHI, SharedMemoryBlock);
  PrefetchNextPHI->addIncoming(NextPrefetchLoad, PrefetchBlock);
  PrefetchNextPHI->addIncoming(PrefetchNextPHI, InnerLatch);
  
  // We need to fix the phi nodes of the inner header now that the prefetch 
  // block is a new predecessor
  for (auto &I : *InnerHeader) {
    if (auto *PHI = dyn_cast<PHINode>(&I)) {
      int NumIncoming = PHI->getNumIncomingValues();
      if (NumIncoming == 3)
        continue;
      else if (NumIncoming != 2) {
        assert(false && "Invalid number of incoming values");
      }

      // Add the same value as the shared memory block
      for (unsigned i = 0; i < PHI->getNumIncomingValues(); i++) {
        if (PHI->getIncomingBlock(i) == SharedMemoryBlock) {
          PHI->addIncoming(PHI->getIncomingValue(i), PrefetchBlock);
        }
      }
    }
  }

  PrefetchPHI->addIncoming(PrefetchNextPHI, OuterLatch);
}

// Returns where the memory barrier will be inserted
// Insertion point will incrementally be updated to farther down the block
Instruction *NVPTXMemOpts::CoalescePattern1(LoadInst *LI, std::vector<std::pair<LoadInst*, GetElementPtrInst*>> &LoadsToReplace) {
  assert (LI && "LI is null");
  auto GEP = dyn_cast<GetElementPtrInst>(LI->getPointerOperand());
  assert (GEP && "GEP is null");
  auto M = LI->getParent()->getParent()->getParent();

  // First, we need to create a shared memory buffer to store the data
  // We will use the same type as the original array
  auto sharedArrayType = getSharedArrayType(GEP);
  auto sharedArray = CreateSharedArray(sharedArrayType, LI);

  IRBuilder<> Builder(LI);
  // First, we need to load the value from the original array.
  // This will be loaded into shared memory.
  auto LoadInst = Builder.CreateLoad(GEP->getSourceElementType(), GEP);

  Value *SharedGEP = sharedArray;
  Type *SharedType = sharedArrayType;
  
  // This loop will create GEP for potentially multi-dimensional arrays 
  for (int i = 2; i >= 0; i--) {
    if (getThreadIdx(LI->getParent()->getParent(), i) == nullptr) {
      continue;
    }
    auto threadIdx = getThreadIdx(LI->getParent()->getParent(), i);
    auto TidZeroExt = Builder.CreateZExt(threadIdx, Type::getInt64Ty(M->getContext()));
    auto ZeroVal = ConstantInt::get(Type::getInt64Ty(M->getContext()), 0);
    SharedGEP = Builder.CreateInBoundsGEP(SharedType, SharedGEP, std::vector<Value*>{ZeroVal, TidZeroExt});
    SharedType = SharedType->getArrayElementType();
  }
  
  // store the value from the original array to the shared memory array
  Instruction* LastStore = Builder.CreateStore(LoadInst, SharedGEP);

  // Finally, replace the load location with the shared memory location
  LoadsToReplace.push_back(std::make_pair(LI, cast<GetElementPtrInst>(SharedGEP)));

  // return the insertion point for the barrier. This is where the __syncthreads() will be inserted
  return LastStore->getNextNode();
}

void NVPTXMemOpts::CoalescePattern2(LoadInst *LI, PHINode *IndVar, Loop *L, ScalarEvolution &SE) {
  auto Func = L->getHeader()->getParent();
  auto M = LI->getParent()->getParent()->getParent();
  LLVMContext &Context = L->getLoopPreheader()->getContext();
  
  // Next, we need to create a new outer loop that will iterate the same range
  // but with a stride of `thread block size`
  auto *NewIndVar = L->getCanonicalInductionVariable();
  auto OuterHeader = L->getHeader();
  auto OuterPreheader = L->getLoopPreheader();
  auto OuterLatch = L->getLoopLatch();
  auto OuterBody = OuterLatch->getSinglePredecessor();
  if (!OuterBody) {
    return;
  }

  // Next, we need to create the inner loop's basic blocks
  BasicBlock *InnerHeader = BasicBlock::Create(Context, "inner.header", Func, OuterHeader);
  BasicBlock *InnerBody = BasicBlock::Create(Context, "inner.body", Func, InnerHeader);
  BasicBlock *InnerLatch = BasicBlock::Create(Context, "inner.latch", Func, InnerBody);
  BasicBlock *SharedMemoryBlock = BasicBlock::Create(Context, "shared_memory", Func, InnerLatch);
  BasicBlock *PrefetchBlock = BasicBlock::Create(Context, "prefetch", Func, SharedMemoryBlock);

  auto GEP = dyn_cast<GetElementPtrInst>(LI->getPointerOperand());
  assert (GEP && "GEP is null");

  // Tile the loop by creating a new induction variable and an inner loop

  // Create a new induction variable for the outer loop
  IRBuilder<> Builder(OuterHeader->getFirstNonPHI());
  auto *NewStepValue = ConstantInt::get(IndVar->getType(), threadBlockSize[0]);
  PHINode *NewIndVarPHI = Builder.CreatePHI(IndVar->getType(), 2, "new_indvar");
  NewIndVarPHI->addIncoming(ConstantInt::get(IndVar->getType(), 0), L->getLoopPreheader());

  // Create new increment instruction
  Builder.SetInsertPoint(L->getLoopLatch()->getTerminator());
  Value *NewInc = Builder.CreateAdd(NewIndVarPHI, NewStepValue, "new_inc");
  NewIndVarPHI->addIncoming(NewInc, L->getLoopLatch());

  // Next, we need to update the loop condition to reflect the new induction variable
  auto *Cond = L->getHeader()->getTerminator()->getOperand(0);
  auto *UpperBound = dyn_cast<ICmpInst>(Cond)->getOperand(1);
  Builder.SetInsertPoint(L->getHeader()->getTerminator());
  auto *NewCond = Builder.CreateICmpSLT(NewIndVarPHI, UpperBound, "new_cond");
  L->getHeader()->getTerminator()->setOperand(0, NewCond);

  // Redirect outer loop to inner loop header
  OuterHeader->getTerminator()->setSuccessor(0, SharedMemoryBlock);
  Builder.SetInsertPoint(SharedMemoryBlock);
  Builder.CreateBr(InnerHeader);

  // Inner loop induction variable
  Builder.SetInsertPoint(InnerHeader);
  PHINode *InnerIndVar = Builder.CreatePHI(IndVar->getType(), 2, "inner_indvar");
  InnerIndVar->addIncoming(ConstantInt::get(IndVar->getType(), 0), SharedMemoryBlock);

  // Condition check for inner loop header
  auto *InnerCond = Builder.CreateICmpSLT(InnerIndVar, NewStepValue, "inner_cond");
  Builder.CreateCondBr(InnerCond, InnerBody, OuterLatch);

  // Create a new variable old_indvar + inner_indvar, and replace all uses of old_indvar with this
  Builder.SetInsertPoint(InnerBody);
  auto *NewIndVarValue = Builder.CreateAdd(NewIndVar, InnerIndVar, "new_indvar_value");
  // Branch to Outer Loop Body
  Builder.CreateBr(OuterBody);
  // Replace all uses of old induction only in the outer body
  for (auto &I : *OuterBody) {
    I.replaceUsesOfWith(IndVar, NewIndVarValue);
  }
  
  // Replace successor with inner exit
  OuterBody->getTerminator()->setSuccessor(0, InnerLatch);

  // Increment inner loop induction variable
  Builder.SetInsertPoint(InnerLatch);
  Value *InnerInc = Builder.CreateAdd(InnerIndVar, ConstantInt::get(IndVar->getType(), 1), "inner_inc");
  Builder.CreateBr(InnerHeader);
  InnerIndVar->addIncoming(InnerInc, InnerLatch);

  // We need special handling for the phis in the outer loop
  // one of the incoming values is no longer dominated by the outer body.
  // We need to insert a dummy phi node in the outer latch to fix this
  // these are the non-induction variable phis in the outer loop header
  for (auto &I : *OuterHeader) {
    if (auto *PHI = dyn_cast<PHINode>(&I)) {
      // Check if any of the incoming values are defined in the outer body
      for (unsigned i = 0; i < PHI->getNumIncomingValues(); i++) {
        auto PhiIncoming = dyn_cast<Instruction>(PHI->getIncomingValue(i));
        if (PHI->getIncomingBlock(i) == OuterLatch && 
          PhiIncoming &&
          PhiIncoming->getParent() == OuterBody) {
            Builder.SetInsertPoint(InnerHeader->getFirstNonPHI());
            auto *DummyPHI = Builder.CreatePHI(PHI->getType(), 2, "new_phi");
            DummyPHI->addIncoming(PHI, SharedMemoryBlock);
            DummyPHI->addIncoming(PhiIncoming, InnerLatch);
            PHI->setOperand(i, DummyPHI);
        }
      }
    }
  }
  
  // The loop is now tiled

  auto sharedArrayType = getSharedArrayType(GEP);
  auto sharedArray = CreateSharedArray(sharedArrayType, LI);
  
  // We need to copy the data from the original array to the shared memory array
  // buffer will be populated before entering the inner loop
  auto idx = GEP->idx_begin();
  auto SignExt = dyn_cast<SExtInst>(idx);
  assert(SignExt && "SignExt is null");
  auto Index = dyn_cast<BinaryOperator>(SignExt->getOperand(0));
  auto ConstantForHW = Index->getOperand(1) == NewIndVarValue ? Index->getOperand(0) : Index->getOperand(1);

  Value *NewBaseAddress = LoadStoreToSharedMemory(SharedMemoryBlock, LI, GEP, 
                                                           sharedArray, sharedArrayType, 
                                                           ConstantForHW, NewIndVarPHI, Builder);

  // Insert a barrier
  InsertBarrier(SharedMemoryBlock->getTerminator());

  // In the inner loop body, we want to replace the load instruction with the shared memory load
  Builder.SetInsertPoint(OuterBody->getFirstNonPHIOrDbgOrLifetime());
  auto InnerIndVarSext = Builder.CreateSExt(InnerIndVar, Type::getInt64Ty(M->getContext()));
  auto InnerZero = ConstantInt::get(Type::getInt64Ty(M->getContext()), 0);
  auto InnerSharedGEP = Builder.CreateInBoundsGEP(sharedArrayType, sharedArray, std::vector<Value*>{InnerIndVarSext});
  auto InnerSharedLoad = Builder.CreateLoad(GEP->getSourceElementType(), InnerSharedGEP);
  LI->replaceAllUsesWith(InnerSharedLoad);

  // Insert another barrier at the end of the inner loop
  InsertBarrier(OuterLatch->getFirstNonPHIOrDbgOrLifetime()); 

  // inputs: OuterPreheader, SharedMemoryBlock, PrefetchBlock, InnerHeader, InnerLatch, Outer Latch, ConstantForHW, threadIdx call, GEP, NewBaseAddress, NewStepValue, UpperBounds, M
  Prefetch(OuterPreheader, OuterLatch, OuterHeader, 
           SharedMemoryBlock, PrefetchBlock, InnerHeader, 
           InnerLatch, GEP, ConstantForHW, getThreadIdx(Func, 0), 
           NewBaseAddress, NewStepValue, UpperBound, Builder);
}

// We had to turn off this coalescing because it was causing incorrect program outputs
// We were unable to debug the issue in time. We left our code here for reference
void NVPTXMemOpts::CoalescePattern3(LoadInst *LI, PHINode *IndVar, Loop *L, ScalarEvolution &SE) {
  return;
  auto Func = L->getHeader()->getParent();
  auto M = LI->getParent()->getParent()->getParent();
  LLVMContext &Context = L->getLoopPreheader()->getContext();
  auto GEP = dyn_cast<GetElementPtrInst>(LI->getPointerOperand());

  // Next, we need to create a new outer loop that will iterate the same range
  // but with a stride of `thread block size`
  auto *NewIndVar = L->getCanonicalInductionVariable();
  auto OuterHeader = L->getHeader();
  auto OuterPreheader = L->getLoopPreheader();
  auto OuterLatch = L->getLoopLatch();
  auto OuterBody = OuterLatch->getSinglePredecessor();
  if (!OuterBody) {
    return;
  }

  // Create the inner loop's basic blocks
  BasicBlock *InnerHeader = BasicBlock::Create(Context, "inner.header", Func, OuterHeader);
  BasicBlock *InnerBody = BasicBlock::Create(Context, "inner.body", Func, InnerHeader);
  BasicBlock *InnerLatch = BasicBlock::Create(Context, "inner.latch", Func, InnerBody);
  BasicBlock *SharedMemoryBlock = BasicBlock::Create(Context, "shared_memory", Func, InnerLatch);
  BasicBlock *PrefetchBlock = BasicBlock::Create(Context, "prefetch", Func, SharedMemoryBlock);

  IRBuilder<> Builder(OuterHeader->getFirstNonPHI());

  /* We were not able to tile this loop/shared memroy correctly.
   we omitted our code as it looks very similar to CoalescePattern2*/

  // Our stepping value is possibly not correct. TODO:: Fix this
  Prefetch(OuterPreheader, OuterLatch, OuterHeader, 
          SharedMemoryBlock, PrefetchBlock, InnerHeader, 
          InnerLatch, GEP, nullptr, getThreadIdx(Func, 0), 
          nullptr, nullptr, nullptr, Builder);

  return;
}

// ================================================================
// ======================= Driver Functions =======================
// ================================================================

bool NVPTXMemOpts::CoalesceLoops(LoopInfo &LIInfo, ScalarEvolution &SE, const DataLayout &DL) {
  bool Changed = false;
  for (auto Loop : LIInfo){
    // if the loop is not in simplified form, we cannot analyze it
    if (!Loop->isLoopSimplifyForm()) {
      continue;
    }

    // Loop induction variable must start at 0 and increment by 1
    auto *IndVar = Loop->getCanonicalInductionVariable();
    if (!IndVar) {  
      return false;
    }

    std::vector<LoadInst*> toDelete;
    for (auto *BB : Loop->getBlocks()){
      for (auto Iter = BB->begin(); Iter != BB->end(); ++Iter) {
        Instruction &I = *Iter;
        if (LoadInst *LI = dyn_cast<LoadInst>(&I)) {
          if (PatternMatch2(LI, IndVar)) {
            CoalescePattern2(LI, IndVar, Loop, SE);
            Changed = true;
            toDelete.push_back(LI);
          } else if (PatternMatch3(LI, IndVar)) {
            CoalescePattern3(LI, IndVar, Loop, SE);
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
  }
  return Changed;
}

bool NVPTXMemOpts::CoalesceBasicBlocks(Function &F) {
  bool Changed = false;
  // TODO:: This is not a nvptx function so todelete wont work
  std::vector<LoadInst*> toDelete;
  for (auto &BB : F) {
    Instruction *BarrierInsertionPoint = nullptr;
    std::vector<std::pair<LoadInst*, GetElementPtrInst*>> LoadsToReplace;
    for (auto Iter = BB.begin(); Iter != BB.end(); ++Iter) {
      Instruction &I = *Iter;
      auto *LI = dyn_cast<LoadInst>(&I);
      if (!LI) { continue; }
      if (!PatternMatch1(LI)) { continue; }
      if (!isCallCoalescable(LI)) { continue; }
      BarrierInsertionPoint = CoalescePattern1(LI, LoadsToReplace);
    }
    
    if (BarrierInsertionPoint) {
      InsertBarrier(BarrierInsertionPoint);
    }

    // Replace loads with shared memory loads
    for (auto &pair : LoadsToReplace) {
      auto LI = pair.first;
      auto GEP = pair.second;
      // GEP is array type. We need to get the element type
      auto BaseType = cast<ArrayType>(GEP->getSourceElementType())->getElementType();
      
      IRBuilder<> Builder(BarrierInsertionPoint);
      auto sharedLoad = Builder.CreateLoad(BaseType, GEP);
      LI->replaceAllUsesWith(sharedLoad);
      LI->eraseFromParent();
    }

    for (auto LI : toDelete) {
      // assert that the LI has no uses
      assert(LI->use_empty());
      LI->eraseFromParent();
    }
    toDelete.clear();
  }
  return Changed;
}

bool NVPTXMemOpts::runOnFunction(Function &F) {
  // Use marker function to find out the threadblock size
  GetThreadBlockSize(F);
  if (threadBlockSize.empty()) {  // cannot reason without thread block size
    return false;
  }

  LoopInfo &LIInfo = getAnalysis<LoopInfoWrapperPass>(F).getLoopInfo();
  ScalarEvolution &SE = getAnalysis<ScalarEvolutionWrapperPass>(F).getSE();
  const DataLayout &DL = F.getParent()->getDataLayout();

  bool Changed = false;
  // First coalesce loops, then non-loop basic blocks
  Changed |= CoalesceLoops(LIInfo, SE, DL);
  Changed |= CoalesceBasicBlocks(F);

  return Changed;
}

bool NVPTXMemOpts::runOnModule(Module &M) {
  bool Changed = false;
  for (auto &F : M) {
    if (F.empty() || F.isDeclaration())
      continue;

    Changed |= runOnFunction(F);
    threadBlockSize.clear();
  }
  return Changed;
}

// } // end anonymous namespace

namespace llvm {
void initializeNVPTXMemOptsPass(PassRegistry &);
}

INITIALIZE_PASS(NVPTXMemOpts, "nvptx-mem-opts",
                "Memory coalescing and prefetching",
                true, false)

ModulePass *llvm::createNVPTXMemOptsPass() {
  return new NVPTXMemOpts();
}
