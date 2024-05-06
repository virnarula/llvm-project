; ModuleID = 'test-cuda-nvptx64-nvidia-cuda-sm_35.bc'
source_filename = "test.cu"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%struct.__cuda_builtin_blockDim_t = type { i8 }
%struct.__cuda_builtin_blockIdx_t = type { i8 }
%struct.__cuda_builtin_threadIdx_t = type { i8 }

@blockDim = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_blockDim_t, align 1
@blockIdx = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_blockIdx_t, align 1
@threadIdx = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_threadIdx_t, align 1
@_ZZ24vectorMultiply_coalescedPKfS0_PfiE13input1_shared = internal addrspace(3) global [16 x float] undef, align 4
@_ZZ24vectorMultiply_coalescedPKfS0_PfiE13input2_shared = internal addrspace(3) global [16 x float] undef, align 4

; Function Attrs: convergent mustprogress noinline norecurse nounwind 
define dso_local void @dummy_func() #0 {
entry:
  call void @__tb_size_marker_1D(i32 noundef 1) #4
  call void @__tb_size_marker_2D(i32 noundef 1, i32 noundef 1) #4
  call void @__tb_size_marker_3D(i32 noundef 1, i32 noundef 1, i32 noundef 1) #4
  ret void
}

; Function Attrs: convergent nounwind
declare dso_local void @__tb_size_marker_1D(i32 noundef) #1

; Function Attrs: convergent nounwind
declare dso_local void @__tb_size_marker_2D(i32 noundef, i32 noundef) #1

; Function Attrs: convergent nounwind
declare dso_local void @__tb_size_marker_3D(i32 noundef, i32 noundef, i32 noundef) #1

; Function Attrs: convergent mustprogress noinline norecurse nounwind 
define dso_local void @_Z14vectorMultiplyPKfS0_Pfi(ptr noundef %input1, ptr noundef %input2, ptr noundef %output, i32 noundef %numElements) #0 {
entry:
  %input1.addr = alloca ptr, align 8
  %input2.addr = alloca ptr, align 8
  %output.addr = alloca ptr, align 8
  %numElements.addr = alloca i32, align 4
  %i = alloca i32, align 4
  store ptr %input1, ptr %input1.addr, align 8
  store ptr %input2, ptr %input2.addr, align 8
  store ptr %output, ptr %output.addr, align 8
  store i32 %numElements, ptr %numElements.addr, align 4
  %0 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %1 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %mul = mul i32 %0, %1
  %2 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %add = add i32 %mul, %2
  store i32 %add, ptr %i, align 4
  call void @__tb_size_marker_1D(i32 noundef 256) #4
  %3 = load i32, ptr %i, align 4
  %4 = load i32, ptr %numElements.addr, align 4
  %cmp = icmp slt i32 %3, %4
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %5 = load ptr, ptr %input1.addr, align 8
  %6 = load i32, ptr %i, align 4
  %idxprom = sext i32 %6 to i64
  %arrayidx = getelementptr inbounds float, ptr %5, i64 %idxprom
  %7 = load float, ptr %arrayidx, align 4
  %8 = load ptr, ptr %input2.addr, align 8
  %9 = load i32, ptr %i, align 4
  %idxprom3 = sext i32 %9 to i64
  %arrayidx4 = getelementptr inbounds float, ptr %8, i64 %idxprom3
  %10 = load float, ptr %arrayidx4, align 4
  %mul5 = fmul contract float %7, %10
  %11 = load ptr, ptr %output.addr, align 8
  %12 = load i32, ptr %i, align 4
  %idxprom6 = sext i32 %12 to i64
  %arrayidx7 = getelementptr inbounds float, ptr %11, i64 %idxprom6
  store float %mul5, ptr %arrayidx7, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind 
define dso_local void @_Z24vectorMultiply_coalescedPKfS0_Pfi(ptr noundef %input1, ptr noundef %input2, ptr noundef %output, i32 noundef %numElements) #0 {
entry:
  %input1.addr = alloca ptr, align 8
  %input2.addr = alloca ptr, align 8
  %output.addr = alloca ptr, align 8
  %numElements.addr = alloca i32, align 4
  %i = alloca i32, align 4
  store ptr %input1, ptr %input1.addr, align 8
  store ptr %input2, ptr %input2.addr, align 8
  store ptr %output, ptr %output.addr, align 8
  store i32 %numElements, ptr %numElements.addr, align 4
  %0 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %1 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %mul = mul i32 %0, %1
  %2 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %add = add i32 %mul, %2
  store i32 %add, ptr %i, align 4
  %3 = load ptr, ptr %input1.addr, align 8
  %4 = load i32, ptr %i, align 4
  %idxprom = sext i32 %4 to i64
  %arrayidx = getelementptr inbounds float, ptr %3, i64 %idxprom
  %5 = load float, ptr %arrayidx, align 4
  %6 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %idxprom4 = zext i32 %6 to i64
  %arrayidx5 = getelementptr inbounds [16 x float], ptr addrspacecast (ptr addrspace(3) @_ZZ24vectorMultiply_coalescedPKfS0_PfiE13input1_shared to ptr), i64 0, i64 %idxprom4
  store float %5, ptr %arrayidx5, align 4
  %7 = load ptr, ptr %input2.addr, align 8
  %8 = load i32, ptr %i, align 4
  %idxprom6 = sext i32 %8 to i64
  %arrayidx7 = getelementptr inbounds float, ptr %7, i64 %idxprom6
  %9 = load float, ptr %arrayidx7, align 4
  %10 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %idxprom9 = zext i32 %10 to i64
  %arrayidx10 = getelementptr inbounds [16 x float], ptr addrspacecast (ptr addrspace(3) @_ZZ24vectorMultiply_coalescedPKfS0_PfiE13input2_shared to ptr), i64 0, i64 %idxprom9
  store float %9, ptr %arrayidx10, align 4
  call void @llvm.nvvm.barrier0()
  %11 = load i32, ptr %i, align 4
  %12 = load i32, ptr %numElements.addr, align 4
  %cmp = icmp slt i32 %11, %12
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %13 = load ptr, ptr %input1.addr, align 8
  %14 = load i32, ptr %i, align 4
  %idxprom11 = sext i32 %14 to i64
  %arrayidx12 = getelementptr inbounds float, ptr %13, i64 %idxprom11
  %15 = load float, ptr %arrayidx12, align 4
  %16 = load ptr, ptr %input2.addr, align 8
  %17 = load i32, ptr %i, align 4
  %idxprom13 = sext i32 %17 to i64
  %arrayidx14 = getelementptr inbounds float, ptr %16, i64 %idxprom13
  %18 = load float, ptr %arrayidx14, align 4
  %mul15 = fmul contract float %15, %18
  %19 = load ptr, ptr %output.addr, align 8
  %20 = load i32, ptr %i, align 4
  %idxprom16 = sext i32 %20 to i64
  %arrayidx17 = getelementptr inbounds float, ptr %19, i64 %idxprom16
  store float %mul15, ptr %arrayidx17, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier0() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3

attributes #0 = { convergent mustprogress noinline norecurse nounwind  "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_35" "target-features"="+ptx75,+sm_35" "uniform-work-group-size"="true" }
attributes #1 = { convergent nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_35" "target-features"="+ptx75,+sm_35" }
attributes #2 = { convergent nocallback nounwind }
attributes #3 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { convergent nounwind }

!llvm.module.flags = !{!0, !1, !2, !3}
!nvvm.annotations = !{!4, !5, !6}
!llvm.ident = !{!7, !8}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 11, i32 5]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!3 = !{i32 7, !"frame-pointer", i32 2}
!4 = !{ptr @dummy_func, !"kernel", i32 1}
!5 = !{ptr @_Z14vectorMultiplyPKfS0_Pfi, !"kernel", i32 1}
!6 = !{ptr @_Z24vectorMultiply_coalescedPKfS0_Pfi, !"kernel", i32 1}
!7 = !{!"clang version 18.1.3 (https://github.com/virnarula/llvm-project.git 7709d36816961113db93da8bb9c3a1d05b7c2c0f)"}
!8 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
