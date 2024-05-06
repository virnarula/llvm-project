
; ModuleID = 'test-cuda-nvptx64-nvidia-cuda-sm_35.ll'
source_filename = "./test.cu"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%struct.__cuda_builtin_blockDim_t = type { i8 }
%struct.__cuda_builtin_blockIdx_t = type { i8 }
%struct.__cuda_builtin_threadIdx_t = type { i8 }

@blockDim = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_blockDim_t, align 1
@blockIdx = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_blockIdx_t, align 1
@threadIdx = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_threadIdx_t, align 1
@_ZZ30vectorScalarMultiply_coalescedPKfPffiE12input_shared = internal addrspace(3) global [16 x float] undef, align 4

; Function Attrs: convergent mustprogress noinline norecurse nounwind
define dso_local void @dummy_func() #0 {
entry:
  call void @__tb_size_marker_1D(i32 noundef 1) #4
  call void @__tb_size_marker_2D(i32 noundef 1, i32 noundef 1) #4
  call void @__tb_size_marker_3D(i32 noundef 1, i32 noundef 1, i32 noundef 1) #4
  ret void
}

; Function Attrs: convergent nounwind
declare dso_local void @__tb_size_marker_1D(i32 noundef %0) #1

; Function Attrs: convergent nounwind
declare dso_local void @__tb_size_marker_2D(i32 noundef %0, i32 noundef %1) #1

; Function Attrs: convergent nounwind
declare dso_local void @__tb_size_marker_3D(i32 noundef %0, i32 noundef %1, i32 noundef %2) #1

; Function Attrs: convergent mustprogress noinline norecurse nounwind
define dso_local void @_Z20vectorScalarMultiplyPKfPffi(ptr noundef %input, ptr noundef %output, float noundef %scalar, i32 noundef %numElements) #0 {
; CHECK-LABEL: @_Z20vectorScalarMultiplyPKfPffi
; CHECK:   %output3 = addrspacecast ptr %output to ptr addrspace(1)
; CHECK-NEXT:   %input1 = addrspacecast ptr %input to ptr addrspace(1)
; CHECK-NEXT:   %0 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
; CHECK-NEXT:   %1 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
; CHECK-NEXT:   %mul = mul i32 %0, %1
; CHECK-NEXT:   %2 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
; CHECK-NEXT:   %add = add i32 %mul, %2
; CHECK-NEXT:   %cmp = icmp slt i32 %add, %numElements
; CHECK-NEXT:   %idxprom = sext i32 %add to i64
; CHECK-NEXT:   %arrayidx = getelementptr inbounds float, ptr addrspace(1) %input1, i64 %idxprom
; CHECK-NEXT:   %arrayidx5 = getelementptr inbounds float, ptr addrspace(1) %output3, i64 %idxprom
; CHECK-NEXT:   br i1 %cmp, label %if.then, label %if.end
; CHECK:   if.then: 
; CHECK-NEXT:   %3 = load float, ptr addrspace(1) %arrayidx, align 4
; CHECK-NEXT:   %4 = zext i32 %2 to i64
; CHECK-NEXT:   %5 = getelementptr inbounds [16 x float], ptr addrspace(3) @_Z20vectorScalarMultiplyPKfPffiinput_shared, i64 0, i64 %4
; CHECK-NEXT:   store float %3, ptr addrspace(3) %5, align 4
; CHECK-NEXT:   call void @llvm.nvvm.barrier0()
; CHECK-NEXT:   %6 = load float, ptr addrspace(3) %5, align 4
; CHECK-NEXT:   %mul3 = fmul contract float %6, %scalar
; CHECK-NEXT:   store float %mul3, ptr addrspace(1) %arrayidx5, align 4
; CHECK-NEXT:   br label %if.end
; CHECK:   if.end
; CHECK-NEXT:   ret void
entry:
  %output3 = addrspacecast ptr %output to ptr addrspace(1)
  %input1 = addrspacecast ptr %input to ptr addrspace(1)
  call void @__tb_size_marker_1D(i32 noundef 16) #4
  %0 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %1 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %mul = mul i32 %0, %1
  %2 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %add = add i32 %mul, %2
  %cmp = icmp slt i32 %add, %numElements
  %idxprom = sext i32 %add to i64
  %arrayidx = getelementptr inbounds float, ptr addrspace(1) %input1, i64 %idxprom
  %arrayidx5 = getelementptr inbounds float, ptr addrspace(1) %output3, i64 %idxprom
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %3 = load float, ptr addrspace(1) %arrayidx, align 4
  %mul3 = fmul contract float %3, %scalar
  store float %mul3, ptr addrspace(1) %arrayidx5, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind
define dso_local void @_Z30vectorScalarMultiply_coalescedPKfPffi(ptr noundef %input, ptr noundef %output, float noundef %scalar, i32 noundef %numElements) #0 {
entry:
  %output3 = addrspacecast ptr %output to ptr addrspace(1)
  %input1 = addrspacecast ptr %input to ptr addrspace(1)
  %0 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %1 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %mul = mul i32 %0, %1
  %2 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %add = add i32 %mul, %2
  %cmp = icmp slt i32 %add, %numElements
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %idxprom = sext i32 %add to i64
  %arrayidx = getelementptr inbounds float, ptr addrspace(1) %input1, i64 %idxprom
  %3 = load float, ptr addrspace(1) %arrayidx, align 4
  %idxprom4 = zext i32 %2 to i64
  %arrayidx5 = getelementptr inbounds [16 x float], ptr addrspace(3) @_ZZ30vectorScalarMultiply_coalescedPKfPffiE12input_shared, i64 0, i64 %idxprom4
  store float %3, ptr addrspace(3) %arrayidx5, align 4
  call void @llvm.nvvm.barrier0()
  %4 = load float, ptr addrspace(3) %arrayidx5, align 4
  %mul9 = fmul contract float %4, %scalar
  %arrayidx11 = getelementptr inbounds float, ptr addrspace(1) %output3, i64 %idxprom
  store float %mul9, ptr addrspace(1) %arrayidx11, align 4
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

attributes #0 = { convergent mustprogress noinline norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_35" "target-features"="+ptx75,+sm_35" "uniform-work-group-size"="true" }
attributes #1 = { convergent nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_35" "target-features"="+ptx75,+sm_35" }
attributes #2 = { convergent nocallback nounwind "target-cpu"="sm_35" }
attributes #3 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) "target-cpu"="sm_35" }
attributes #4 = { convergent nounwind }

!llvm.module.flags = !{!0, !1, !2, !3}
!nvvm.annotations = !{!4, !5, !6}
!llvm.ident = !{!7, !8}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 11, i32 5]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!3 = !{i32 7, !"frame-pointer", i32 2}
!4 = !{ptr @dummy_func, !"kernel", i32 1}
!5 = !{ptr @_Z20vectorScalarMultiplyPKfPffi, !"kernel", i32 1}
!6 = !{ptr @_Z30vectorScalarMultiply_coalescedPKfPffi, !"kernel", i32 1}
!7 = !{!"clang version 18.1.3 (https://github.com/virnarula/llvm-project.git 7709d36816961113db93da8bb9c3a1d05b7c2c0f)"}
!8 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}