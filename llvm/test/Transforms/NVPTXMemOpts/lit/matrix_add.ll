; ModuleID = 'test-cuda-nvptx64-nvidia-cuda-sm_35.ll'
source_filename = "./test.cu"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%struct.__cuda_builtin_blockIdx_t = type { i8 }
%struct.__cuda_builtin_blockDim_t = type { i8 }
%struct.__cuda_builtin_threadIdx_t = type { i8 }

@blockIdx = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_blockIdx_t, align 1
@blockDim = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_blockDim_t, align 1
@threadIdx = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_threadIdx_t, align 1
@_ZZ19matrixAdd_coalescedPKfS0_PfiiE8A_shared = internal addrspace(3) global [16 x [16 x float]] undef, align 4
@_ZZ19matrixAdd_coalescedPKfS0_PfiiE8B_shared = internal addrspace(3) global [16 x [16 x float]] undef, align 4

; Function Attrs: convergent mustprogress noinline norecurse nounwind
define dso_local void @dummy_func() #0 {
; CHECK-LABEL: @dummy_func
; CHECK: entry
; CHECK-NEXT: ret void
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
define dso_local void @_Z9matrixAddPKfS0_Pfii(ptr noundef %A, ptr noundef %B, ptr noundef %C, i32 noundef %numRows, i32 noundef %numCols) #0 {
; CHECK-LABEL: @_Z9matrixAddPKfS0_Pfii
; CHECK: entry:
; CHECK-NEXT: %C5 = addrspacecast ptr %C to ptr addrspace(1)
; CHECK-NEXT: %B3 = addrspacecast ptr %B to ptr addrspace(1)
; CHECK-NEXT: %A1 = addrspacecast ptr %A to ptr addrspace(1)
; CHECK-NEXT: %0 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
; CHECK-NEXT: %1 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
; CHECK-NEXT: %mul = mul i32 %0, %1
; CHECK-NEXT: %2 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
; CHECK-NEXT: %add = add i32 %mul, %2
; CHECK-NEXT: %3 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
; CHECK-NEXT: %4 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
; CHECK-NEXT: %mul5 = mul i32 %3, %4
; CHECK-NEXT: %5 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
; CHECK-NEXT: %add7 = add i32 %mul5, %5
; CHECK-NEXT: %cmp = icmp slt i32 %add, %numRows
; CHECK-NEXT: br i1 %cmp, label %land.lhs.true, label %if.end
; CHECK: land.lhs.true:
; CHECK-NEXT: %cmp8 = icmp slt i32 %add7, %numCols
; CHECK-NEXT: br i1 %cmp8, label %if.then, label %if.end
; CHECK: if.then:
; CHECK-NEXT: %mul9 = mul nsw i32 %add, %numCols
; CHECK-NEXT: %add10 = add nsw i32 %mul9, %add7
; CHECK-NEXT: %idxprom = sext i32 %add10 to i64
; CHECK-NEXT: %arrayidx = getelementptr inbounds float, ptr addrspace(1) %A1, i64 %idxprom
; CHECK-NEXT: %6 = load float, ptr addrspace(1) %arrayidx, align 4
; CHECK-NEXT: %7 = zext i32 %2 to i64
; CHECK-NEXT: %8 = getelementptr inbounds [16 x [16 x float]], ptr addrspace(3) @_Z9matrixAddPKfS0_Pfiiinput_shared, i64 0, i64 %7
; CHECK-NEXT: %9 = zext i32 %5 to i64
; CHECK-NEXT: %10 = getelementptr inbounds [16 x float], ptr addrspace(3) %8, i64 0, i64 %9
; CHECK-NEXT: store float %6, ptr addrspace(3) %10, align 4
; CHECK-NEXT: %arrayidx12 = getelementptr inbounds float, ptr addrspace(1) %B3, i64 %idxprom
; CHECK-NEXT: %11 = load float, ptr addrspace(1) %arrayidx12, align 4
; CHECK-NEXT: %12 = zext i32 %2 to i64
; CHECK-NEXT: %13 = getelementptr inbounds [16 x [16 x float]], ptr addrspace(3) @_Z9matrixAddPKfS0_Pfiiinput_shared1, i64 0, i64 %12
; CHECK-NEXT: %14 = zext i32 %5 to i64
; CHECK-NEXT: %15 = getelementptr inbounds [16 x float], ptr addrspace(3) %13, i64 0, i64 %14
; CHECK-NEXT: store float %11, ptr addrspace(3) %15, align 4
; CHECK-NEXT: call void @llvm.nvvm.barrier0()
; CHECK-NEXT: %16 = load float, ptr addrspace(3) %10, align 4
; CHECK-NEXT: %17 = load float, ptr addrspace(3) %15, align 4
; CHECK-NEXT: %add13 = fadd contract float %16, %17
; CHECK-NEXT: %arrayidx15 = getelementptr inbounds float, ptr addrspace(1) %C5, i64 %idxprom
; CHECK-NEXT: store float %add13, ptr addrspace(1) %arrayidx15, align 4
; CHECK-NEXT: br label %if.end
; CHECK: if.end:
; CHECK-NEXT: ret void
entry:
  %C5 = addrspacecast ptr %C to ptr addrspace(1)
  %B3 = addrspacecast ptr %B to ptr addrspace(1)
  %A1 = addrspacecast ptr %A to ptr addrspace(1)
  call void @__tb_size_marker_2D(i32 noundef 16, i32 noundef 16) #4
  %0 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %1 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
  %mul = mul i32 %0, %1
  %2 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %add = add i32 %mul, %2
  %3 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %4 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %mul5 = mul i32 %3, %4
  %5 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %add7 = add i32 %mul5, %5
  %cmp = icmp slt i32 %add, %numRows
  br i1 %cmp, label %land.lhs.true, label %if.end

land.lhs.true:                                    ; preds = %entry
  %cmp8 = icmp slt i32 %add7, %numCols
  br i1 %cmp8, label %if.then, label %if.end

if.then:                                          ; preds = %land.lhs.true
  %mul9 = mul nsw i32 %add, %numCols
  %add10 = add nsw i32 %mul9, %add7
  %idxprom = sext i32 %add10 to i64
  %arrayidx = getelementptr inbounds float, ptr addrspace(1) %A1, i64 %idxprom
  %6 = load float, ptr addrspace(1) %arrayidx, align 4
  %arrayidx12 = getelementptr inbounds float, ptr addrspace(1) %B3, i64 %idxprom
  %7 = load float, ptr addrspace(1) %arrayidx12, align 4
  %add13 = fadd contract float %6, %7
  %arrayidx15 = getelementptr inbounds float, ptr addrspace(1) %C5, i64 %idxprom
  store float %add13, ptr addrspace(1) %arrayidx15, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %land.lhs.true, %entry
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind
define dso_local void @_Z19matrixAdd_coalescedPKfS0_Pfii(ptr noundef %A, ptr noundef %B, ptr noundef %C, i32 noundef %numRows, i32 noundef %numCols) #0 {
entry:
  %C5 = addrspacecast ptr %C to ptr addrspace(1)
  %B3 = addrspacecast ptr %B to ptr addrspace(1)
  %A1 = addrspacecast ptr %A to ptr addrspace(1)
  %0 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %1 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
  %mul = mul i32 %0, %1
  %2 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %add = add i32 %mul, %2
  %3 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %4 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %mul5 = mul i32 %3, %4
  %5 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %add7 = add i32 %mul5, %5
  %cmp = icmp slt i32 %add, %numRows
  br i1 %cmp, label %land.lhs.true, label %if.end

land.lhs.true:                                    ; preds = %entry
  %cmp8 = icmp slt i32 %add7, %numCols
  br i1 %cmp8, label %if.then, label %if.end

if.then:                                          ; preds = %land.lhs.true
  %mul9 = mul nsw i32 %add, %numCols
  %add10 = add nsw i32 %mul9, %add7
  %idxprom = sext i32 %add10 to i64
  %arrayidx = getelementptr inbounds float, ptr addrspace(1) %A1, i64 %idxprom
  %6 = load float, ptr addrspace(1) %arrayidx, align 4
  %idxprom12 = zext i32 %2 to i64
  %arrayidx13 = getelementptr inbounds [16 x [16 x float]], ptr addrspace(3) @_ZZ19matrixAdd_coalescedPKfS0_PfiiE8A_shared, i64 0, i64 %idxprom12
  %idxprom15 = zext i32 %5 to i64
  %arrayidx16 = getelementptr inbounds [16 x float], ptr addrspace(3) %arrayidx13, i64 0, i64 %idxprom15
  store float %6, ptr addrspace(3) %arrayidx16, align 4
  %arrayidx20 = getelementptr inbounds float, ptr addrspace(1) %B3, i64 %idxprom
  %7 = load float, ptr addrspace(1) %arrayidx20, align 4
  %arrayidx23 = getelementptr inbounds [16 x [16 x float]], ptr addrspace(3) @_ZZ19matrixAdd_coalescedPKfS0_PfiiE8B_shared, i64 0, i64 %idxprom12
  %arrayidx26 = getelementptr inbounds [16 x float], ptr addrspace(3) %arrayidx23, i64 0, i64 %idxprom15
  store float %7, ptr addrspace(3) %arrayidx26, align 4
  call void @llvm.nvvm.barrier0()
  %8 = load float, ptr addrspace(3) %arrayidx16, align 4
  %9 = load float, ptr addrspace(3) %arrayidx26, align 4
  %add41 = fadd contract float %8, %9
  %arrayidx43 = getelementptr inbounds float, ptr addrspace(1) %C5, i64 %idxprom
  store float %add41, ptr addrspace(1) %arrayidx43, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %land.lhs.true, %entry
  ret void
}

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier0() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y() #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #3

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
!5 = !{ptr @_Z9matrixAddPKfS0_Pfii, !"kernel", i32 1}
!6 = !{ptr @_Z19matrixAdd_coalescedPKfS0_Pfii, !"kernel", i32 1}
!7 = !{!"clang version 18.1.3 (https://github.com/virnarula/llvm-project.git 7709d36816961113db93da8bb9c3a1d05b7c2c0f)"}
!8 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}