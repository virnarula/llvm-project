; ModuleID = 'test-cuda-nvptx64-nvidia-cuda-sm_35.ll'
source_filename = "test.cu"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%struct.__cuda_builtin_blockDim_t = type { i8 }
%struct.__cuda_builtin_blockIdx_t = type { i8 }
%struct.__cuda_builtin_threadIdx_t = type { i8 }

@blockDim = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_blockDim_t, align 1
@blockIdx = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_blockIdx_t, align 1
@threadIdx = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_threadIdx_t, align 1

; Function Attrs: convergent mustprogress noinline norecurse nounwind
define dso_local void @dummy_func() #0 {
; CHECK-LABEL: @dummy_func
; CHECK: entry:
; CHECK-NEXT: ret void
entry:
  call void @__tb_size_marker_1D(i32 noundef 1) #3
  call void @__tb_size_marker_2D(i32 noundef 1, i32 noundef 1) #3
  call void @__tb_size_marker_3D(i32 noundef 1, i32 noundef 1, i32 noundef 1) #3
  ret void
}

; Function Attrs: convergent nounwind
declare dso_local void @__tb_size_marker_1D(i32 noundef) #1

; Function Attrs: convergent nounwind
declare dso_local void @__tb_size_marker_2D(i32 noundef, i32 noundef) #1

; Function Attrs: convergent nounwind
declare dso_local void @__tb_size_marker_3D(i32 noundef, i32 noundef, i32 noundef) #1

; Function Attrs: convergent mustprogress noinline norecurse nounwind
define dso_local void @_Z7naiveMMPKfPfi(ptr noundef %A, ptr noundef %C, i32 noundef %w) #0 {
; CHECK-LABEL: @_Z7naiveMMPKfPfi
; CHECK: entry:
; CHECK-NEXT: %C3 = addrspacecast ptr %C to ptr addrspace(1)
; CHECK-NEXT: %A1 = addrspacecast ptr %A to ptr addrspace(1)
; CHECK-NEXT: %0 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
; CHECK-NEXT: %1 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
; CHECK-NEXT: %mul = mul i32 %0, %1
; CHECK-NEXT: %2 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
; CHECK-NEXT: %add = add i32 %mul, %2
; CHECK-NEXT: br label %for.cond

; CHECK: for.cond:                                         ; preds = %for.inc12, %entry
; CHECK-NEXT: %sum.0 = phi float [ 0.000000e+00, %entry ], [ %sum.1, %for.inc12 ]
; CHECK-NEXT: %i.0 = phi i32 [ 0, %entry ], [ %inc13, %for.inc12 ]
; CHECK-NEXT: %cmp = icmp slt i32 %i.0, %w
; CHECK-NEXT: br i1 %cmp, label %for.body, label %for.end14

; CHECK: for.body:                                         ; preds = %for.cond
; CHECK-NEXT: br label %for.cond3

; CHECK: for.cond3:                                        ; preds = %for.inc, %for.body
; CHECK-NEXT: %sum.1 = phi float [ %sum.0, %for.body ], [ %add11, %for.inc ]
; CHECK-NEXT: %j.0 = phi i32 [ 0, %for.body ], [ %inc, %for.inc ]
; CHECK-NEXT: %cmp4 = icmp slt i32 %j.0, %w
; CHECK-NEXT: br i1 %cmp4, label %for.body5, label %for.end

; CHECK: for.body5:                                        ; preds = %for.cond3
; CHECK-NEXT: %3 = add i32 %i.0, %j.0
; CHECK-NEXT: %idxprom = sext i32 %3 to i64
; CHECK-NEXT: %arrayidx = getelementptr inbounds float, ptr addrspace(1) %A1, i64 %idxprom
; CHECK-NEXT: %4 = load float, ptr addrspace(1) %arrayidx, align 4
; CHECK-NEXT: %mul10 = fmul contract float %4, %4
; CHECK-NEXT: %add11 = fadd contract float %sum.1, %mul10
; CHECK-NEXT: br label %for.inc

; CHECK: for.inc:                                          ; preds = %for.body5
; CHECK-NEXT: %inc = add nsw i32 %j.0, 1
; CHECK-NEXT: br label %for.cond3, !llvm.loop !8

; CHECK: for.end:                                          ; preds = %for.cond3
; CHECK-NEXT: br label %for.inc12

; CHECK: for.inc12:                                        ; preds = %for.end
; CHECK-NEXT: %inc13 = add nsw i32 %i.0, 1
; CHECK-NEXT: br label %for.cond, !llvm.loop !10

; CHECK: for.end14:                                        ; preds = %for.cond
; CHECK-NEXT: %idxprom15 = sext i32 %add to i64
; CHECK-NEXT: %arrayidx16 = getelementptr inbounds float, ptr addrspace(1) %C3, i64 %idxprom15
; CHECK-NEXT: store float %sum.0, ptr addrspace(1) %arrayidx16, align 4
; CHECK-NEXT: ret void




entry:
  %C3 = addrspacecast ptr %C to ptr addrspace(1)
  %A1 = addrspacecast ptr %A to ptr addrspace(1)
  %0 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %1 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %mul = mul i32 %0, %1
  %2 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %add = add i32 %mul, %2
  br label %for.cond

for.cond:                                         ; preds = %for.inc12, %entry
  %sum.0 = phi float [ 0.000000e+00, %entry ], [ %sum.1, %for.inc12 ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc13, %for.inc12 ]
  %cmp = icmp slt i32 %i.0, %w
  br i1 %cmp, label %for.body, label %for.end14

for.body:                                         ; preds = %for.cond
  br label %for.cond3

for.cond3:                                        ; preds = %for.inc, %for.body
  %sum.1 = phi float [ %sum.0, %for.body ], [ %add11, %for.inc ]
  %j.0 = phi i32 [ 0, %for.body ], [ %inc, %for.inc ]
  %cmp4 = icmp slt i32 %j.0, %w
  br i1 %cmp4, label %for.body5, label %for.end

for.body5:                                        ; preds = %for.cond3
  %3 = add i32 %i.0, %j.0
  %idxprom = sext i32 %3 to i64
  %arrayidx = getelementptr inbounds float, ptr addrspace(1) %A1, i64 %idxprom
  %4 = load float, ptr addrspace(1) %arrayidx, align 4
  %mul10 = fmul contract float %4, %4
  %add11 = fadd contract float %sum.1, %mul10
  br label %for.inc

for.inc:                                          ; preds = %for.body5
  %inc = add nsw i32 %j.0, 1
  br label %for.cond3, !llvm.loop !8

for.end:                                          ; preds = %for.cond3
  br label %for.inc12

for.inc12:                                        ; preds = %for.end
  %inc13 = add nsw i32 %i.0, 1
  br label %for.cond, !llvm.loop !10

for.end14:                                        ; preds = %for.cond
  %idxprom15 = sext i32 %add to i64
  %arrayidx16 = getelementptr inbounds float, ptr addrspace(1) %C3, i64 %idxprom15
  store float %sum.0, ptr addrspace(1) %arrayidx16, align 4
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x() #2

attributes #0 = { convergent mustprogress noinline norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_35" "target-features"="+ptx75,+sm_35" "uniform-work-group-size"="true" }
attributes #1 = { convergent nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_35" "target-features"="+ptx75,+sm_35" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) "target-cpu"="sm_35" }
attributes #3 = { convergent nounwind }

!llvm.module.flags = !{!0, !1, !2, !3}
!nvvm.annotations = !{!4, !5}
!llvm.ident = !{!6, !7}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 11, i32 5]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!3 = !{i32 7, !"frame-pointer", i32 2}
!4 = !{ptr @dummy_func, !"kernel", i32 1}
!5 = !{ptr @_Z7naiveMMPKfPfi, !"kernel", i32 1}
!6 = !{!"clang version 18.1.3 (https://github.com/virnarula/llvm-project.git 7709d36816961113db93da8bb9c3a1d05b7c2c0f)"}
!7 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!8 = distinct !{!8, !9}
!9 = !{!"llvm.loop.mustprogress"}
!10 = distinct !{!10, !9}