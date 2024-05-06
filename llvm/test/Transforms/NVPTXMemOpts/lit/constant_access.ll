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
; CHECK: entry
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
define dso_local void @_Z7naiveMMPPKfS1_PPfi(ptr noundef %A, ptr noundef %B, ptr noundef %C, i32 noundef %w) #0 {;
; CHECK: @_Z7naiveMMPPKfS1_PPfi
; CHECK: entry:
; CHECK-NOT: call void @__tb_size_marker_
; CHECK: br label %for.cond
entry:
  %C5 = addrspacecast ptr %C to ptr addrspace(1)
  %B3 = addrspacecast ptr %B to ptr addrspace(1)
  %A1 = addrspacecast ptr %A to ptr addrspace(1)
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
  call void @__tb_size_marker_2D(i32 noundef 128, i32 noundef 43) #3
  br label %for.cond

for.cond:                                         ; preds = %for.inc16, %entry
  %sum.0 = phi float [ 0.000000e+00, %entry ], [ %sum.1, %for.inc16 ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc17, %for.inc16 ]
  %cmp = icmp slt i32 %i.0, %w
  br i1 %cmp, label %for.body, label %for.end18

for.body:                                         ; preds = %for.cond
  br label %for.cond8

for.cond8:                                        ; preds = %for.inc, %for.body
  %sum.1 = phi float [ %sum.0, %for.body ], [ %add15, %for.inc ]
  %j.0 = phi i32 [ 0, %for.body ], [ %inc, %for.inc ]
  %cmp9 = icmp slt i32 %j.0, %w
  br i1 %cmp9, label %for.body10, label %for.end

for.body10:                                       ; preds = %for.cond8
  %arrayidx = getelementptr inbounds ptr, ptr addrspace(1) %A1, i64 10
  %6 = load ptr, ptr addrspace(1) %arrayidx, align 8
  %arrayidx11 = getelementptr inbounds float, ptr %6, i64 15
  %7 = load float, ptr %arrayidx11, align 4
  %arrayidx12 = getelementptr inbounds ptr, ptr addrspace(1) %B3, i64 23
  %8 = load ptr, ptr addrspace(1) %arrayidx12, align 8
  %arrayidx13 = getelementptr inbounds float, ptr %8, i64 54
  %9 = load float, ptr %arrayidx13, align 4
  %mul14 = fmul contract float %7, %9
  %add15 = fadd contract float %sum.1, %mul14
  br label %for.inc

for.inc:                                          ; preds = %for.body10
  %inc = add nsw i32 %j.0, 1
  br label %for.cond8, !llvm.loop !8

for.end:                                          ; preds = %for.cond8
  br label %for.inc16

for.inc16:                                        ; preds = %for.end
  %inc17 = add nsw i32 %i.0, 1
  br label %for.cond, !llvm.loop !10

for.end18:                                        ; preds = %for.cond
  %idxprom = sext i32 %add7 to i64
  %arrayidx19 = getelementptr inbounds ptr, ptr addrspace(1) %C5, i64 %idxprom
  %10 = load ptr, ptr addrspace(1) %arrayidx19, align 8
  %idxprom20 = sext i32 %add to i64
  %arrayidx21 = getelementptr inbounds float, ptr %10, i64 %idxprom20
  store float %sum.0, ptr %arrayidx21, align 4
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y() #2

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
!5 = !{ptr @_Z7naiveMMPPKfS1_PPfi, !"kernel", i32 1}
!6 = !{!"clang version 18.1.3 (https://github.com/virnarula/llvm-project.git 7709d36816961113db93da8bb9c3a1d05b7c2c0f)"}
!7 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!8 = distinct !{!8, !9}
!9 = !{!"llvm.loop.mustprogress"}
!10 = distinct !{!10, !9}