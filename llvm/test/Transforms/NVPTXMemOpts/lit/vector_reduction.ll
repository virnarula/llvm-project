; ModuleID = 'test-cuda-nvptx64-nvidia-cuda-sm_35.ll'
source_filename = "test.cu"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%struct.__cuda_builtin_blockIdx_t = type { i8 }
%struct.__cuda_builtin_blockDim_t = type { i8 }
%struct.__cuda_builtin_threadIdx_t = type { i8 }

@blockIdx = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_blockIdx_t, align 1
@blockDim = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_blockDim_t, align 1
@threadIdx = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_threadIdx_t, align 1
@_ZZ9coalescedPfS_iiE6shared = internal addrspace(3) global [16 x float] undef, align 4

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
define dso_local void @_Z5naivePfS_ii(ptr noundef %input, ptr noundef %output, i32 noundef %row, i32 noundef %cols) #0 {
; CHECK-LABEL: @_Z5naivePfS_ii
; CHECK: if.then:  
; CHECK-NEXT: %6 = mul i32 %cols, %add7
; CHECK-NEXT: %prefetch_add = add i32 %6, %2
; CHECK-NEXT: %7 = sext i32 %prefetch_add to i64
; CHECK-NEXT: %8 = getelementptr inbounds float, ptr addrspace(1) %input1, i64 %7
; CHECK-NEXT: %9 = load float, ptr addrspace(1) %8, align 4
; CHECK-NEXT: br label %for.cond

; CHECK: prefetch:
; CHECK-NEXT: %next_prefetch_add_threadidx = add i32 %next_prefetch_add, %2
; CHECK: %11 = getelementptr inbounds float, ptr addrspace(1) %input1, i64 %10
; CHECK-NEXT: %12 = load float, ptr addrspace(1) %11, align 4
; CHECK-NEXT: br label %inner.header

; CHECK: shared_memory:
; CHECK-NEXT: %new_add = add i32 %6, %new_indvar
; CHECK-NEXT: %new_add_threadidx = add i32 %new_add, %2
; CHECK-NEXT: %13 = sext i32 %new_add_threadidx to i64
; CHECK-NEXT: %14 = getelementptr inbounds float, ptr addrspace(1) %input1, i64 %13
; CHECK-NEXT: %15 = load float, ptr addrspace(1) %14, align 4
; CHECK-NEXT: %16 = sext i32 %2 to i64
; CHECK-NEXT: %17 = getelementptr inbounds [256 x float], ptr addrspace(3) @_Z5naivePfS_iiinput_shared, i64 0, i64 %16
; CHECK-NEXT: store float %15, ptr addrspace(3) %17, align 4
; CHECK-NEXT: call void @llvm.nvvm.barrier0()
; CHECK-NEXT: %next_prefetch_add = add i32 %new_add, 256
; CHECK-NEXT: %next_prefetch_slt = icmp slt i32 %next_prefetch_add, %cols
; CHECK-NEXT: br i1 %next_prefetch_slt, label %prefetch, label %inner.header

; CHECK: inner.latch:                                      ; preds = %for.body
; CHECK-NEXT: %inner_inc = add i32 %inner_indvar, 1
; CHECK-NEXT: br label %inner.header

; CHECK: inner.body:                                       ; preds = %inner.header
; CHECK-NEXT: %new_indvar_value = add i32 %i.0, %inner_indvar
; CHECK-NEXT: br label %for.body

; CHECK: inner.header:                                     ; preds = %prefetch, %shared_memory, %inner.latch
; CHECK-NEXT: %inner_indvar = phi i32 [ 0, %shared_memory ], [ %inner_inc, %inner.latch ], [ 0, %prefetch ]
; CHECK-NEXT: %new_phi = phi float [ %sum.0, %shared_memory ], [ %add12, %inner.latch ], [ %sum.0, %prefetch ]
; CHECK-NEXT: %prefetch_phi_update = phi float [ %prefetch_phi, %shared_memory ], [ %12, %prefetch ], [ %prefetch_phi_update, %inner.latch ]
; CHECK-NEXT: %inner_cond = icmp slt i32 %inner_indvar, 256
; CHECK-NEXT: br i1 %inner_cond, label %inner.body, label %for.inc

entry:
  %output3 = addrspacecast ptr %output to ptr addrspace(1)
  %input1 = addrspacecast ptr %input to ptr addrspace(1)
  %0 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %1 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %mul = mul i32 %0, %1
  %2 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %add = add i32 %mul, %2
  %3 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %4 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
  %mul5 = mul i32 %3, %4
  %5 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %add7 = add i32 %mul5, %5
  call void @__tb_size_marker_1D(i32 noundef 256) #4
  %cmp = icmp slt i32 %add, %cols
  br i1 %cmp, label %land.lhs.true, label %if.end

land.lhs.true:                                    ; preds = %entry
  %cmp8 = icmp slt i32 %add7, %row
  br i1 %cmp8, label %if.then, label %if.end

if.then:                                          ; preds = %land.lhs.true
  %6 = mul i32 %cols, %add7
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %if.then
  %sum.0 = phi float [ 0.000000e+00, %if.then ], [ %add12, %for.inc ]
  %i.0 = phi i32 [ 0, %if.then ], [ %inc, %for.inc ]
  %cmp9 = icmp slt i32 %i.0, %cols
  br i1 %cmp9, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %7 = add i32 %6, %i.0
  %idxprom = sext i32 %7 to i64
  %arrayidx = getelementptr inbounds float, ptr addrspace(1) %input1, i64 %idxprom
  %8 = load float, ptr addrspace(1) %arrayidx, align 4
  %add12 = fadd contract float %sum.0, %8
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add nsw i32 %i.0, 1
  br label %for.cond, !llvm.loop !9

for.end:                                          ; preds = %for.cond
  %idxprom13 = sext i32 %add to i64
  %arrayidx14 = getelementptr inbounds float, ptr addrspace(1) %output3, i64 %idxprom13
  store float %sum.0, ptr addrspace(1) %arrayidx14, align 4
  br label %if.end

if.end:                                           ; preds = %for.end, %land.lhs.true, %entry
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind
define dso_local void @_Z9coalescedPfS_ii(ptr noundef %input, ptr noundef %output, i32 noundef %row, i32 noundef %cols) #0 {
entry:
  %output3 = addrspacecast ptr %output to ptr addrspace(1)
  %input1 = addrspacecast ptr %input to ptr addrspace(1)
  %0 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %1 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %mul = mul i32 %0, %1
  %2 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %add = add i32 %mul, %2
  %3 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %4 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
  %mul5 = mul i32 %3, %4
  %5 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %add7 = add i32 %mul5, %5
  br label %for.cond

for.cond:                                         ; preds = %for.inc21, %entry
  %sum.0 = phi float [ 0.000000e+00, %entry ], [ %sum.1, %for.inc21 ]
  %i.0 = phi i32 [ 0, %entry ], [ %add22, %for.inc21 ]
  %cmp = icmp slt i32 %i.0, %cols
  br i1 %cmp, label %for.body, label %for.end23

for.body:                                         ; preds = %for.cond
  %mul8 = mul nsw i32 %add7, %cols
  %add9 = add nsw i32 %mul8, %i.0
  %add11 = add i32 %add9, %2
  %idxprom = zext i32 %add11 to i64
  %arrayidx = getelementptr inbounds float, ptr addrspace(1) %input1, i64 %idxprom
  %6 = load float, ptr addrspace(1) %arrayidx, align 4
  %idxprom13 = zext i32 %2 to i64
  %arrayidx14 = getelementptr inbounds [16 x float], ptr addrspace(3) @_ZZ9coalescedPfS_iiE6shared, i64 0, i64 %idxprom13
  store float %6, ptr addrspace(3) %arrayidx14, align 4
  call void @llvm.nvvm.barrier0()
  br label %for.cond15

for.cond15:                                       ; preds = %for.inc, %for.body
  %lsr.iv = phi ptr addrspace(3) [ %scevgep, %for.inc ], [ @_ZZ9coalescedPfS_iiE6shared, %for.body ]
  %sum.1 = phi float [ %sum.0, %for.body ], [ %add20, %for.inc ]
  %k.0 = phi i32 [ 0, %for.body ], [ %inc, %for.inc ]
  %cmp16 = icmp slt i32 %k.0, 16
  br i1 %cmp16, label %for.body17, label %for.end

for.body17:                                       ; preds = %for.cond15
  %7 = load float, ptr addrspace(3) %lsr.iv, align 4
  %add20 = fadd contract float %sum.1, %7
  br label %for.inc

for.inc:                                          ; preds = %for.body17
  %inc = add nsw i32 %k.0, 1
  %scevgep = getelementptr i8, ptr addrspace(3) %lsr.iv, i64 4
  br label %for.cond15, !llvm.loop !11

for.end:                                          ; preds = %for.cond15
  call void @llvm.nvvm.barrier0()
  br label %for.inc21

for.inc21:                                        ; preds = %for.end
  %add22 = add nsw i32 %i.0, 16
  br label %for.cond, !llvm.loop !12

for.end23:                                        ; preds = %for.cond
  %cmp24 = icmp slt i32 %add, %cols
  br i1 %cmp24, label %land.lhs.true, label %if.end

land.lhs.true:                                    ; preds = %for.end23
  %cmp25 = icmp slt i32 %add7, %row
  %idxprom26 = sext i32 %add to i64
  %arrayidx27 = getelementptr inbounds float, ptr addrspace(1) %output3, i64 %idxprom26
  br i1 %cmp25, label %if.then, label %if.end

if.then:                                          ; preds = %land.lhs.true
  store float %sum.0, ptr addrspace(1) %arrayidx27, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %land.lhs.true, %for.end23
  ret void
}

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier0() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y() #3

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
!5 = !{ptr @_Z5naivePfS_ii, !"kernel", i32 1}
!6 = !{ptr @_Z9coalescedPfS_ii, !"kernel", i32 1}
!7 = !{!"clang version 18.1.3 (https://github.com/virnarula/llvm-project.git 7709d36816961113db93da8bb9c3a1d05b7c2c0f)"}
!8 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!9 = distinct !{!9, !10}
!10 = !{!"llvm.loop.mustprogress"}
!11 = distinct !{!11, !10}
!12 = distinct !{!12, !10}