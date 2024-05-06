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
@_ZZ25matmul_coalesced_prefetchPKfS0_PfiE8A_shared = internal addrspace(3) global [16 x float] undef, align 4

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
define dso_local void @_Z6matmulPKfS0_Pfi(ptr noundef %A, ptr noundef %B, ptr noundef %C, i32 noundef %N) #0 {
; CHECK: entry:
; CHECK-NEXT: %C5 = addrspacecast ptr %C to ptr addrspace(1)
; CHECK-NEXT: %B3 = addrspacecast ptr %B to ptr addrspace(1)
; CHECK-NEXT: %A1 = addrspacecast ptr %A to ptr addrspace(1)
; CHECK-NEXT: %0 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
; CHECK-NEXT: %1 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
; CHECK-NEXT: %mul = mul i32 %0, %1
; CHECK-NEXT: %2 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
; CHECK-NEXT: %add = add i32 %mul, %2
; CHECK-NEXT: %3 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
; CHECK-NEXT: %4 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
; CHECK-NEXT: %mul5 = mul i32 %3, %4
; CHECK-NEXT: %5 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
; CHECK-NEXT: %add7 = add i32 %mul5, %5
; CHECK-NEXT: %6 = mul i32 %N, %add7
; CHECK-NEXT: %prefetch_add = add i32 %6, %2
; CHECK-NEXT: %7 = sext i32 %prefetch_add to i64
; CHECK-NEXT: %8 = getelementptr inbounds float, ptr addrspace(1) %A1, i64 %7
; CHECK-NEXT: %9 = load float, ptr addrspace(1) %8, align 4
; CHECK-NEXT: br label %for.cond
; CHECK: prefetch:
; CHECK-NEXT: %next_prefetch_add_threadidx = add i32 %next_prefetch_add, %2
; CHECK: %11 = getelementptr inbounds float, ptr addrspace(1) %A1, i64 %10
; CHECK-NEXT: %12 = load float, ptr addrspace(1) %11, align 4
; CHECK-NEXT: br label %inner.header
; CHECK: shared_memory:                                   
; CHECK-NEXT: %new_add = add i32 %6, %new_indvar
; CHECK-NEXT: %new_add_threadidx = add i32 %new_add, %2
; CHECK-NEXT: %13 = sext i32 %new_add_threadidx to i64
; CHECK-NEXT: %14 = getelementptr inbounds float, ptr addrspace(1) %A1, i64 %13
; CHECK-NEXT: %15 = load float, ptr addrspace(1) %14, align 4
; CHECK-NEXT: %16 = sext i32 %2 to i64
; CHECK-NEXT: %17 = getelementptr inbounds [16 x float], ptr addrspace(3) @_Z6matmulPKfS0_Pfiinput_shared, i64 0, i64 %16
; CHECK-NEXT: store float %15, ptr addrspace(3) %17, align 4
; CHECK-NEXT: call void @llvm.nvvm.barrier0()
; CHECK-NEXT: %next_prefetch_add = add i32 %new_add, 16
; CHECK-NEXT: %next_prefetch_slt = icmp slt i32 %next_prefetch_add, %N
; CHECK-NEXT: br i1 %next_prefetch_slt, label %prefetch, label %inner.header
; CHECK: inner.latch:                                     
; CHECK-NEXT: %inner_inc = add i32 %inner_indvar, 1
; CHECK-NEXT: br label %inner.header
; CHECK: inner.body:                                     
; CHECK: br label %for.body
; CHECK: inner.header:                                    
; CHECK-NEXT: %inner_indvar = phi i32 [ 0, %shared_memory ], [ %inner_inc, %inner.latch ], [ 0, %prefetch ]
; CHECK-NEXT: %new_phi = phi float [ %sum.0, %shared_memory ], [ %add15, %inner.latch ], [ %sum.0, %prefetch ]
; CHECK-NEXT: %prefetch_phi_update = phi float [ %prefetch_phi, %shared_memory ], [ %12, %prefetch ], [ %prefetch_phi_update, %inner.latch ]
; CHECK-NEXT: %inner_cond = icmp slt i32 %inner_indvar, 16
; CHECK-NEXT: br i1 %inner_cond, label %inner.body, label %for.inc
; CHECK: for.cond:                                         ; preds = %for.inc, %entry
; CHECK-NEXT: %lsr.iv = phi i32 [ %lsr.iv.next, %for.inc ], [ %add, %entry ]
; CHECK-NEXT: %sum.0 = phi float [ 0.000000e+00, %entry ], [ %new_phi, %for.inc ]
; CHECK-NEXT: %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
; CHECK-NEXT: %new_indvar = phi i32 [ 0, %entry ], [ %new_inc, %for.inc ]
; CHECK-NEXT: %prefetch_phi = phi float [ %9, %entry ], [ %prefetch_phi_update, %for.inc ]
; CHECK: br i1 %new_cond, label %shared_memory, label %for.end
; CHECK: for.body:                                         ; preds = %inner.body
; CHECK-NEXT: %18 = sext i32 %inner_indvar to i64
; CHECK-NEXT: %19 = getelementptr inbounds [16 x float], ptr addrspace(3) @_Z6matmulPKfS0_Pfiinput_shared, i64 %18
; CHECK-NEXT: %20 = load float, ptr addrspace(3) %19, align 4
; CHECK: %add15 = fadd contract float %sum.0, %mul14
; CHECK-NEXT: br label %inner.latch
; CHECK: for.inc:                                          ; preds = %inner.header
; CHECK-NEXT: call void @llvm.nvvm.barrier0()
; CHECK-NEXT: %inc = add nsw i32 %i.0, 1
; CHECK-NEXT: %lsr.iv.next = add i32 %lsr.iv, %N
; CHECK-NEXT: %new_inc = add i32 %new_indvar, 16
; CHECK-NEXT: br label %for.cond, !llvm.loop !9
; CHECK: for.end:                                          ; preds = %for.cond
; CHECK-NEXT: %add17 = add nsw i32 %6, %add
; CHECK-NEXT: %idxprom18 = sext i32 %add17 to i64
; CHECK-NEXT: %arrayidx19 = getelementptr inbounds float, ptr addrspace(1) %C5, i64 %idxprom18
; CHECK-NEXT: store float %sum.0, ptr addrspace(1) %arrayidx19, align 4
; CHECK-NEXT: ret void

entry:
  %C5 = addrspacecast ptr %C to ptr addrspace(1)
  %B3 = addrspacecast ptr %B to ptr addrspace(1)
  %A1 = addrspacecast ptr %A to ptr addrspace(1)
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
  call void @__tb_size_marker_1D(i32 noundef 16) #4
  %6 = mul i32 %N, %add7
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %lsr.iv = phi i32 [ %lsr.iv.next, %for.inc ], [ %add, %entry ]
  %sum.0 = phi float [ 0.000000e+00, %entry ], [ %add15, %for.inc ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i.0, %N
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %7 = add i32 %6, %i.0
  %idxprom = sext i32 %7 to i64
  %arrayidx = getelementptr inbounds float, ptr addrspace(1) %A1, i64 %idxprom
  %8 = load float, ptr addrspace(1) %arrayidx, align 4
  %idxprom12 = sext i32 %lsr.iv to i64
  %arrayidx13 = getelementptr inbounds float, ptr addrspace(1) %B3, i64 %idxprom12
  %9 = load float, ptr addrspace(1) %arrayidx13, align 4
  %mul14 = fmul contract float %8, %9
  %add15 = fadd contract float %sum.0, %mul14
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add nsw i32 %i.0, 1
  %lsr.iv.next = add i32 %lsr.iv, %N
  br label %for.cond, !llvm.loop !9

for.end:                                          ; preds = %for.cond
  %add17 = add nsw i32 %6, %add
  %idxprom18 = sext i32 %add17 to i64
  %arrayidx19 = getelementptr inbounds float, ptr addrspace(1) %C5, i64 %idxprom18
  store float %sum.0, ptr addrspace(1) %arrayidx19, align 4
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind
define dso_local void @_Z25matmul_coalesced_prefetchPKfS0_Pfi(ptr noundef %A, ptr noundef %B, ptr noundef %C, i32 noundef %N) #0 {
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
  %mul8 = mul nsw i32 %add, %N
  %add10 = add i32 %mul8, %5
  %idxprom = zext i32 %add10 to i64
  %arrayidx = getelementptr inbounds float, ptr addrspace(1) %A1, i64 %idxprom
  %6 = load float, ptr addrspace(1) %arrayidx, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc34, %entry
  %sum.0 = phi float [ 0.000000e+00, %entry ], [ %sum.1, %for.inc34 ]
  %tmp.0 = phi float [ %6, %entry ], [ %tmp.1, %for.inc34 ]
  %i.0 = phi i32 [ 0, %entry ], [ %add14, %for.inc34 ]
  %cmp = icmp slt i32 %i.0, %N
  br i1 %cmp, label %for.body, label %for.end36

for.body:                                         ; preds = %for.cond
  %idxprom12 = zext i32 %5 to i64
  %arrayidx13 = getelementptr inbounds [16 x float], ptr addrspace(3) @_ZZ25matmul_coalesced_prefetchPKfS0_PfiE8A_shared, i64 0, i64 %idxprom12
  store float %tmp.0, ptr addrspace(3) %arrayidx13, align 4
  call void @llvm.nvvm.barrier0()
  %add14 = add nsw i32 %i.0, 16
  %cmp15 = icmp slt i32 %add14, %N
  %add20 = add i32 %add10, %add14
  %idxprom21 = zext i32 %add20 to i64
  %arrayidx22 = getelementptr inbounds float, ptr addrspace(1) %A1, i64 %idxprom21
  br i1 %cmp15, label %if.then, label %if.end

if.then:                                          ; preds = %for.body
  %7 = load float, ptr addrspace(1) %arrayidx22, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body
  %tmp.1 = phi float [ %7, %if.then ], [ %tmp.0, %for.body ]
  br label %for.cond23

for.cond23:                                       ; preds = %for.inc, %if.end
  %lsr.iv = phi ptr addrspace(3) [ %scevgep, %for.inc ], [ @_ZZ25matmul_coalesced_prefetchPKfS0_PfiE8A_shared, %if.end ]
  %sum.1 = phi float [ %sum.0, %if.end ], [ %add33, %for.inc ]
  %j.0 = phi i32 [ 0, %if.end ], [ %inc, %for.inc ]
  %cmp24 = icmp slt i32 %j.0, 16
  br i1 %cmp24, label %for.body25, label %for.end

for.body25:                                       ; preds = %for.cond23
  %8 = load float, ptr addrspace(3) %lsr.iv, align 4
  %mul28 = mul nsw i32 %i.0, %N
  %add29 = add nsw i32 %mul28, %add7
  %idxprom30 = sext i32 %add29 to i64
  %arrayidx31 = getelementptr inbounds float, ptr addrspace(1) %B3, i64 %idxprom30
  %9 = load float, ptr addrspace(1) %arrayidx31, align 4
  %mul32 = fmul contract float %8, %9
  %add33 = fadd contract float %sum.1, %mul32
  br label %for.inc

for.inc:                                          ; preds = %for.body25
  %inc = add nsw i32 %j.0, 1
  %scevgep = getelementptr i8, ptr addrspace(3) %lsr.iv, i64 4
  br label %for.cond23, !llvm.loop !11

for.end:                                          ; preds = %for.cond23
  call void @llvm.nvvm.barrier0()
  br label %for.inc34

for.inc34:                                        ; preds = %for.end
  br label %for.cond, !llvm.loop !12

for.end36:                                        ; preds = %for.cond
  %cmp37 = icmp slt i32 %add, %N
  br i1 %cmp37, label %land.lhs.true, label %if.end44

land.lhs.true:                                    ; preds = %for.end36
  %cmp38 = icmp slt i32 %add7, %N
  %add41 = add nsw i32 %mul8, %add7
  %idxprom42 = sext i32 %add41 to i64
  %arrayidx43 = getelementptr inbounds float, ptr addrspace(1) %C5, i64 %idxprom42
  br i1 %cmp38, label %if.then39, label %if.end44

if.then39:                                        ; preds = %land.lhs.true
  store float %sum.0, ptr addrspace(1) %arrayidx43, align 4
  br label %if.end44

if.end44:                                         ; preds = %if.then39, %land.lhs.true, %for.end36
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
!5 = !{ptr @_Z6matmulPKfS0_Pfi, !"kernel", i32 1}
!6 = !{ptr @_Z25matmul_coalesced_prefetchPKfS0_Pfi, !"kernel", i32 1}
!7 = !{!"clang version 18.1.3 (https://github.com/virnarula/llvm-project.git 7709d36816961113db93da8bb9c3a1d05b7c2c0f)"}
!8 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!9 = distinct !{!9, !10}
!10 = !{!"llvm.loop.mustprogress"}
!11 = distinct !{!11, !10}
!12 = distinct !{!12, !10}