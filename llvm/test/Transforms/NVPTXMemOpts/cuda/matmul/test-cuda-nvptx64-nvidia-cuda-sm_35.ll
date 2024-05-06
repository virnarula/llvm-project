; ModuleID = 'test-cuda-nvptx64-nvidia-cuda-sm_35.bc'
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
entry:
  %A.addr = alloca ptr, align 8
  %B.addr = alloca ptr, align 8
  %C.addr = alloca ptr, align 8
  %N.addr = alloca i32, align 4
  %col = alloca i32, align 4
  %row = alloca i32, align 4
  %sum = alloca float, align 4
  %i = alloca i32, align 4
  store ptr %A, ptr %A.addr, align 8
  store ptr %B, ptr %B.addr, align 8
  store ptr %C, ptr %C.addr, align 8
  store i32 %N, ptr %N.addr, align 4
  %0 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %1 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %mul = mul i32 %0, %1
  %2 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %add = add i32 %mul, %2
  store i32 %add, ptr %col, align 4
  %3 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %4 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
  %mul5 = mul i32 %3, %4
  %5 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %add7 = add i32 %mul5, %5
  store i32 %add7, ptr %row, align 4
  call void @__tb_size_marker_1D(i32 noundef 16) #4
  store float 0.000000e+00, ptr %sum, align 4
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %6 = load i32, ptr %i, align 4
  %7 = load i32, ptr %N.addr, align 4
  %cmp = icmp slt i32 %6, %7
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %8 = load ptr, ptr %A.addr, align 8
  %9 = load i32, ptr %row, align 4
  %10 = load i32, ptr %N.addr, align 4
  %mul8 = mul nsw i32 %9, %10
  %11 = load i32, ptr %i, align 4
  %add9 = add nsw i32 %mul8, %11
  %idxprom = sext i32 %add9 to i64
  %arrayidx = getelementptr inbounds float, ptr %8, i64 %idxprom
  %12 = load float, ptr %arrayidx, align 4
  %13 = load ptr, ptr %B.addr, align 8
  %14 = load i32, ptr %i, align 4
  %15 = load i32, ptr %N.addr, align 4
  %mul10 = mul nsw i32 %14, %15
  %16 = load i32, ptr %col, align 4
  %add11 = add nsw i32 %mul10, %16
  %idxprom12 = sext i32 %add11 to i64
  %arrayidx13 = getelementptr inbounds float, ptr %13, i64 %idxprom12
  %17 = load float, ptr %arrayidx13, align 4
  %mul14 = fmul contract float %12, %17
  %18 = load float, ptr %sum, align 4
  %add15 = fadd contract float %18, %mul14
  store float %add15, ptr %sum, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %19 = load i32, ptr %i, align 4
  %inc = add nsw i32 %19, 1
  store i32 %inc, ptr %i, align 4
  br label %for.cond, !llvm.loop !9

for.end:                                          ; preds = %for.cond
  %20 = load float, ptr %sum, align 4
  %21 = load ptr, ptr %C.addr, align 8
  %22 = load i32, ptr %row, align 4
  %23 = load i32, ptr %N.addr, align 4
  %mul16 = mul nsw i32 %22, %23
  %24 = load i32, ptr %col, align 4
  %add17 = add nsw i32 %mul16, %24
  %idxprom18 = sext i32 %add17 to i64
  %arrayidx19 = getelementptr inbounds float, ptr %21, i64 %idxprom18
  store float %20, ptr %arrayidx19, align 4
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind 
define dso_local void @_Z25matmul_coalesced_prefetchPKfS0_Pfi(ptr noundef %A, ptr noundef %B, ptr noundef %C, i32 noundef %N) #0 {
entry:
  %A.addr = alloca ptr, align 8
  %B.addr = alloca ptr, align 8
  %C.addr = alloca ptr, align 8
  %N.addr = alloca i32, align 4
  %row = alloca i32, align 4
  %col = alloca i32, align 4
  %sum = alloca float, align 4
  %tmp = alloca float, align 4
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  store ptr %A, ptr %A.addr, align 8
  store ptr %B, ptr %B.addr, align 8
  store ptr %C, ptr %C.addr, align 8
  store i32 %N, ptr %N.addr, align 4
  %0 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %1 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
  %mul = mul i32 %0, %1
  %2 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %add = add i32 %mul, %2
  store i32 %add, ptr %row, align 4
  %3 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %4 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %mul5 = mul i32 %3, %4
  %5 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %add7 = add i32 %mul5, %5
  store i32 %add7, ptr %col, align 4
  store float 0.000000e+00, ptr %sum, align 4
  %6 = load ptr, ptr %A.addr, align 8
  %7 = load i32, ptr %row, align 4
  %8 = load i32, ptr %N.addr, align 4
  %mul8 = mul nsw i32 %7, %8
  %9 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %add10 = add i32 %mul8, %9
  %idxprom = zext i32 %add10 to i64
  %arrayidx = getelementptr inbounds float, ptr %6, i64 %idxprom
  %10 = load float, ptr %arrayidx, align 4
  store float %10, ptr %tmp, align 4
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc34, %entry
  %11 = load i32, ptr %i, align 4
  %12 = load i32, ptr %N.addr, align 4
  %cmp = icmp slt i32 %11, %12
  br i1 %cmp, label %for.body, label %for.end36

for.body:                                         ; preds = %for.cond
  %13 = load float, ptr %tmp, align 4
  %14 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %idxprom12 = zext i32 %14 to i64
  %arrayidx13 = getelementptr inbounds [16 x float], ptr addrspacecast (ptr addrspace(3) @_ZZ25matmul_coalesced_prefetchPKfS0_PfiE8A_shared to ptr), i64 0, i64 %idxprom12
  store float %13, ptr %arrayidx13, align 4
  call void @llvm.nvvm.barrier0()
  %15 = load i32, ptr %i, align 4
  %add14 = add nsw i32 %15, 16
  %16 = load i32, ptr %N.addr, align 4
  %cmp15 = icmp slt i32 %add14, %16
  br i1 %cmp15, label %if.then, label %if.end

if.then:                                          ; preds = %for.body
  %17 = load ptr, ptr %A.addr, align 8
  %18 = load i32, ptr %row, align 4
  %19 = load i32, ptr %N.addr, align 4
  %mul16 = mul nsw i32 %18, %19
  %20 = load i32, ptr %i, align 4
  %add17 = add nsw i32 %mul16, %20
  %add18 = add nsw i32 %add17, 16
  %21 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %add20 = add i32 %add18, %21
  %idxprom21 = zext i32 %add20 to i64
  %arrayidx22 = getelementptr inbounds float, ptr %17, i64 %idxprom21
  %22 = load float, ptr %arrayidx22, align 4
  store float %22, ptr %tmp, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body
  store i32 0, ptr %j, align 4
  br label %for.cond23

for.cond23:                                       ; preds = %for.inc, %if.end
  %23 = load i32, ptr %j, align 4
  %cmp24 = icmp slt i32 %23, 16
  br i1 %cmp24, label %for.body25, label %for.end

for.body25:                                       ; preds = %for.cond23
  %24 = load i32, ptr %j, align 4
  %idxprom26 = sext i32 %24 to i64
  %arrayidx27 = getelementptr inbounds [16 x float], ptr addrspacecast (ptr addrspace(3) @_ZZ25matmul_coalesced_prefetchPKfS0_PfiE8A_shared to ptr), i64 0, i64 %idxprom26
  %25 = load float, ptr %arrayidx27, align 4
  %26 = load ptr, ptr %B.addr, align 8
  %27 = load i32, ptr %i, align 4
  %28 = load i32, ptr %N.addr, align 4
  %mul28 = mul nsw i32 %27, %28
  %29 = load i32, ptr %col, align 4
  %add29 = add nsw i32 %mul28, %29
  %idxprom30 = sext i32 %add29 to i64
  %arrayidx31 = getelementptr inbounds float, ptr %26, i64 %idxprom30
  %30 = load float, ptr %arrayidx31, align 4
  %mul32 = fmul contract float %25, %30
  %31 = load float, ptr %sum, align 4
  %add33 = fadd contract float %31, %mul32
  store float %add33, ptr %sum, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body25
  %32 = load i32, ptr %j, align 4
  %inc = add nsw i32 %32, 1
  store i32 %inc, ptr %j, align 4
  br label %for.cond23, !llvm.loop !11

for.end:                                          ; preds = %for.cond23
  call void @llvm.nvvm.barrier0()
  br label %for.inc34

for.inc34:                                        ; preds = %for.end
  %33 = load i32, ptr %i, align 4
  %add35 = add nsw i32 %33, 16
  store i32 %add35, ptr %i, align 4
  br label %for.cond, !llvm.loop !12

for.end36:                                        ; preds = %for.cond
  %34 = load i32, ptr %row, align 4
  %35 = load i32, ptr %N.addr, align 4
  %cmp37 = icmp slt i32 %34, %35
  br i1 %cmp37, label %land.lhs.true, label %if.end44

land.lhs.true:                                    ; preds = %for.end36
  %36 = load i32, ptr %col, align 4
  %37 = load i32, ptr %N.addr, align 4
  %cmp38 = icmp slt i32 %36, %37
  br i1 %cmp38, label %if.then39, label %if.end44

if.then39:                                        ; preds = %land.lhs.true
  %38 = load float, ptr %sum, align 4
  %39 = load ptr, ptr %C.addr, align 8
  %40 = load i32, ptr %row, align 4
  %41 = load i32, ptr %N.addr, align 4
  %mul40 = mul nsw i32 %40, %41
  %42 = load i32, ptr %col, align 4
  %add41 = add nsw i32 %mul40, %42
  %idxprom42 = sext i32 %add41 to i64
  %arrayidx43 = getelementptr inbounds float, ptr %39, i64 %idxprom42
  store float %38, ptr %arrayidx43, align 4
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
!5 = !{ptr @_Z6matmulPKfS0_Pfi, !"kernel", i32 1}
!6 = !{ptr @_Z25matmul_coalesced_prefetchPKfS0_Pfi, !"kernel", i32 1}
!7 = !{!"clang version 18.1.3 (https://github.com/virnarula/llvm-project.git 7709d36816961113db93da8bb9c3a1d05b7c2c0f)"}
!8 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!9 = distinct !{!9, !10}
!10 = !{!"llvm.loop.mustprogress"}
!11 = distinct !{!11, !10}
!12 = distinct !{!12, !10}
