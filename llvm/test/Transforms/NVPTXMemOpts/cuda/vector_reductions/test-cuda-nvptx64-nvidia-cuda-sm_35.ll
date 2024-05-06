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
@_ZZ9coalescedPfS_iiE6shared = internal addrspace(3) global [16 x float] undef, align 4

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
define dso_local void @_Z5naivePfS_ii(ptr noundef %input, ptr noundef %output, i32 noundef %row, i32 noundef %cols) #0 {
entry:
  %input.addr = alloca ptr, align 8
  %output.addr = alloca ptr, align 8
  %row.addr = alloca i32, align 4
  %cols.addr = alloca i32, align 4
  %idx = alloca i32, align 4
  %idy = alloca i32, align 4
  %sum = alloca float, align 4
  %i = alloca i32, align 4
  store ptr %input, ptr %input.addr, align 8
  store ptr %output, ptr %output.addr, align 8
  store i32 %row, ptr %row.addr, align 4
  store i32 %cols, ptr %cols.addr, align 4
  %0 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %1 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %mul = mul i32 %0, %1
  %2 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %add = add i32 %mul, %2
  store i32 %add, ptr %idx, align 4
  %3 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %4 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
  %mul5 = mul i32 %3, %4
  %5 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %add7 = add i32 %mul5, %5
  store i32 %add7, ptr %idy, align 4
  call void @__tb_size_marker_1D(i32 noundef 256) #4
  %6 = load i32, ptr %idx, align 4
  %7 = load i32, ptr %cols.addr, align 4
  %cmp = icmp slt i32 %6, %7
  br i1 %cmp, label %land.lhs.true, label %if.end

land.lhs.true:                                    ; preds = %entry
  %8 = load i32, ptr %idy, align 4
  %9 = load i32, ptr %row.addr, align 4
  %cmp8 = icmp slt i32 %8, %9
  br i1 %cmp8, label %if.then, label %if.end

if.then:                                          ; preds = %land.lhs.true
  store float 0.000000e+00, ptr %sum, align 4
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %if.then
  %10 = load i32, ptr %i, align 4
  %11 = load i32, ptr %cols.addr, align 4
  %cmp9 = icmp slt i32 %10, %11
  br i1 %cmp9, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %12 = load ptr, ptr %input.addr, align 8
  %13 = load i32, ptr %idy, align 4
  %14 = load i32, ptr %cols.addr, align 4
  %mul10 = mul nsw i32 %13, %14
  %15 = load i32, ptr %i, align 4
  %add11 = add nsw i32 %mul10, %15
  %idxprom = sext i32 %add11 to i64
  %arrayidx = getelementptr inbounds float, ptr %12, i64 %idxprom
  %16 = load float, ptr %arrayidx, align 4
  %17 = load float, ptr %sum, align 4
  %add12 = fadd contract float %17, %16
  store float %add12, ptr %sum, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %18 = load i32, ptr %i, align 4
  %inc = add nsw i32 %18, 1
  store i32 %inc, ptr %i, align 4
  br label %for.cond, !llvm.loop !9

for.end:                                          ; preds = %for.cond
  %19 = load float, ptr %sum, align 4
  %20 = load ptr, ptr %output.addr, align 8
  %21 = load i32, ptr %idx, align 4
  %idxprom13 = sext i32 %21 to i64
  %arrayidx14 = getelementptr inbounds float, ptr %20, i64 %idxprom13
  store float %19, ptr %arrayidx14, align 4
  br label %if.end

if.end:                                           ; preds = %for.end, %land.lhs.true, %entry
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind 
define dso_local void @_Z9coalescedPfS_ii(ptr noundef %input, ptr noundef %output, i32 noundef %row, i32 noundef %cols) #0 {
entry:
  %input.addr = alloca ptr, align 8
  %output.addr = alloca ptr, align 8
  %row.addr = alloca i32, align 4
  %cols.addr = alloca i32, align 4
  %idx = alloca i32, align 4
  %idy = alloca i32, align 4
  %sum = alloca float, align 4
  %i = alloca i32, align 4
  %k = alloca i32, align 4
  store ptr %input, ptr %input.addr, align 8
  store ptr %output, ptr %output.addr, align 8
  store i32 %row, ptr %row.addr, align 4
  store i32 %cols, ptr %cols.addr, align 4
  %0 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %1 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %mul = mul i32 %0, %1
  %2 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %add = add i32 %mul, %2
  store i32 %add, ptr %idx, align 4
  %3 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %4 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
  %mul5 = mul i32 %3, %4
  %5 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %add7 = add i32 %mul5, %5
  store i32 %add7, ptr %idy, align 4
  store float 0.000000e+00, ptr %sum, align 4
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc21, %entry
  %6 = load i32, ptr %i, align 4
  %7 = load i32, ptr %cols.addr, align 4
  %cmp = icmp slt i32 %6, %7
  br i1 %cmp, label %for.body, label %for.end23

for.body:                                         ; preds = %for.cond
  %8 = load ptr, ptr %input.addr, align 8
  %9 = load i32, ptr %idy, align 4
  %10 = load i32, ptr %cols.addr, align 4
  %mul8 = mul nsw i32 %9, %10
  %11 = load i32, ptr %i, align 4
  %add9 = add nsw i32 %mul8, %11
  %12 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %add11 = add i32 %add9, %12
  %idxprom = zext i32 %add11 to i64
  %arrayidx = getelementptr inbounds float, ptr %8, i64 %idxprom
  %13 = load float, ptr %arrayidx, align 4
  %14 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %idxprom13 = zext i32 %14 to i64
  %arrayidx14 = getelementptr inbounds [16 x float], ptr addrspacecast (ptr addrspace(3) @_ZZ9coalescedPfS_iiE6shared to ptr), i64 0, i64 %idxprom13
  store float %13, ptr %arrayidx14, align 4
  call void @llvm.nvvm.barrier0()
  store i32 0, ptr %k, align 4
  br label %for.cond15

for.cond15:                                       ; preds = %for.inc, %for.body
  %15 = load i32, ptr %k, align 4
  %cmp16 = icmp slt i32 %15, 16
  br i1 %cmp16, label %for.body17, label %for.end

for.body17:                                       ; preds = %for.cond15
  %16 = load i32, ptr %k, align 4
  %idxprom18 = sext i32 %16 to i64
  %arrayidx19 = getelementptr inbounds [16 x float], ptr addrspacecast (ptr addrspace(3) @_ZZ9coalescedPfS_iiE6shared to ptr), i64 0, i64 %idxprom18
  %17 = load float, ptr %arrayidx19, align 4
  %18 = load float, ptr %sum, align 4
  %add20 = fadd contract float %18, %17
  store float %add20, ptr %sum, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body17
  %19 = load i32, ptr %k, align 4
  %inc = add nsw i32 %19, 1
  store i32 %inc, ptr %k, align 4
  br label %for.cond15, !llvm.loop !11

for.end:                                          ; preds = %for.cond15
  call void @llvm.nvvm.barrier0()
  br label %for.inc21

for.inc21:                                        ; preds = %for.end
  %20 = load i32, ptr %i, align 4
  %add22 = add nsw i32 %20, 16
  store i32 %add22, ptr %i, align 4
  br label %for.cond, !llvm.loop !12

for.end23:                                        ; preds = %for.cond
  %21 = load i32, ptr %idx, align 4
  %22 = load i32, ptr %cols.addr, align 4
  %cmp24 = icmp slt i32 %21, %22
  br i1 %cmp24, label %land.lhs.true, label %if.end

land.lhs.true:                                    ; preds = %for.end23
  %23 = load i32, ptr %idy, align 4
  %24 = load i32, ptr %row.addr, align 4
  %cmp25 = icmp slt i32 %23, %24
  br i1 %cmp25, label %if.then, label %if.end

if.then:                                          ; preds = %land.lhs.true
  %25 = load float, ptr %sum, align 4
  %26 = load ptr, ptr %output.addr, align 8
  %27 = load i32, ptr %idx, align 4
  %idxprom26 = sext i32 %27 to i64
  %arrayidx27 = getelementptr inbounds float, ptr %26, i64 %idxprom26
  store float %25, ptr %arrayidx27, align 4
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
!5 = !{ptr @_Z5naivePfS_ii, !"kernel", i32 1}
!6 = !{ptr @_Z9coalescedPfS_ii, !"kernel", i32 1}
!7 = !{!"clang version 18.1.3 (https://github.com/virnarula/llvm-project.git 7709d36816961113db93da8bb9c3a1d05b7c2c0f)"}
!8 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!9 = distinct !{!9, !10}
!10 = !{!"llvm.loop.mustprogress"}
!11 = distinct !{!11, !10}
!12 = distinct !{!12, !10}
