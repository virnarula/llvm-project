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
@_ZZ19stencil1d_coalescediPfS_E12input_shared = internal addrspace(3) global [8 x [16 x float]] undef, align 4

; Function Attrs: convergent mustprogress noinline norecurse nounwind 
define dso_local void @dummy_func() #0 {
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
define dso_local void @_Z9stencil1diPfS_(i32 noundef %n, ptr noundef %input, ptr noundef %output) #0 {
entry:
  %n.addr = alloca i32, align 4
  %input.addr = alloca ptr, align 8
  %output.addr = alloca ptr, align 8
  %idx = alloca i32, align 4
  %result = alloca float, align 4
  %i = alloca i32, align 4
  store i32 %n, ptr %n.addr, align 4
  store ptr %input, ptr %input.addr, align 8
  store ptr %output, ptr %output.addr, align 8
  %0 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %1 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %mul = mul i32 %0, %1
  %2 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %add = add i32 %mul, %2
  store i32 %add, ptr %idx, align 4
  %3 = load i32, ptr %idx, align 4
  %cmp = icmp sgt i32 %3, 4
  br i1 %cmp, label %land.lhs.true, label %if.end

land.lhs.true:                                    ; preds = %entry
  %4 = load i32, ptr %idx, align 4
  %5 = load i32, ptr %n.addr, align 4
  %sub = sub nsw i32 %5, 5
  %cmp3 = icmp slt i32 %4, %sub
  br i1 %cmp3, label %if.then, label %if.end

if.then:                                          ; preds = %land.lhs.true
  store float 0.000000e+00, ptr %result, align 4
  store i32 -4, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %if.then
  %6 = load i32, ptr %i, align 4
  %cmp4 = icmp slt i32 %6, 4
  br i1 %cmp4, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %7 = load ptr, ptr %input.addr, align 8
  %8 = load i32, ptr %idx, align 4
  %9 = load i32, ptr %i, align 4
  %add5 = add nsw i32 %8, %9
  %idxprom = sext i32 %add5 to i64
  %arrayidx = getelementptr inbounds float, ptr %7, i64 %idxprom
  %10 = load float, ptr %arrayidx, align 4
  %11 = load float, ptr %result, align 4
  %add6 = fadd contract float %11, %10
  store float %add6, ptr %result, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %12 = load i32, ptr %i, align 4
  %inc = add nsw i32 %12, 1
  store i32 %inc, ptr %i, align 4
  br label %for.cond, !llvm.loop !9

for.end:                                          ; preds = %for.cond
  %13 = load float, ptr %result, align 4
  %div = fdiv contract float %13, 7.000000e+00
  %14 = load ptr, ptr %output.addr, align 8
  %15 = load i32, ptr %idx, align 4
  %idxprom7 = sext i32 %15 to i64
  %arrayidx8 = getelementptr inbounds float, ptr %14, i64 %idxprom7
  store float %div, ptr %arrayidx8, align 4
  br label %if.end

if.end:                                           ; preds = %for.end, %land.lhs.true, %entry
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind 
define dso_local void @_Z19stencil1d_coalescediPfS_(i32 noundef %n, ptr noundef %input, ptr noundef %output) #0 {
entry:
  %n.addr = alloca i32, align 4
  %input.addr = alloca ptr, align 8
  %output.addr = alloca ptr, align 8
  %idx = alloca i32, align 4
  %result = alloca float, align 4
  %i = alloca i32, align 4
  %i12 = alloca i32, align 4
  store i32 %n, ptr %n.addr, align 4
  store ptr %input, ptr %input.addr, align 8
  store ptr %output, ptr %output.addr, align 8
  %0 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %1 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %mul = mul i32 %0, %1
  %2 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %add = add i32 %mul, %2
  store i32 %add, ptr %idx, align 4
  %3 = load i32, ptr %idx, align 4
  %cmp = icmp sgt i32 %3, 4
  br i1 %cmp, label %land.lhs.true, label %if.end

land.lhs.true:                                    ; preds = %entry
  %4 = load i32, ptr %idx, align 4
  %5 = load i32, ptr %n.addr, align 4
  %sub = sub nsw i32 %5, 5
  %cmp3 = icmp slt i32 %4, %sub
  br i1 %cmp3, label %if.then, label %if.end

if.then:                                          ; preds = %land.lhs.true
  store float 0.000000e+00, ptr %result, align 4
  store i32 -4, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %if.then
  %6 = load i32, ptr %i, align 4
  %cmp4 = icmp slt i32 %6, 4
  br i1 %cmp4, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %7 = load ptr, ptr %input.addr, align 8
  %8 = load i32, ptr %idx, align 4
  %9 = load i32, ptr %i, align 4
  %add5 = add nsw i32 %8, %9
  %idxprom = sext i32 %add5 to i64
  %arrayidx = getelementptr inbounds float, ptr %7, i64 %idxprom
  %10 = load float, ptr %arrayidx, align 4
  %11 = load i32, ptr %i, align 4
  %add6 = add nsw i32 %11, 4
  %idxprom7 = sext i32 %add6 to i64
  %arrayidx8 = getelementptr inbounds [8 x [16 x float]], ptr addrspacecast (ptr addrspace(3) @_ZZ19stencil1d_coalescediPfS_E12input_shared to ptr), i64 0, i64 %idxprom7
  %12 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %idxprom10 = zext i32 %12 to i64
  %arrayidx11 = getelementptr inbounds [16 x float], ptr %arrayidx8, i64 0, i64 %idxprom10
  store float %10, ptr %arrayidx11, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %13 = load i32, ptr %i, align 4
  %inc = add nsw i32 %13, 1
  store i32 %inc, ptr %i, align 4
  br label %for.cond, !llvm.loop !11

for.end:                                          ; preds = %for.cond
  store i32 -4, ptr %i12, align 4
  br label %for.cond13

for.cond13:                                       ; preds = %for.inc23, %for.end
  %14 = load i32, ptr %i12, align 4
  %cmp14 = icmp slt i32 %14, 4
  br i1 %cmp14, label %for.body15, label %for.end25

for.body15:                                       ; preds = %for.cond13
  %15 = load i32, ptr %i12, align 4
  %add16 = add nsw i32 %15, 4
  %idxprom17 = sext i32 %add16 to i64
  %arrayidx18 = getelementptr inbounds [8 x [16 x float]], ptr addrspacecast (ptr addrspace(3) @_ZZ19stencil1d_coalescediPfS_E12input_shared to ptr), i64 0, i64 %idxprom17
  %16 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %idxprom20 = zext i32 %16 to i64
  %arrayidx21 = getelementptr inbounds [16 x float], ptr %arrayidx18, i64 0, i64 %idxprom20
  %17 = load float, ptr %arrayidx21, align 4
  %18 = load float, ptr %result, align 4
  %add22 = fadd contract float %18, %17
  store float %add22, ptr %result, align 4
  br label %for.inc23

for.inc23:                                        ; preds = %for.body15
  %19 = load i32, ptr %i12, align 4
  %inc24 = add nsw i32 %19, 1
  store i32 %inc24, ptr %i12, align 4
  br label %for.cond13, !llvm.loop !12

for.end25:                                        ; preds = %for.cond13
  %20 = load float, ptr %result, align 4
  %div = fdiv contract float %20, 7.000000e+00
  %21 = load ptr, ptr %output.addr, align 8
  %22 = load i32, ptr %idx, align 4
  %idxprom26 = sext i32 %22 to i64
  %arrayidx27 = getelementptr inbounds float, ptr %21, i64 %idxprom26
  store float %div, ptr %arrayidx27, align 4
  br label %if.end

if.end:                                           ; preds = %for.end25, %land.lhs.true, %entry
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x() #2

attributes #0 = { convergent mustprogress noinline norecurse nounwind  "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_35" "target-features"="+ptx75,+sm_35" "uniform-work-group-size"="true" }
attributes #1 = { convergent nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_35" "target-features"="+ptx75,+sm_35" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { convergent nounwind }

!llvm.module.flags = !{!0, !1, !2, !3}
!nvvm.annotations = !{!4, !5, !6}
!llvm.ident = !{!7, !8}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 11, i32 5]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!3 = !{i32 7, !"frame-pointer", i32 2}
!4 = !{ptr @dummy_func, !"kernel", i32 1}
!5 = !{ptr @_Z9stencil1diPfS_, !"kernel", i32 1}
!6 = !{ptr @_Z19stencil1d_coalescediPfS_, !"kernel", i32 1}
!7 = !{!"clang version 18.1.3 (https://github.com/virnarula/llvm-project.git 7709d36816961113db93da8bb9c3a1d05b7c2c0f)"}
!8 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!9 = distinct !{!9, !10}
!10 = !{!"llvm.loop.mustprogress"}
!11 = distinct !{!11, !10}
!12 = distinct !{!12, !10}
