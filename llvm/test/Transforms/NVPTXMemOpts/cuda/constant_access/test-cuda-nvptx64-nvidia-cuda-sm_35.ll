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
define dso_local void @_Z7naiveMMPPKfS1_PPfi(ptr noundef %A, ptr noundef %B, ptr noundef %C, i32 noundef %w) #0 {
entry:
  %A.addr = alloca ptr, align 8
  %B.addr = alloca ptr, align 8
  %C.addr = alloca ptr, align 8
  %w.addr = alloca i32, align 4
  %idx = alloca i32, align 4
  %idy = alloca i32, align 4
  %sum = alloca float, align 4
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  store ptr %A, ptr %A.addr, align 8
  store ptr %B, ptr %B.addr, align 8
  store ptr %C, ptr %C.addr, align 8
  store i32 %w, ptr %w.addr, align 4
  %0 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %1 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %mul = mul i32 %0, %1
  %2 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %add = add i32 %mul, %2
  store i32 %add, ptr %idx, align 4
  %3 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
  %4 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %mul5 = mul i32 %3, %4
  %5 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %add7 = add i32 %mul5, %5
  store i32 %add7, ptr %idy, align 4
  call void @__tb_size_marker_2D(i32 noundef 128, i32 noundef 43) #3
  store float 0.000000e+00, ptr %sum, align 4
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc16, %entry
  %6 = load i32, ptr %i, align 4
  %7 = load i32, ptr %w.addr, align 4
  %cmp = icmp slt i32 %6, %7
  br i1 %cmp, label %for.body, label %for.end18

for.body:                                         ; preds = %for.cond
  store i32 0, ptr %j, align 4
  br label %for.cond8

for.cond8:                                        ; preds = %for.inc, %for.body
  %8 = load i32, ptr %j, align 4
  %9 = load i32, ptr %w.addr, align 4
  %cmp9 = icmp slt i32 %8, %9
  br i1 %cmp9, label %for.body10, label %for.end

for.body10:                                       ; preds = %for.cond8
  %10 = load ptr, ptr %A.addr, align 8
  %arrayidx = getelementptr inbounds ptr, ptr %10, i64 10
  %11 = load ptr, ptr %arrayidx, align 8
  %arrayidx11 = getelementptr inbounds float, ptr %11, i64 15
  %12 = load float, ptr %arrayidx11, align 4
  %13 = load ptr, ptr %B.addr, align 8
  %arrayidx12 = getelementptr inbounds ptr, ptr %13, i64 23
  %14 = load ptr, ptr %arrayidx12, align 8
  %arrayidx13 = getelementptr inbounds float, ptr %14, i64 54
  %15 = load float, ptr %arrayidx13, align 4
  %mul14 = fmul contract float %12, %15
  %16 = load float, ptr %sum, align 4
  %add15 = fadd contract float %16, %mul14
  store float %add15, ptr %sum, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body10
  %17 = load i32, ptr %j, align 4
  %inc = add nsw i32 %17, 1
  store i32 %inc, ptr %j, align 4
  br label %for.cond8, !llvm.loop !8

for.end:                                          ; preds = %for.cond8
  br label %for.inc16

for.inc16:                                        ; preds = %for.end
  %18 = load i32, ptr %i, align 4
  %inc17 = add nsw i32 %18, 1
  store i32 %inc17, ptr %i, align 4
  br label %for.cond, !llvm.loop !10

for.end18:                                        ; preds = %for.cond
  %19 = load float, ptr %sum, align 4
  %20 = load ptr, ptr %C.addr, align 8
  %21 = load i32, ptr %idy, align 4
  %idxprom = sext i32 %21 to i64
  %arrayidx19 = getelementptr inbounds ptr, ptr %20, i64 %idxprom
  %22 = load ptr, ptr %arrayidx19, align 8
  %23 = load i32, ptr %idx, align 4
  %idxprom20 = sext i32 %23 to i64
  %arrayidx21 = getelementptr inbounds float, ptr %22, i64 %idxprom20
  store float %19, ptr %arrayidx21, align 4
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

attributes #0 = { convergent mustprogress noinline norecurse nounwind  "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_35" "target-features"="+ptx75,+sm_35" "uniform-work-group-size"="true" }
attributes #1 = { convergent nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_35" "target-features"="+ptx75,+sm_35" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
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
