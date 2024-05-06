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
define dso_local void @_Z7naiveMMPKfPfi(ptr noundef %A, ptr noundef %C, i32 noundef %w) #0 {
entry:
  %A.addr = alloca ptr, align 8
  %C.addr = alloca ptr, align 8
  %w.addr = alloca i32, align 4
  %idx = alloca i32, align 4
  %sum = alloca float, align 4
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  store ptr %A, ptr %A.addr, align 8
  store ptr %C, ptr %C.addr, align 8
  store i32 %w, ptr %w.addr, align 4
  %0 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %1 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %mul = mul i32 %0, %1
  %2 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %add = add i32 %mul, %2
  store i32 %add, ptr %idx, align 4
  store float 0.000000e+00, ptr %sum, align 4
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc12, %entry
  %3 = load i32, ptr %i, align 4
  %4 = load i32, ptr %w.addr, align 4
  %cmp = icmp slt i32 %3, %4
  br i1 %cmp, label %for.body, label %for.end14

for.body:                                         ; preds = %for.cond
  store i32 0, ptr %j, align 4
  br label %for.cond3

for.cond3:                                        ; preds = %for.inc, %for.body
  %5 = load i32, ptr %j, align 4
  %6 = load i32, ptr %w.addr, align 4
  %cmp4 = icmp slt i32 %5, %6
  br i1 %cmp4, label %for.body5, label %for.end

for.body5:                                        ; preds = %for.cond3
  %7 = load ptr, ptr %A.addr, align 8
  %8 = load i32, ptr %i, align 4
  %9 = load i32, ptr %j, align 4
  %add6 = add nsw i32 %8, %9
  %idxprom = sext i32 %add6 to i64
  %arrayidx = getelementptr inbounds float, ptr %7, i64 %idxprom
  %10 = load float, ptr %arrayidx, align 4
  %11 = load ptr, ptr %A.addr, align 8
  %12 = load i32, ptr %i, align 4
  %13 = load i32, ptr %j, align 4
  %add7 = add nsw i32 %12, %13
  %idxprom8 = sext i32 %add7 to i64
  %arrayidx9 = getelementptr inbounds float, ptr %11, i64 %idxprom8
  %14 = load float, ptr %arrayidx9, align 4
  %mul10 = fmul contract float %10, %14
  %15 = load float, ptr %sum, align 4
  %add11 = fadd contract float %15, %mul10
  store float %add11, ptr %sum, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body5
  %16 = load i32, ptr %j, align 4
  %inc = add nsw i32 %16, 1
  store i32 %inc, ptr %j, align 4
  br label %for.cond3, !llvm.loop !8

for.end:                                          ; preds = %for.cond3
  br label %for.inc12

for.inc12:                                        ; preds = %for.end
  %17 = load i32, ptr %i, align 4
  %inc13 = add nsw i32 %17, 1
  store i32 %inc13, ptr %i, align 4
  br label %for.cond, !llvm.loop !10

for.end14:                                        ; preds = %for.cond
  %18 = load float, ptr %sum, align 4
  %19 = load ptr, ptr %C.addr, align 8
  %20 = load i32, ptr %idx, align 4
  %idxprom15 = sext i32 %20 to i64
  %arrayidx16 = getelementptr inbounds float, ptr %19, i64 %idxprom15
  store float %18, ptr %arrayidx16, align 4
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x() #2

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
!5 = !{ptr @_Z7naiveMMPKfPfi, !"kernel", i32 1}
!6 = !{!"clang version 18.1.3 (https://github.com/virnarula/llvm-project.git 7709d36816961113db93da8bb9c3a1d05b7c2c0f)"}
!7 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!8 = distinct !{!8, !9}
!9 = !{!"llvm.loop.mustprogress"}
!10 = distinct !{!10, !9}
