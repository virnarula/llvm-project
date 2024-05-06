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
@_ZZ25transposeMatrix_coalescedPKfPfiiE12input_shared = internal addrspace(3) global [16 x [16 x float]] undef, align 4

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
define dso_local void @_Z15transposeMatrixPKfPfii(ptr noundef %input, ptr noundef %output, i32 noundef %width, i32 noundef %height) #0 {
entry:
  %input.addr = alloca ptr, align 8
  %output.addr = alloca ptr, align 8
  %width.addr = alloca i32, align 4
  %height.addr = alloca i32, align 4
  %x = alloca i32, align 4
  %y = alloca i32, align 4
  %pos = alloca i32, align 4
  %transPos = alloca i32, align 4
  store ptr %input, ptr %input.addr, align 8
  store ptr %output, ptr %output.addr, align 8
  store i32 %width, ptr %width.addr, align 4
  store i32 %height, ptr %height.addr, align 4
  %0 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %1 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %mul = mul i32 %0, %1
  %2 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %add = add i32 %mul, %2
  store i32 %add, ptr %x, align 4
  %3 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %4 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
  %mul5 = mul i32 %3, %4
  %5 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %add7 = add i32 %mul5, %5
  store i32 %add7, ptr %y, align 4
  call void @__tb_size_marker_2D(i32 noundef 16, i32 noundef 16) #4
  %6 = load i32, ptr %x, align 4
  %7 = load i32, ptr %width.addr, align 4
  %cmp = icmp slt i32 %6, %7
  br i1 %cmp, label %land.lhs.true, label %if.end

land.lhs.true:                                    ; preds = %entry
  %8 = load i32, ptr %y, align 4
  %9 = load i32, ptr %height.addr, align 4
  %cmp8 = icmp slt i32 %8, %9
  br i1 %cmp8, label %if.then, label %if.end

if.then:                                          ; preds = %land.lhs.true
  %10 = load i32, ptr %y, align 4
  %11 = load i32, ptr %width.addr, align 4
  %mul9 = mul nsw i32 %10, %11
  %12 = load i32, ptr %x, align 4
  %add10 = add nsw i32 %mul9, %12
  store i32 %add10, ptr %pos, align 4
  %13 = load i32, ptr %x, align 4
  %14 = load i32, ptr %height.addr, align 4
  %mul11 = mul nsw i32 %13, %14
  %15 = load i32, ptr %y, align 4
  %add12 = add nsw i32 %mul11, %15
  store i32 %add12, ptr %transPos, align 4
  %16 = load ptr, ptr %input.addr, align 8
  %17 = load i32, ptr %pos, align 4
  %idxprom = sext i32 %17 to i64
  %arrayidx = getelementptr inbounds float, ptr %16, i64 %idxprom
  %18 = load float, ptr %arrayidx, align 4
  %19 = load ptr, ptr %output.addr, align 8
  %20 = load i32, ptr %transPos, align 4
  %idxprom13 = sext i32 %20 to i64
  %arrayidx14 = getelementptr inbounds float, ptr %19, i64 %idxprom13
  store float %18, ptr %arrayidx14, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %land.lhs.true, %entry
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind 
define dso_local void @_Z25transposeMatrix_coalescedPKfPfii(ptr noundef %input, ptr noundef %output, i32 noundef %width, i32 noundef %height) #0 {
entry:
  %input.addr = alloca ptr, align 8
  %output.addr = alloca ptr, align 8
  %width.addr = alloca i32, align 4
  %height.addr = alloca i32, align 4
  %x = alloca i32, align 4
  %y = alloca i32, align 4
  %pos = alloca i32, align 4
  %transPos = alloca i32, align 4
  store ptr %input, ptr %input.addr, align 8
  store ptr %output, ptr %output.addr, align 8
  store i32 %width, ptr %width.addr, align 4
  store i32 %height, ptr %height.addr, align 4
  %0 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %1 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %mul = mul i32 %0, %1
  %2 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %add = add i32 %mul, %2
  store i32 %add, ptr %x, align 4
  %3 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %4 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
  %mul5 = mul i32 %3, %4
  %5 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %add7 = add i32 %mul5, %5
  store i32 %add7, ptr %y, align 4
  %6 = load ptr, ptr %input.addr, align 8
  %7 = load i32, ptr %y, align 4
  %8 = load i32, ptr %width.addr, align 4
  %mul8 = mul nsw i32 %7, %8
  %9 = load i32, ptr %x, align 4
  %add9 = add nsw i32 %mul8, %9
  %idxprom = sext i32 %add9 to i64
  %arrayidx = getelementptr inbounds float, ptr %6, i64 %idxprom
  %10 = load float, ptr %arrayidx, align 4
  %11 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %idxprom11 = zext i32 %11 to i64
  %arrayidx12 = getelementptr inbounds [16 x [16 x float]], ptr addrspacecast (ptr addrspace(3) @_ZZ25transposeMatrix_coalescedPKfPfiiE12input_shared to ptr), i64 0, i64 %idxprom11
  %12 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %idxprom14 = zext i32 %12 to i64
  %arrayidx15 = getelementptr inbounds [16 x float], ptr %arrayidx12, i64 0, i64 %idxprom14
  store float %10, ptr %arrayidx15, align 4
  call void @llvm.nvvm.barrier0()
  %13 = load i32, ptr %x, align 4
  %14 = load i32, ptr %width.addr, align 4
  %cmp = icmp slt i32 %13, %14
  br i1 %cmp, label %land.lhs.true, label %if.end

land.lhs.true:                                    ; preds = %entry
  %15 = load i32, ptr %y, align 4
  %16 = load i32, ptr %height.addr, align 4
  %cmp16 = icmp slt i32 %15, %16
  br i1 %cmp16, label %if.then, label %if.end

if.then:                                          ; preds = %land.lhs.true
  %17 = load i32, ptr %y, align 4
  %18 = load i32, ptr %width.addr, align 4
  %mul17 = mul nsw i32 %17, %18
  %19 = load i32, ptr %x, align 4
  %add18 = add nsw i32 %mul17, %19
  store i32 %add18, ptr %pos, align 4
  %20 = load i32, ptr %x, align 4
  %21 = load i32, ptr %height.addr, align 4
  %mul19 = mul nsw i32 %20, %21
  %22 = load i32, ptr %y, align 4
  %add20 = add nsw i32 %mul19, %22
  store i32 %add20, ptr %transPos, align 4
  %23 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %idxprom22 = zext i32 %23 to i64
  %arrayidx23 = getelementptr inbounds [16 x [16 x float]], ptr addrspacecast (ptr addrspace(3) @_ZZ25transposeMatrix_coalescedPKfPfiiE12input_shared to ptr), i64 0, i64 %idxprom22
  %24 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %idxprom25 = zext i32 %24 to i64
  %arrayidx26 = getelementptr inbounds [16 x float], ptr %arrayidx23, i64 0, i64 %idxprom25
  %25 = load float, ptr %arrayidx26, align 4
  %26 = load ptr, ptr %output.addr, align 8
  %27 = load i32, ptr %transPos, align 4
  %idxprom27 = sext i32 %27 to i64
  %arrayidx28 = getelementptr inbounds float, ptr %26, i64 %idxprom27
  store float %25, ptr %arrayidx28, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %land.lhs.true, %entry
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
!5 = !{ptr @_Z15transposeMatrixPKfPfii, !"kernel", i32 1}
!6 = !{ptr @_Z25transposeMatrix_coalescedPKfPfii, !"kernel", i32 1}
!7 = !{!"clang version 18.1.3 (https://github.com/virnarula/llvm-project.git 7709d36816961113db93da8bb9c3a1d05b7c2c0f)"}
!8 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
