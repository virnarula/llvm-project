; ModuleID = 'test-cuda-nvptx64-nvidia-cuda-sm_35.bc'
source_filename = "./test.cu"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%struct.__cuda_builtin_blockIdx_t = type { i8 }
%struct.__cuda_builtin_blockDim_t = type { i8 }
%struct.__cuda_builtin_threadIdx_t = type { i8 }

@blockIdx = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_blockIdx_t, align 1
@blockDim = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_blockDim_t, align 1
@threadIdx = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_threadIdx_t, align 1
@_ZZ19matrixAdd_coalescedPKfS0_PfiiE8A_shared = internal addrspace(3) global [16 x [16 x float]] undef, align 4
@_ZZ19matrixAdd_coalescedPKfS0_PfiiE8B_shared = internal addrspace(3) global [16 x [16 x float]] undef, align 4

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
define dso_local void @_Z9matrixAddPKfS0_Pfii(ptr noundef %A, ptr noundef %B, ptr noundef %C, i32 noundef %numRows, i32 noundef %numCols) #0 {
entry:
  %A.addr = alloca ptr, align 8
  %B.addr = alloca ptr, align 8
  %C.addr = alloca ptr, align 8
  %numRows.addr = alloca i32, align 4
  %numCols.addr = alloca i32, align 4
  %row = alloca i32, align 4
  %col = alloca i32, align 4
  %idx = alloca i32, align 4
  store ptr %A, ptr %A.addr, align 8
  store ptr %B, ptr %B.addr, align 8
  store ptr %C, ptr %C.addr, align 8
  store i32 %numRows, ptr %numRows.addr, align 4
  store i32 %numCols, ptr %numCols.addr, align 4
  call void @__tb_size_marker_2D(i32 noundef 16, i32 noundef 16) #4
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
  %6 = load i32, ptr %row, align 4
  %7 = load i32, ptr %numRows.addr, align 4
  %cmp = icmp slt i32 %6, %7
  br i1 %cmp, label %land.lhs.true, label %if.end

land.lhs.true:                                    ; preds = %entry
  %8 = load i32, ptr %col, align 4
  %9 = load i32, ptr %numCols.addr, align 4
  %cmp8 = icmp slt i32 %8, %9
  br i1 %cmp8, label %if.then, label %if.end

if.then:                                          ; preds = %land.lhs.true
  %10 = load i32, ptr %row, align 4
  %11 = load i32, ptr %numCols.addr, align 4
  %mul9 = mul nsw i32 %10, %11
  %12 = load i32, ptr %col, align 4
  %add10 = add nsw i32 %mul9, %12
  store i32 %add10, ptr %idx, align 4
  %13 = load ptr, ptr %A.addr, align 8
  %14 = load i32, ptr %idx, align 4
  %idxprom = sext i32 %14 to i64
  %arrayidx = getelementptr inbounds float, ptr %13, i64 %idxprom
  %15 = load float, ptr %arrayidx, align 4
  %16 = load ptr, ptr %B.addr, align 8
  %17 = load i32, ptr %idx, align 4
  %idxprom11 = sext i32 %17 to i64
  %arrayidx12 = getelementptr inbounds float, ptr %16, i64 %idxprom11
  %18 = load float, ptr %arrayidx12, align 4
  %add13 = fadd contract float %15, %18
  %19 = load ptr, ptr %C.addr, align 8
  %20 = load i32, ptr %idx, align 4
  %idxprom14 = sext i32 %20 to i64
  %arrayidx15 = getelementptr inbounds float, ptr %19, i64 %idxprom14
  store float %add13, ptr %arrayidx15, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %land.lhs.true, %entry
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind 
define dso_local void @_Z19matrixAdd_coalescedPKfS0_Pfii(ptr noundef %A, ptr noundef %B, ptr noundef %C, i32 noundef %numRows, i32 noundef %numCols) #0 {
entry:
  %A.addr = alloca ptr, align 8
  %B.addr = alloca ptr, align 8
  %C.addr = alloca ptr, align 8
  %numRows.addr = alloca i32, align 4
  %numCols.addr = alloca i32, align 4
  %row = alloca i32, align 4
  %col = alloca i32, align 4
  %idx = alloca i32, align 4
  store ptr %A, ptr %A.addr, align 8
  store ptr %B, ptr %B.addr, align 8
  store ptr %C, ptr %C.addr, align 8
  store i32 %numRows, ptr %numRows.addr, align 4
  store i32 %numCols, ptr %numCols.addr, align 4
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
  %6 = load i32, ptr %row, align 4
  %7 = load i32, ptr %numRows.addr, align 4
  %cmp = icmp slt i32 %6, %7
  br i1 %cmp, label %land.lhs.true, label %if.end

land.lhs.true:                                    ; preds = %entry
  %8 = load i32, ptr %col, align 4
  %9 = load i32, ptr %numCols.addr, align 4
  %cmp8 = icmp slt i32 %8, %9
  br i1 %cmp8, label %if.then, label %if.end

if.then:                                          ; preds = %land.lhs.true
  %10 = load ptr, ptr %A.addr, align 8
  %11 = load i32, ptr %row, align 4
  %12 = load i32, ptr %numCols.addr, align 4
  %mul9 = mul nsw i32 %11, %12
  %13 = load i32, ptr %col, align 4
  %add10 = add nsw i32 %mul9, %13
  %idxprom = sext i32 %add10 to i64
  %arrayidx = getelementptr inbounds float, ptr %10, i64 %idxprom
  %14 = load float, ptr %arrayidx, align 4
  %15 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %idxprom12 = zext i32 %15 to i64
  %arrayidx13 = getelementptr inbounds [16 x [16 x float]], ptr addrspacecast (ptr addrspace(3) @_ZZ19matrixAdd_coalescedPKfS0_PfiiE8A_shared to ptr), i64 0, i64 %idxprom12
  %16 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %idxprom15 = zext i32 %16 to i64
  %arrayidx16 = getelementptr inbounds [16 x float], ptr %arrayidx13, i64 0, i64 %idxprom15
  store float %14, ptr %arrayidx16, align 4
  %17 = load ptr, ptr %B.addr, align 8
  %18 = load i32, ptr %row, align 4
  %19 = load i32, ptr %numCols.addr, align 4
  %mul17 = mul nsw i32 %18, %19
  %20 = load i32, ptr %col, align 4
  %add18 = add nsw i32 %mul17, %20
  %idxprom19 = sext i32 %add18 to i64
  %arrayidx20 = getelementptr inbounds float, ptr %17, i64 %idxprom19
  %21 = load float, ptr %arrayidx20, align 4
  %22 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %idxprom22 = zext i32 %22 to i64
  %arrayidx23 = getelementptr inbounds [16 x [16 x float]], ptr addrspacecast (ptr addrspace(3) @_ZZ19matrixAdd_coalescedPKfS0_PfiiE8B_shared to ptr), i64 0, i64 %idxprom22
  %23 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %idxprom25 = zext i32 %23 to i64
  %arrayidx26 = getelementptr inbounds [16 x float], ptr %arrayidx23, i64 0, i64 %idxprom25
  store float %21, ptr %arrayidx26, align 4
  call void @llvm.nvvm.barrier0()
  %24 = load i32, ptr %row, align 4
  %25 = load i32, ptr %numCols.addr, align 4
  %mul27 = mul nsw i32 %24, %25
  %26 = load i32, ptr %col, align 4
  %add28 = add nsw i32 %mul27, %26
  store i32 %add28, ptr %idx, align 4
  %27 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %idxprom30 = zext i32 %27 to i64
  %arrayidx31 = getelementptr inbounds [16 x [16 x float]], ptr addrspacecast (ptr addrspace(3) @_ZZ19matrixAdd_coalescedPKfS0_PfiiE8A_shared to ptr), i64 0, i64 %idxprom30
  %28 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %idxprom33 = zext i32 %28 to i64
  %arrayidx34 = getelementptr inbounds [16 x float], ptr %arrayidx31, i64 0, i64 %idxprom33
  %29 = load float, ptr %arrayidx34, align 4
  %30 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %idxprom36 = zext i32 %30 to i64
  %arrayidx37 = getelementptr inbounds [16 x [16 x float]], ptr addrspacecast (ptr addrspace(3) @_ZZ19matrixAdd_coalescedPKfS0_PfiiE8B_shared to ptr), i64 0, i64 %idxprom36
  %31 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %idxprom39 = zext i32 %31 to i64
  %arrayidx40 = getelementptr inbounds [16 x float], ptr %arrayidx37, i64 0, i64 %idxprom39
  %32 = load float, ptr %arrayidx40, align 4
  %add41 = fadd contract float %29, %32
  %33 = load ptr, ptr %C.addr, align 8
  %34 = load i32, ptr %idx, align 4
  %idxprom42 = sext i32 %34 to i64
  %arrayidx43 = getelementptr inbounds float, ptr %33, i64 %idxprom42
  store float %add41, ptr %arrayidx43, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %land.lhs.true, %entry
  ret void
}

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier0() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y() #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3

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
!5 = !{ptr @_Z9matrixAddPKfS0_Pfii, !"kernel", i32 1}
!6 = !{ptr @_Z19matrixAdd_coalescedPKfS0_Pfii, !"kernel", i32 1}
!7 = !{!"clang version 18.1.3 (https://github.com/virnarula/llvm-project.git 7709d36816961113db93da8bb9c3a1d05b7c2c0f)"}
!8 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
