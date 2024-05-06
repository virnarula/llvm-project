; ModuleID = 'test.cu'
source_filename = "test.cu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.dim3 = type { i32, i32, i32 }

$_ZN4dim3C2Ejjj = comdat any

$_ZSt4fabsf = comdat any

@stderr = external global ptr, align 8
@.str = private unnamed_addr constant [43 x i8] c"Result verification failed at element %d!\0A\00", align 1
@.str.1 = private unnamed_addr constant [13 x i8] c"Test PASSED\0A\00", align 1

; Function Attrs: mustprogress noinline norecurse optnone uwtable
define dso_local void @__device_stub__dummy_func() #0 {
entry:
  %grid_dim = alloca %struct.dim3, align 8
  %block_dim = alloca %struct.dim3, align 8
  %shmem_size = alloca i64, align 8
  %stream = alloca ptr, align 8
  %grid_dim.coerce = alloca { i64, i32 }, align 8
  %block_dim.coerce = alloca { i64, i32 }, align 8
  %kernel_args = alloca ptr, i64 1, align 16
  %0 = call i32 @__cudaPopCallConfiguration(ptr %grid_dim, ptr %block_dim, ptr %shmem_size, ptr %stream)
  %1 = load i64, ptr %shmem_size, align 8
  %2 = load ptr, ptr %stream, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %grid_dim.coerce, ptr align 8 %grid_dim, i64 12, i1 false)
  %3 = getelementptr inbounds { i64, i32 }, ptr %grid_dim.coerce, i32 0, i32 0
  %4 = load i64, ptr %3, align 8
  %5 = getelementptr inbounds { i64, i32 }, ptr %grid_dim.coerce, i32 0, i32 1
  %6 = load i32, ptr %5, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %block_dim.coerce, ptr align 8 %block_dim, i64 12, i1 false)
  %7 = getelementptr inbounds { i64, i32 }, ptr %block_dim.coerce, i32 0, i32 0
  %8 = load i64, ptr %7, align 8
  %9 = getelementptr inbounds { i64, i32 }, ptr %block_dim.coerce, i32 0, i32 1
  %10 = load i32, ptr %9, align 8
  %call = call noundef i32 @cudaLaunchKernel(ptr noundef @__device_stub__dummy_func, i64 %4, i32 %6, i64 %8, i32 %10, ptr noundef %kernel_args, i64 noundef %1, ptr noundef %2)
  br label %setup.end

setup.end:                                        ; preds = %entry
  ret void
}

declare i32 @__cudaPopCallConfiguration(ptr, ptr, ptr, ptr)

declare i32 @cudaLaunchKernel(ptr, i64, i32, i64, i32, ptr, i64, ptr)

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #1

; Function Attrs: mustprogress noinline norecurse optnone uwtable
define dso_local void @_Z29__device_stub__vectorMultiplyPKfS0_Pfi(ptr noundef %input1, ptr noundef %input2, ptr noundef %output, i32 noundef %numElements) #0 {
entry:
  %input1.addr = alloca ptr, align 8
  %input2.addr = alloca ptr, align 8
  %output.addr = alloca ptr, align 8
  %numElements.addr = alloca i32, align 4
  %grid_dim = alloca %struct.dim3, align 8
  %block_dim = alloca %struct.dim3, align 8
  %shmem_size = alloca i64, align 8
  %stream = alloca ptr, align 8
  %grid_dim.coerce = alloca { i64, i32 }, align 8
  %block_dim.coerce = alloca { i64, i32 }, align 8
  store ptr %input1, ptr %input1.addr, align 8
  store ptr %input2, ptr %input2.addr, align 8
  store ptr %output, ptr %output.addr, align 8
  store i32 %numElements, ptr %numElements.addr, align 4
  %kernel_args = alloca ptr, i64 4, align 16
  %0 = getelementptr ptr, ptr %kernel_args, i32 0
  store ptr %input1.addr, ptr %0, align 8
  %1 = getelementptr ptr, ptr %kernel_args, i32 1
  store ptr %input2.addr, ptr %1, align 8
  %2 = getelementptr ptr, ptr %kernel_args, i32 2
  store ptr %output.addr, ptr %2, align 8
  %3 = getelementptr ptr, ptr %kernel_args, i32 3
  store ptr %numElements.addr, ptr %3, align 8
  %4 = call i32 @__cudaPopCallConfiguration(ptr %grid_dim, ptr %block_dim, ptr %shmem_size, ptr %stream)
  %5 = load i64, ptr %shmem_size, align 8
  %6 = load ptr, ptr %stream, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %grid_dim.coerce, ptr align 8 %grid_dim, i64 12, i1 false)
  %7 = getelementptr inbounds { i64, i32 }, ptr %grid_dim.coerce, i32 0, i32 0
  %8 = load i64, ptr %7, align 8
  %9 = getelementptr inbounds { i64, i32 }, ptr %grid_dim.coerce, i32 0, i32 1
  %10 = load i32, ptr %9, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %block_dim.coerce, ptr align 8 %block_dim, i64 12, i1 false)
  %11 = getelementptr inbounds { i64, i32 }, ptr %block_dim.coerce, i32 0, i32 0
  %12 = load i64, ptr %11, align 8
  %13 = getelementptr inbounds { i64, i32 }, ptr %block_dim.coerce, i32 0, i32 1
  %14 = load i32, ptr %13, align 8
  %call = call noundef i32 @cudaLaunchKernel(ptr noundef @_Z29__device_stub__vectorMultiplyPKfS0_Pfi, i64 %8, i32 %10, i64 %12, i32 %14, ptr noundef %kernel_args, i64 noundef %5, ptr noundef %6)
  br label %setup.end

setup.end:                                        ; preds = %entry
  ret void
}

; Function Attrs: mustprogress noinline norecurse optnone uwtable
define dso_local void @_Z39__device_stub__vectorMultiply_coalescedPKfS0_Pfi(ptr noundef %input1, ptr noundef %input2, ptr noundef %output, i32 noundef %numElements) #0 {
entry:
  %input1.addr = alloca ptr, align 8
  %input2.addr = alloca ptr, align 8
  %output.addr = alloca ptr, align 8
  %numElements.addr = alloca i32, align 4
  %grid_dim = alloca %struct.dim3, align 8
  %block_dim = alloca %struct.dim3, align 8
  %shmem_size = alloca i64, align 8
  %stream = alloca ptr, align 8
  %grid_dim.coerce = alloca { i64, i32 }, align 8
  %block_dim.coerce = alloca { i64, i32 }, align 8
  store ptr %input1, ptr %input1.addr, align 8
  store ptr %input2, ptr %input2.addr, align 8
  store ptr %output, ptr %output.addr, align 8
  store i32 %numElements, ptr %numElements.addr, align 4
  %kernel_args = alloca ptr, i64 4, align 16
  %0 = getelementptr ptr, ptr %kernel_args, i32 0
  store ptr %input1.addr, ptr %0, align 8
  %1 = getelementptr ptr, ptr %kernel_args, i32 1
  store ptr %input2.addr, ptr %1, align 8
  %2 = getelementptr ptr, ptr %kernel_args, i32 2
  store ptr %output.addr, ptr %2, align 8
  %3 = getelementptr ptr, ptr %kernel_args, i32 3
  store ptr %numElements.addr, ptr %3, align 8
  %4 = call i32 @__cudaPopCallConfiguration(ptr %grid_dim, ptr %block_dim, ptr %shmem_size, ptr %stream)
  %5 = load i64, ptr %shmem_size, align 8
  %6 = load ptr, ptr %stream, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %grid_dim.coerce, ptr align 8 %grid_dim, i64 12, i1 false)
  %7 = getelementptr inbounds { i64, i32 }, ptr %grid_dim.coerce, i32 0, i32 0
  %8 = load i64, ptr %7, align 8
  %9 = getelementptr inbounds { i64, i32 }, ptr %grid_dim.coerce, i32 0, i32 1
  %10 = load i32, ptr %9, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %block_dim.coerce, ptr align 8 %block_dim, i64 12, i1 false)
  %11 = getelementptr inbounds { i64, i32 }, ptr %block_dim.coerce, i32 0, i32 0
  %12 = load i64, ptr %11, align 8
  %13 = getelementptr inbounds { i64, i32 }, ptr %block_dim.coerce, i32 0, i32 1
  %14 = load i32, ptr %13, align 8
  %call = call noundef i32 @cudaLaunchKernel(ptr noundef @_Z39__device_stub__vectorMultiply_coalescedPKfS0_Pfi, i64 %8, i32 %10, i64 %12, i32 %14, ptr noundef %kernel_args, i64 noundef %5, ptr noundef %6)
  br label %setup.end

setup.end:                                        ; preds = %entry
  ret void
}

; Function Attrs: mustprogress noinline norecurse optnone uwtable
define dso_local noundef i32 @main() #2 {
entry:
  %retval = alloca i32, align 4
  %numElements = alloca i32, align 4
  %size = alloca i64, align 8
  %h_input1 = alloca ptr, align 8
  %h_input2 = alloca ptr, align 8
  %h_output = alloca ptr, align 8
  %i = alloca i32, align 4
  %d_input1 = alloca ptr, align 8
  %d_input2 = alloca ptr, align 8
  %d_output = alloca ptr, align 8
  %threadsPerBlock = alloca i32, align 4
  %blocksPerGrid = alloca i32, align 4
  %agg.tmp = alloca %struct.dim3, align 4
  %agg.tmp13 = alloca %struct.dim3, align 4
  %agg.tmp.coerce = alloca { i64, i32 }, align 4
  %agg.tmp13.coerce = alloca { i64, i32 }, align 4
  %i16 = alloca i32, align 4
  store i32 0, ptr %retval, align 4
  store i32 50000, ptr %numElements, align 4
  %0 = load i32, ptr %numElements, align 4
  %conv = sext i32 %0 to i64
  %mul = mul i64 %conv, 4
  store i64 %mul, ptr %size, align 8
  %1 = load i64, ptr %size, align 8
  %call = call noalias ptr @malloc(i64 noundef %1) #10
  store ptr %call, ptr %h_input1, align 8
  %2 = load i64, ptr %size, align 8
  %call1 = call noalias ptr @malloc(i64 noundef %2) #10
  store ptr %call1, ptr %h_input2, align 8
  %3 = load i64, ptr %size, align 8
  %call2 = call noalias ptr @malloc(i64 noundef %3) #10
  store ptr %call2, ptr %h_output, align 8
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %4 = load i32, ptr %i, align 4
  %5 = load i32, ptr %numElements, align 4
  %cmp = icmp slt i32 %4, %5
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %6 = load i32, ptr %i, align 4
  %conv3 = sitofp i32 %6 to float
  %7 = load ptr, ptr %h_input1, align 8
  %8 = load i32, ptr %i, align 4
  %idxprom = sext i32 %8 to i64
  %arrayidx = getelementptr inbounds float, ptr %7, i64 %idxprom
  store float %conv3, ptr %arrayidx, align 4
  %9 = load i32, ptr %i, align 4
  %mul4 = mul nsw i32 2, %9
  %conv5 = sitofp i32 %mul4 to float
  %10 = load ptr, ptr %h_input2, align 8
  %11 = load i32, ptr %i, align 4
  %idxprom6 = sext i32 %11 to i64
  %arrayidx7 = getelementptr inbounds float, ptr %10, i64 %idxprom6
  store float %conv5, ptr %arrayidx7, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %12 = load i32, ptr %i, align 4
  %inc = add nsw i32 %12, 1
  store i32 %inc, ptr %i, align 4
  br label %for.cond, !llvm.loop !7

for.end:                                          ; preds = %for.cond
  %13 = load i64, ptr %size, align 8
  %call8 = call noundef i32 @_ZL10cudaMallocIfE9cudaErrorPPT_m(ptr noundef %d_input1, i64 noundef %13)
  %14 = load i64, ptr %size, align 8
  %call9 = call noundef i32 @_ZL10cudaMallocIfE9cudaErrorPPT_m(ptr noundef %d_input2, i64 noundef %14)
  %15 = load i64, ptr %size, align 8
  %call10 = call noundef i32 @_ZL10cudaMallocIfE9cudaErrorPPT_m(ptr noundef %d_output, i64 noundef %15)
  %16 = load ptr, ptr %d_input1, align 8
  %17 = load ptr, ptr %h_input1, align 8
  %18 = load i64, ptr %size, align 8
  %call11 = call i32 @cudaMemcpy(ptr noundef %16, ptr noundef %17, i64 noundef %18, i32 noundef 1)
  %19 = load ptr, ptr %d_input2, align 8
  %20 = load ptr, ptr %h_input2, align 8
  %21 = load i64, ptr %size, align 8
  %call12 = call i32 @cudaMemcpy(ptr noundef %19, ptr noundef %20, i64 noundef %21, i32 noundef 1)
  store i32 256, ptr %threadsPerBlock, align 4
  %22 = load i32, ptr %numElements, align 4
  %23 = load i32, ptr %threadsPerBlock, align 4
  %add = add nsw i32 %22, %23
  %sub = sub nsw i32 %add, 1
  %24 = load i32, ptr %threadsPerBlock, align 4
  %div = sdiv i32 %sub, %24
  store i32 %div, ptr %blocksPerGrid, align 4
  %25 = load i32, ptr %blocksPerGrid, align 4
  call void @_ZN4dim3C2Ejjj(ptr noundef nonnull align 4 dereferenceable(12) %agg.tmp, i32 noundef %25, i32 noundef 1, i32 noundef 1)
  %26 = load i32, ptr %threadsPerBlock, align 4
  call void @_ZN4dim3C2Ejjj(ptr noundef nonnull align 4 dereferenceable(12) %agg.tmp13, i32 noundef %26, i32 noundef 1, i32 noundef 1)
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp.coerce, ptr align 4 %agg.tmp, i64 12, i1 false)
  %27 = getelementptr inbounds { i64, i32 }, ptr %agg.tmp.coerce, i32 0, i32 0
  %28 = load i64, ptr %27, align 4
  %29 = getelementptr inbounds { i64, i32 }, ptr %agg.tmp.coerce, i32 0, i32 1
  %30 = load i32, ptr %29, align 4
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp13.coerce, ptr align 4 %agg.tmp13, i64 12, i1 false)
  %31 = getelementptr inbounds { i64, i32 }, ptr %agg.tmp13.coerce, i32 0, i32 0
  %32 = load i64, ptr %31, align 4
  %33 = getelementptr inbounds { i64, i32 }, ptr %agg.tmp13.coerce, i32 0, i32 1
  %34 = load i32, ptr %33, align 4
  %call14 = call i32 @__cudaPushCallConfiguration(i64 %28, i32 %30, i64 %32, i32 %34, i64 noundef 0, ptr noundef null)
  %tobool = icmp ne i32 %call14, 0
  br i1 %tobool, label %kcall.end, label %kcall.configok

kcall.configok:                                   ; preds = %for.end
  %35 = load ptr, ptr %d_input1, align 8
  %36 = load ptr, ptr %d_input2, align 8
  %37 = load ptr, ptr %d_output, align 8
  %38 = load i32, ptr %numElements, align 4
  call void @_Z29__device_stub__vectorMultiplyPKfS0_Pfi(ptr noundef %35, ptr noundef %36, ptr noundef %37, i32 noundef %38) #11
  br label %kcall.end

kcall.end:                                        ; preds = %kcall.configok, %for.end
  %39 = load ptr, ptr %h_output, align 8
  %40 = load ptr, ptr %d_output, align 8
  %41 = load i64, ptr %size, align 8
  %call15 = call i32 @cudaMemcpy(ptr noundef %39, ptr noundef %40, i64 noundef %41, i32 noundef 2)
  store i32 0, ptr %i16, align 4
  br label %for.cond17

for.cond17:                                       ; preds = %for.inc31, %kcall.end
  %42 = load i32, ptr %i16, align 4
  %43 = load i32, ptr %numElements, align 4
  %cmp18 = icmp slt i32 %42, %43
  br i1 %cmp18, label %for.body19, label %for.end33

for.body19:                                       ; preds = %for.cond17
  %44 = load ptr, ptr %h_input1, align 8
  %45 = load i32, ptr %i16, align 4
  %idxprom20 = sext i32 %45 to i64
  %arrayidx21 = getelementptr inbounds float, ptr %44, i64 %idxprom20
  %46 = load float, ptr %arrayidx21, align 4
  %47 = load ptr, ptr %h_input2, align 8
  %48 = load i32, ptr %i16, align 4
  %idxprom22 = sext i32 %48 to i64
  %arrayidx23 = getelementptr inbounds float, ptr %47, i64 %idxprom22
  %49 = load float, ptr %arrayidx23, align 4
  %50 = load ptr, ptr %h_output, align 8
  %51 = load i32, ptr %i16, align 4
  %idxprom25 = sext i32 %51 to i64
  %arrayidx26 = getelementptr inbounds float, ptr %50, i64 %idxprom25
  %52 = load float, ptr %arrayidx26, align 4
  %neg = fneg float %52
  %53 = call float @llvm.fmuladd.f32(float %46, float %49, float %neg)
  %call27 = call noundef float @_ZSt4fabsf(float noundef %53)
  %conv28 = fpext float %call27 to double
  %cmp29 = fcmp ogt double %conv28, 1.000000e-05
  br i1 %cmp29, label %if.then, label %if.end

if.then:                                          ; preds = %for.body19
  %54 = load ptr, ptr @stderr, align 8
  %55 = load i32, ptr %i16, align 4
  %call30 = call i32 (ptr, ptr, ...) @fprintf(ptr noundef %54, ptr noundef @.str, i32 noundef %55)
  call void @exit(i32 noundef 1) #12
  unreachable

if.end:                                           ; preds = %for.body19
  br label %for.inc31

for.inc31:                                        ; preds = %if.end
  %56 = load i32, ptr %i16, align 4
  %inc32 = add nsw i32 %56, 1
  store i32 %inc32, ptr %i16, align 4
  br label %for.cond17, !llvm.loop !9

for.end33:                                        ; preds = %for.cond17
  %call34 = call i32 (ptr, ...) @printf(ptr noundef @.str.1)
  %57 = load ptr, ptr %d_input1, align 8
  %call35 = call i32 @cudaFree(ptr noundef %57)
  %58 = load ptr, ptr %d_input2, align 8
  %call36 = call i32 @cudaFree(ptr noundef %58)
  %59 = load ptr, ptr %d_output, align 8
  %call37 = call i32 @cudaFree(ptr noundef %59)
  %60 = load ptr, ptr %h_input1, align 8
  call void @free(ptr noundef %60) #13
  %61 = load ptr, ptr %h_input2, align 8
  call void @free(ptr noundef %61) #13
  %62 = load ptr, ptr %h_output, align 8
  call void @free(ptr noundef %62) #13
  ret i32 0
}

; Function Attrs: nounwind allocsize(0)
declare noalias ptr @malloc(i64 noundef) #3

; Function Attrs: mustprogress noinline optnone uwtable
define internal noundef i32 @_ZL10cudaMallocIfE9cudaErrorPPT_m(ptr noundef %devPtr, i64 noundef %size) #4 {
entry:
  %devPtr.addr = alloca ptr, align 8
  %size.addr = alloca i64, align 8
  store ptr %devPtr, ptr %devPtr.addr, align 8
  store i64 %size, ptr %size.addr, align 8
  %0 = load ptr, ptr %devPtr.addr, align 8
  %1 = load i64, ptr %size.addr, align 8
  %call = call i32 @cudaMalloc(ptr noundef %0, i64 noundef %1)
  ret i32 %call
}

declare i32 @cudaMemcpy(ptr noundef, ptr noundef, i64 noundef, i32 noundef) #5

declare i32 @__cudaPushCallConfiguration(i64, i32, i64, i32, i64 noundef, ptr noundef) #5

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZN4dim3C2Ejjj(ptr noundef nonnull align 4 dereferenceable(12) %this, i32 noundef %vx, i32 noundef %vy, i32 noundef %vz) unnamed_addr #6 comdat align 2 {
entry:
  %this.addr = alloca ptr, align 8
  %vx.addr = alloca i32, align 4
  %vy.addr = alloca i32, align 4
  %vz.addr = alloca i32, align 4
  store ptr %this, ptr %this.addr, align 8
  store i32 %vx, ptr %vx.addr, align 4
  store i32 %vy, ptr %vy.addr, align 4
  store i32 %vz, ptr %vz.addr, align 4
  %this1 = load ptr, ptr %this.addr, align 8
  %x = getelementptr inbounds %struct.dim3, ptr %this1, i32 0, i32 0
  %0 = load i32, ptr %vx.addr, align 4
  store i32 %0, ptr %x, align 4
  %y = getelementptr inbounds %struct.dim3, ptr %this1, i32 0, i32 1
  %1 = load i32, ptr %vy.addr, align 4
  store i32 %1, ptr %y, align 4
  %z = getelementptr inbounds %struct.dim3, ptr %this1, i32 0, i32 2
  %2 = load i32, ptr %vz.addr, align 4
  store i32 %2, ptr %z, align 4
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef float @_ZSt4fabsf(float noundef %__x) #6 comdat {
entry:
  %__x.addr = alloca float, align 4
  store float %__x, ptr %__x.addr, align 4
  %0 = load float, ptr %__x.addr, align 4
  %1 = call float @llvm.fabs.f32(float %0)
  ret float %1
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fmuladd.f32(float, float, float) #7

declare i32 @fprintf(ptr noundef, ptr noundef, ...) #5

; Function Attrs: noreturn nounwind
declare void @exit(i32 noundef) #8

declare i32 @printf(ptr noundef, ...) #5

declare i32 @cudaFree(ptr noundef) #5

; Function Attrs: nounwind
declare void @free(ptr noundef) #9

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fabs.f32(float) #7

declare i32 @cudaMalloc(ptr noundef, i64 noundef) #5

attributes #0 = { mustprogress noinline norecurse optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "uniform-work-group-size"="true" }
attributes #1 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress noinline norecurse optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { nounwind allocsize(0) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #4 = { mustprogress noinline optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #5 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #6 = { mustprogress noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #7 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #8 = { noreturn nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #9 = { nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #10 = { nounwind allocsize(0) }
attributes #11 = { "uniform-work-group-size"="true" }
attributes #12 = { noreturn nounwind }
attributes #13 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5}
!llvm.ident = !{!6}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 11, i32 5]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 8, !"PIC Level", i32 2}
!3 = !{i32 7, !"PIE Level", i32 2}
!4 = !{i32 7, !"uwtable", i32 2}
!5 = !{i32 7, !"frame-pointer", i32 2}
!6 = !{!"clang version 18.1.3 (https://github.com/virnarula/llvm-project.git 7709d36816961113db93da8bb9c3a1d05b7c2c0f)"}
!7 = distinct !{!7, !8}
!8 = !{!"llvm.loop.mustprogress"}
!9 = distinct !{!9, !8}
