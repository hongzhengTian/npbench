#!/usr/bin/env bash
# Source this file in an activated environment before running native NPBench.
# Optional inputs: NPBENCH_CUDA_ROOT, NPBENCH_NATIVE_PREFIX, NPBENCH_THREADS,
# NPBENCH_CUDA_ARCH. Benchmark and compiler execution policies are unchanged.

_npbench_native_env() {
    local policy="${NPBENCH_RESOURCE_POLICY:-system}"
    local threads="${NPBENCH_THREADS:-2}"
    local prefix="${NPBENCH_NATIVE_PREFIX:-}"
    local cuda_root="${NPBENCH_CUDA_ROOT:-}"
    local native_lib
    case "$policy" in
        system)
            # Remove repository-added/inherited tuning. Preserve scheduler
            # allocation and let native runtimes select threads and placement.
            unset NPBENCH_THREADS NPBENCH_CPUSET
            unset OMP_NUM_THREADS OMP_THREAD_LIMIT OMP_PROC_BIND OMP_PLACES OMP_DYNAMIC OMP_MAX_ACTIVE_LEVELS OMP_NESTED
            unset OPENBLAS_NUM_THREADS GOTO_NUM_THREADS MKL_NUM_THREADS MKL_DYNAMIC NUMBA_NUM_THREADS NUMBA_THREADING_LAYER
            unset BLIS_NUM_THREADS VECLIB_MAXIMUM_THREADS GOMP_CPU_AFFINITY KMP_AFFINITY KMP_HW_SUBSET TBB_NUM_THREADS
            unset COVA_LLVM_CPU_HELPERS_OPENBLAS_THREADS COVA_LLVM_CPU_HELPERS_GEMV_OPENBLAS_THREADS
            unset COVA_LLVM_CPU_PAIRED_GEMV_THREADS COVA_LLVM_CPU_CHAINED_GEMV_THREADS COVA_LLVM_CPU_REDUCTION_THREADS
            ;;
        fixed)
            if [[ ! "$threads" =~ ^[1-9][0-9]*$ ]]; then
                printf '%s\n' 'NPBENCH_THREADS must be a positive integer in fixed mode' >&2
                return 1
            fi
            export NPBENCH_THREADS="$threads"
            export OMP_NUM_THREADS="$threads" OPENBLAS_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads" NUMBA_NUM_THREADS="$threads"
            export NUMBA_THREADING_LAYER="${NUMBA_THREADING_LAYER:-omp}"
            export OMP_PROC_BIND=true OMP_PLACES=cores
            ;;
        *) printf '%s\n' 'NPBENCH_RESOURCE_POLICY must be system or fixed' >&2; return 1;;
    esac
    export NPBENCH_RESOURCE_POLICY="$policy"
    if [[ -n "$prefix" && ! -d "$prefix/include" ]]; then
        printf 'Native prefix has no include directory: %s\n' "$prefix" >&2
        return 1
    fi
    if [[ -n "$cuda_root" && ( ! -x "$cuda_root/bin/nvcc" || ! -d "$cuda_root/lib64" ) ]]; then
        printf 'CUDA root needs bin/nvcc and lib64: %s\n' "$cuda_root" >&2
        return 1
    fi
    export MPLBACKEND="${MPLBACKEND:-Agg}"
    if [[ -n "$prefix" ]]; then
        export CPATH="$prefix/include${CPATH:+:$CPATH}"
        export CMAKE_PREFIX_PATH="$prefix${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
        for native_lib in "$prefix/lib" "$prefix/lib64" "$prefix/lib/x86_64-linux-gnu"; do
            if [[ -d "$native_lib" ]]; then
                export LIBRARY_PATH="$native_lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
                export LD_LIBRARY_PATH="$native_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            fi
        done
    fi
    if [[ -n "$cuda_root" ]]; then
        export CUDA_PATH="$cuda_root" CUDA_HOME="$cuda_root"
        export CUDACXX="$cuda_root/bin/nvcc"
        export DACE_compiler_cuda_path="$cuda_root"
        export PATH="$cuda_root/bin:$PATH"
        export LD_LIBRARY_PATH="$cuda_root/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        if [[ -n "${NPBENCH_CUDA_ARCH:-}" ]]; then
            export DACE_compiler_cuda_cuda_arch="$NPBENCH_CUDA_ARCH"
        fi
    fi
}
_npbench_native_env
_npbench_native_status=$?
unset -f _npbench_native_env
return "$_npbench_native_status" 2>/dev/null || exit "$_npbench_native_status"
