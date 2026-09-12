#!/usr/bin/env bash
# Source this file in an activated environment before running native NPBench.
# Optional inputs: NPBENCH_CUDA_ROOT, NPBENCH_NATIVE_PREFIX, NPBENCH_THREADS,
# NPBENCH_CUDA_ARCH. Benchmark and compiler execution policies are unchanged.

_npbench_native_env() {
    local threads="${NPBENCH_THREADS:-2}"
    local prefix="${NPBENCH_NATIVE_PREFIX:-}"
    local cuda_root="${NPBENCH_CUDA_ROOT:-}"
    local native_lib
    if [[ ! "$threads" =~ ^[1-9][0-9]*$ ]]; then
        printf '%s\n' 'NPBENCH_THREADS must be a positive integer' >&2
        return 1
    fi
    if [[ -n "$prefix" && ! -d "$prefix/include" ]]; then
        printf 'Native prefix has no include directory: %s\n' "$prefix" >&2
        return 1
    fi
    if [[ -n "$cuda_root" && ( ! -x "$cuda_root/bin/nvcc" || ! -d "$cuda_root/lib64" ) ]]; then
        printf 'CUDA root needs bin/nvcc and lib64: %s\n' "$cuda_root" >&2
        return 1
    fi
    export OMP_NUM_THREADS="$threads" OPENBLAS_NUM_THREADS="$threads"
    export MKL_NUM_THREADS="$threads" NUMBA_NUM_THREADS="$threads"
    export NUMBA_THREADING_LAYER="${NUMBA_THREADING_LAYER:-omp}"
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
