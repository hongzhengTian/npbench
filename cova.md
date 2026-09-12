# CoVA PolyBench integration

This fork adds CoVA implementations for 28 PolyBench workloads through NPBench's existing framework interface.
Each workload contains six files such as `adi_cova_llvm_cpu_serial.py`, each defining one `@cova`-decorated `kernel`.
They use the public `from cova import cova` interface and do not import CoVA's benchmark harness or `run_top_*` entry points.
The original NumPy, Numba, DaCe, CuPy, and other implementations, runner, validator, and presets remain unchanged.
Available files describe the attempted coverage; they do not guarantee that every backend compiles, executes, or validates.

## Framework selection

| NPBench framework (`-f`) | CoVA backend | CoVA tool |
| --- | --- | --- |
| `cova_llvm_cpu_serial` | `cpu` | `llvm-cpu-serial` |
| `cova_llvm_cpu` | `cpu` | `llvm-cpu` |
| `cova_serial_cpu` | `cpu` | `serial-cpu` |
| `cova_openmp_cpu` | `cpu` | `openmp-cpu` |
| `cova_llvm_gpu` | `gpu` | `llvm-gpu` |
| `cova_openmp_gpu` | `gpu` | `openmp-gpu` |

The workloads are `adi`, `atax`, `bicg`, `cholesky`, `correlation`, `covariance`, `doitgen`, `durbin`, `fdtd_2d`, `gemm`, `gemver`, `gesummv`, `gramschmidt`, `heat_3d`, `jacobi_1d`, `jacobi_2d`, `k2mm`, `k3mm`, `lu`, `ludcmp`, `mvt`, `nussinov`, `seidel_2d`, `symm`, `syr2k`, `syrk`, `trisolv`, and `trmm`.
Other NPBench workloads have no CoVA implementations in this integration.

## Environment

First build and activate CoVA using its [installation guide](https://github.com/hongzhengTian/CoVA/blob/main/docs/getting-started/installation.md).
Install NPBench into that same Python environment following the root README or the optional [native environment profiles](native-environment.md).
CoVA is optional and is not installed by NPBench's default requirements or the comparison profiles.
The two repositories can be located independently.
For an existing CoVA source/build installation, expose its public Python package, bindings, and tools as documented by CoVA:

```bash
export COVAPATH=/path/to/CoVA
source "$COVAPATH/covadev/bin/activate"
export PATH="$COVAPATH/build/bin:$PATH"
export PYTHONPATH="$COVAPATH/build/python_packages/cova:$COVAPATH${PYTHONPATH:+:$PYTHONPATH}"
export NPBENCH_COVA_VERSION="$(git -C "$COVAPATH" rev-parse HEAD)"
python -c 'from cova import cova; print("CoVA import available")'
```

`NPBENCH_COVA_VERSION` is written into NPBench's ordinary version column.
Without this override, the adapter uses installed `cova` distribution metadata when present, otherwise `source-unversioned`.
Record any CoVA working-tree diff separately; a commit identifier alone does not describe an edited build.
Use the same explicit CPU thread budget as other CPU frameworks.
The word `serial` describes the selected compiler route; linked BLAS libraries still follow their own thread configuration.

GPU runs additionally need a working CUDA driver/toolkit, and OpenMP GPU requires an offload-capable NVHPC `nvc++`.
Load the site toolchain or expose it explicitly:

```bash
export ROOT_NVHPC=/path/to/nvhpc/Linux_x86_64/version
export COVA_OPENMP_GPU_CXX="$ROOT_NVHPC/compilers/bin/nvc++"
export OMP_TARGET_OFFLOAD=MANDATORY
```

Follow CoVA's installation guide for CUDA and architecture overrides.
CoVA may resolve CUDA libraries from the selected NVHPC installation; retain the generated compiler and backend metadata to identify the effective toolchain.
A GPU framework name records the requested route, and is not proof of actual GPU execution.
Check the `.cova_gen` backend metadata for `actual_device`, `runtime_variant`, and `fallback_reason`; CPU or mixed fallbacks must be reported explicitly.
`OMP_TARGET_OFFLOAD=MANDATORY` prevents ordinary OpenMP target fallback, but does not disable every CoVA compiler-level fallback.

## Running

Use the standard NPBench CLI and an independent working directory:

```bash
npbench_root="$PWD"
mkdir -p .cache/runs
npbench_run_dir="$(mktemp -d "$npbench_root/.cache/runs/cova-gemm.XXXXXX")"
(cd "$npbench_run_dir" && python "$npbench_root/run_benchmark.py" \
    -b gemm -f cova_llvm_cpu_serial -p S -r 1 -t 120)
```

Select another row from the table for another backend, and replace `gemm` to test another supported workload.
`S` is NPBench's smallest standard preset; no CoVA-specific execution preset is introduced.
Compilation occurs inside the first decorated invocation, while later invocations can reuse CoVA artifacts.
The upstream timing helper also makes untimed output-retrieval calls; `-r 1` does not mean that the kernel is called only once.
Keep `npbench.db`, stdout/stderr, exact commands, repository revisions/diffs, environment settings, and generated backend metadata with the run evidence.

## Return and writeback protocol

CoVA can return an updated array instead of modifying the original Python input object.
The adapter restores NPBench's observable return and mutation behavior without changing the upstream runner.
Each benchmark's optional `cova` metadata contains `return_count`, the number of reference return values, and `writeback_args`, the input arrays that the computation updates.
A CoVA kernel explicitly returns the reference results first, followed by those writeback arrays in the declared order.
The framework checks the output count and writeback shapes/dtypes, copies the updated arrays into the original input objects, and exposes only the reference results to NPBench.
For example, GEMM returns `C` from its decorated kernel and declares zero reference returns plus writeback to `C`; NPBench observes an in-place update and a `None` return.
This protocol also covers updates missing from upstream `output_args`, such as `doitgen.A` and `trisolv.x`, without modifying the original validation declarations.

The implementations were adapted from CoVA revision `3727be5e2848befa897531cc31b894e9382d69d2`.
The computational bodies match this NPBench checkout, with Nussinov's small `match` helper inlined to retain one kernel per file.
ADI deliberately uses NPBench's `b = 1.0 + mul2` coefficient, which differs from CoVA's existing canonical PolyBench version (`mul1`).
This choice follows the local reference implementation and is not a correction of upstream ADI.
CoVA's main repository and existing PolyBench implementations are unchanged.

## Validation and timing limits

The adapter uses host NumPy arrays for all six routes because these are the current public CoVA native bindings.
Input cloning occurs in NPBench setup, while the measured CoVA call includes public decorator dispatch, native execution, GPU transfers/synchronization where applicable, and adapter writeback.
CuPy and DaCe GPU normally receive device arrays prepared outside the measured call, so these measurements have different data-transfer boundaries and must not be presented as directly comparable kernel-only performance.
No private native entry point or device-resident shortcut is used to hide this distinction.

The original NPBench validator can miss missing outputs, unused declared outputs, or validation exceptions, and its runner can exit zero after a failure.
A stored `validated=1` is therefore not a complete correctness guarantee.
The adapter does not change these upstream behaviors; inspect raw logs and independently check return count, shape, dtype, finiteness, and updated input arrays when validating the integration.
Retain compilation, execution, numerical, device-fallback, and timeout observations without repairing competing implementations.

Run the focused protocol and source-adaptation tests without compiling CoVA:

```bash
python -m unittest discover -s tests -p test_cova.py -v
```

These tests compare small Python executions of all transplanted kernels with the independent NPBench NumPy implementations and check adapter rejection/writeback behavior.
They do not establish native compiler or GPU support; native S-preset runs and their device evidence remain separate.

For persistent inputs/reference outputs and initialization, fresh-process and same-process timing, see [lifecycle measurements](lifecycle.md).
