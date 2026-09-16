# Native NumPy, Numba, DaCe and CuPy environment

These optional profiles target CPython 3.10 and can extend an existing compatible virtual environment, including CoVA's comparison environment.
They prepare dependencies and native-library discovery while leaving upstream benchmark implementations, framework adapters, runners, and validation unchanged.
A configured environment does not guarantee that every benchmark compiles, executes, validates, or meets a performance target.
They do not install CoVA, import its harness, or require adjacent checkouts.
The ordinary NPBench installation remains available; users select these profiles only when they need these comparison frameworks.

## Python dependencies

Activate the intended virtual environment, then run the following from this checkout.
Inspect the dry run before extending a shared environment; stop on conflicting constraints instead of upgrading unrelated packages.

```bash
python -m pip install --dry-run -r requirements/native-cpu.txt
python -m pip install -r requirements/native-cpu.txt
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
```

For CuPy and DaCe GPU, use `requirements/native-cuda12.txt` in both dependency commands instead.
Install only one CuPy distribution in the environment.
The CUDA profile selects CuPy 13.6 to retain NumPy 1.26 compatibility with DaCe 1.0.2.
The CPU profile includes threadpoolctl for observing loaded native thread pools outside timed calls.
The version constraints preserve the established shared compiler versions; the setup does not modify an installed DaCe package.
This is a reproducible environment selection, not an upstream guarantee that every implementation supports these versions.
An editable installation keeps benchmark sources and JSON metadata available from this checkout, as expected by NPBench's current package layout.

## Native libraries and activation

DaCe CPU requires a C++ compiler, CMake, and a BLAS/LAPACK implementation.
On Debian/Ubuntu, the relevant development packages are `build-essential`, `cmake`, `libopenblas-dev`, and `liblapacke-dev`.
Installing the Python `dace` package alone does not provide `lapacke.h` or `liblapacke`.
A user-owned native prefix is also supported: put headers in `include` and libraries in `lib`, `lib64`, or `lib/x86_64-linux-gnu`, then set `NPBENCH_NATIVE_PREFIX`.
Use native packages matching the operating system; extracting distribution packages into a prefix does not resolve their dependencies automatically.
For an Ubuntu system that already provides OpenBLAS, a user-owned LAPACKE prefix can be prepared without changing system packages:

```bash
mkdir -p .cache/native-debs .cache/native
(cd .cache/native-debs && apt-get download liblapacke-dev liblapacke libtmglib-dev libtmglib3)
for package in .cache/native-debs/*.deb; do
    dpkg-deb --extract "$package" .cache/native
done
export NPBENCH_NATIVE_PREFIX="$PWD/.cache/native/usr"
```

Record the downloaded package versions and checksums with the run evidence.

```bash
export NPBENCH_RESOURCE_POLICY=system
# Optional when LAPACKE is outside system search paths:
export NPBENCH_NATIVE_PREFIX=/path/to/native/usr
source scripts/native-env.sh
```

GPU execution additionally requires a working NVIDIA driver and a complete CUDA 12 toolkit, including cuBLAS, cuSOLVER, cuSPARSE, NVRTC and nvJitLink.
Select one toolkit consistently for compilation and runtime libraries.
A wheel-only CUDA runtime is insufficient for DaCe's native CUDA compilation.

```bash
export NPBENCH_CUDA_ROOT=/path/to/cuda-12
export NPBENCH_CUDA_ARCH=80  # Example: use the actual device's compute capability.
# Optional when the default host compiler is unsupported by the toolkit:
export CUDAHOSTCXX=/path/to/supported/g++
source scripts/native-env.sh
python -c 'import cupy as cp; print(cp.cuda.runtime.getDeviceProperties(0)["name"]); print(cp.arange(8).sum().item())'
```

The script sets `CUDACXX` as well as CUDA library paths, because CMake can otherwise discover a different `nvcc`.
After changing toolkits, use a new run directory so an existing `.dacecache` does not reuse an old CMake compiler selection.
The script configures native-library paths, compiler selection and GPU architecture.
Its default system resource policy clears inherited thread/binding overrides and preserves scheduler allocation while allowing native runtime defaults.
For explicit diagnostic budgets, set NPBENCH_RESOURCE_POLICY=fixed and NPBENCH_THREADS before activation.
See [Reproducible L collections](formal-collection.md) for resource verification and optional NVML monitoring configuration.
It does not override DaCe's CUDA stream policy or configure a device malloc heap.
Independent shell variables and DaCe configuration files can still affect execution; start from a fresh shell when checking defaults.

## Running upstream implementations

Check the installation and CLI first:

```bash
python -m pip check
python run_benchmark.py --help
```

Use the upstream CLI and presets, for example:

```bash
python run_benchmark.py -b gemm -f numpy -p S -r 1 -t 600
python run_benchmark.py -b gemm -f numba -p S -r 1 -t 600
python run_benchmark.py -b gemm -f dace_cpu -p S -r 1 -t 600
python run_benchmark.py -b gemm -f cupy -p S -r 1 -t 600
python run_benchmark.py -b gemm -f dace_gpu -p S -r 1 -t 600
```

These commands demonstrate invocation, not a promise that each implementation succeeds.
The current upstream timing helper makes an additional call to retrieve outputs after its timed calls, in both the first/validation and steady-state phases.
Compilation of larger programs can take several minutes; `-t` limits the first execution, not the entire process or DaCe's preceding compilation.
A short run with `-r 1` does not establish performance results.

Use a new working directory after source or environment changes so generated artifacts from another configuration are not reused.
The CLI script can be invoked by absolute path while SQLite files and compiler caches remain in the run directory.
For example, from this checkout:

```bash
npbench_root="$PWD"
mkdir -p "$npbench_root/.cache/runs"
npbench_run_dir="$(mktemp -d "$npbench_root/.cache/runs/upstream-gemm.XXXXXX")"
(
    cd "$npbench_run_dir" || exit 1
    npbench_exit_status=0
    python "$npbench_root/run_benchmark.py" -b gemm -f numba -p S -r 1 -t 600 > stdout.log 2>&1 || npbench_exit_status=$?
    printf '%s\n' "$npbench_exit_status" > exit-code.txt
    exit "$npbench_exit_status"
)
```

Keep the exact command, upstream commit, local diff, dependency versions, environment settings, standard output/error, exit status, and `npbench.db` with each run.
Distinguish missing sources and environment problems from compilation errors, execution errors, numerical validation errors, and timeouts in the run notes.
Do not change the implementation to turn a failure into success as part of environment setup.

The upstream runner can catch errors without returning a failing process status, and its validator can miss absent outputs or misreport validation exceptions.
Therefore, neither exit code zero nor a stored validation flag alone proves correctness.
Keep raw observations and mark inconclusive validation as unverified; do not infer a pass from missing error reports.
The environment profiles do not repair this upstream reporting behavior.

The checkout contains 54 NumPy benchmarks, 152 Numba variants, and 53 sources each for CuPy and DaCe.
DaCe attempts `fusion`, `parallel`, and `auto_opt` variants from each source for CPU and GPU.
`covariance2` contains only NumPy and Numba implementations; missing sources are not successful tests.
Results obtained with modified implementations or compiler behavior belong to a separate configuration and must not be presented as results from unmodified upstream sources.
