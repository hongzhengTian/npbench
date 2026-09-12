# Reusable references and call lifecycles

These optional features use NPBench's original inputs, presets, NumPy reference, framework implementations and numerical tolerances.
The default command still uses the original runner and `results` table.
No competitor kernel is changed.

## Golden input and output bundles

Generate a reference once, then reuse it across implementations and processes:

```bash
python run_benchmark.py -b gemm -p S --prepare-golden --golden-cache /path/to/goldens
python run_benchmark.py -b gemm -f numba -p S --golden-cache /path/to/goldens --require-golden
```

A bundle contains the actual input values, NumPy returns and the final state of every array argument.
A cache hit imports neither the initializer nor the reference implementation and runs neither function.
`--require-golden` fails on a miss, so a later performance job cannot unexpectedly generate a reference.
Without that flag, a missing entry is generated with one initializer execution and one reference call.
A POSIX file lock prevents concurrent requests for the same entry from generating it repeatedly.
NumPy remains needed to load and compare arrays.

Identity includes the benchmark, selected preset parameters, input contract, initializer/reference source and their statically discoverable local Python imports.
Changing the tested implementation, another preset or comparison tolerances does not invalidate unchanged values.
Changing tolerances rechecks the stored values using the current policy.
External package versions are provenance; they do not silently replace existing golden values.
If a reference depends on external data, dynamically imported code or external configuration, use a separate cache directory when those numerical inputs change.
The initial scope is the numerical Python/PolyBench benchmarks; object arrays are rejected.

Bundles use JSON and non-pickle NumPy arrays, integrity checks, and atomic publication.
Corrupt entries fail explicitly and are never silently regenerated.
Both returns and array states are checked for count, shape, dtype, finite values and the benchmark's original tolerance/norm-error rule.
This detects missing outputs and mutations omitted from upstream `output_args` without editing the benchmark implementation.
`--golden-cache` also works with the native runner, where it replaces reference execution but leaves native timing unchanged.

## Host-to-host lifecycle measurements

```bash
python run_benchmark.py -b gemm -f cova_llvm_cpu -p S \
  --lifecycle --golden-cache /path/to/goldens --require-golden \
  --fresh-process-runs 3 -r 5 -t 600
```

`--lifecycle` selects the optional measurement protocol.
The default golden directory in this mode is `.cache/goldens`.
`--run-dir` changes the parent for run records and isolated artifacts; the default is `.cache/lifecycle`.
Every command creates a new run directory and an empty artifact directory per implementation.
A later process shares that implementation's artifacts.
The timeout bounds each complete worker, including imports and checks, rather than only its timed kernel call.

| Phase | Measured call |
| --- | --- |
| `initialization` | First call in a new artifact directory, including lazy compilation and runtime initialization. |
| `fresh_process` | First call in each later Python process, including artifact loading and process-local runtime initialization. |
| `same_process` | Each of the subsequent `-r` calls within every process. |

Each call consumes a fresh copy of the saved host inputs and returns observable host results.
Source-module imports, decorator construction and adapter preparation finish before timing.
The timer covers lazy compilation or artifact loading, the decorated kernel/compiled application call, necessary host/device transfers, synchronization, result materialization and required array writeback.
Input reset, golden loading, numerical comparison, artifact auditing and Python process startup/teardown are outside this timer.
The parent records complete worker lifetime separately.
A timed call's own output is validated directly; no extra execution is performed just to retrieve a result.

Required host writebacks include declared output arguments and arrays whose final reference values differ from their inputs.
All remaining array arguments are also audited, with read-only device-array inspection outside the timer.
This avoids adding unnecessary device-to-host transfers to the required region result.
The resulting metric is an adapted host-to-host call, not a bare device-resident kernel time.
CoVA already consumes host arrays; CuPy and DaCe GPU conversions now occur inside this optional region.

NumPy, Numba, DaCe CPU/GPU, CuPy and the six CoVA routes have lifecycle adapters.
Use `--implementation LABEL` to select one variant, for example `nopython-mode` for Numba or `fusion`, `parallel`, `auto_opt` for DaCe.
Numba's existing dispatcher has disk caching enabled only in this mode.
Discovery uses the implementation files actually present in the benchmark.
If an upstream Numba file exposes an ordinary Python function, it remains Python and is recorded with `artifact_policy=python`; the adapter does not add the missing decorator or claim JIT execution.
Pure CuPy library calls may need no generated cache entry and use `artifact_policy=library_cache`.
DaCe uses the upstream transformation recipes for the selected variant and restores the resulting compiled SDFG in later processes.
Compilation or restore failures remain failures; missing artifacts do not trigger a replacement cold build disguised as reuse.
CoVA and CuPy use their native persistent caches in isolated directories.
No cache support is claimed for other frameworks.
The initialization observation leaves installed compilers, OS caches and driver caches intact.

With `--fresh-process-runs N -r R`, a successful implementation has `N + 1` workers and `(N + 1) * (R + 1)` timed calls.
There is one initialization observation, `N` fresh-process observations, and `(N + 1) * R` same-process observations.
Workers require a golden hit and never execute its producer.
`-v false` skips numerical comparison, but the lifecycle input bundle is still prepared or loaded; use `--require-golden` to forbid reference generation.
The corresponding records have `validated = NULL`, not a passing validation flag.

## Results and failures

Lifecycle observations go to `lifecycle_results` in the working directory's `npbench.db`.
They never enter the native `results` table, so the existing plotting scripts cannot silently mix the two timing protocols.
The new table retains phase, process/call indices, validation, time, status, golden identity and a full JSON observation.

```sql
SELECT framework, implementation, phase, process_index, call_index, time, status
FROM lifecycle_results
WHERE benchmark = 'gemm'
ORDER BY id;
```

The run's `manifest.json` records arguments, versions, source identities, golden generation/hit evidence, selected environment and worker outcomes.
Per-process JSONL observations and stdout/stderr remain beside the artifacts.
Native artifact hashes and modification times distinguish unchanged reuse from builds or changes; new-process and same-process measurements with changed artifacts are marked `artifact_reuse_failed`.
NumPy has no compiled artifact requirement.
CoVA placement metadata separately identifies GPU execution, CPU fallback and mixed execution; numerical success alone does not qualify a GPU-only comparison.
A failed or aborted worker stops later processes for that implementation, while other implementations can still be collected.
No failed measurement is assigned a zero time or included as a successful sample.
A process exit before a callable starts has no call duration.

These records support lifecycle experiments; they do not by themselves establish fair resource allocation or publication-quality performance.
Run competing measurements sequentially with consistent hardware, thread counts, software configuration and inputs, and retain failure/device coverage when comparing results.
