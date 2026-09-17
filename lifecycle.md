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
  --fresh-process-runs 2 -r 2 -t 600
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
The parent verifies the golden without retaining its arrays, and workers release each call's inputs/results after validation and before preparing the next call.
This avoids retaining an extra workload in host/device memory during subsequent measurements.
Current observations carry `metric_version=3` and `protocol_version=5`: the framework synchronizes execution before copying results to the host, and DaCe GPU waits for all device streams.
Version-2 DaCe GPU observations did not guarantee completion of nonblocking DaCe streams before copyback and must not be pooled with corrected measurements.
Earlier records without metric_version also used different buffer retention.
See [Targeted baseline diagnostics](baseline-diagnostics.md) for the bounded restore adaptation, validation audit, resource evidence and user-run verification matrix.

Required host writebacks include declared output arguments and arrays whose final reference values differ from their inputs.
All remaining array arguments are also audited, with read-only device-array inspection outside the timer.
This avoids adding unnecessary device-to-host transfers to the required region result.
The resulting metric is an adapted host-to-host call, not a bare device-resident kernel time.
CoVA already consumes host arrays; CuPy and DaCe GPU conversions now occur inside this optional region.

NumPy, Numba, DaCe CPU/GPU, CuPy and the six CoVA routes have lifecycle adapters.
Use `--implementation LABEL` to select one variant, for example `nopython-mode` for Numba or `fusion`, `parallel`, `auto_opt` for DaCe.
Numba's existing dispatcher has disk caching enabled only in this mode.
Protocol 6 adds entry-dispatcher cache observations outside the timed region; metric_version remains 3.
Each Numba call records copied before/after dispatcher statistics, signature lists, hit/miss deltas and a reuse verdict.
For numba_disk_cache, a fresh-process call requires at least one disk hit and zero misses; a same-process call requires unchanged nonempty signatures and no new hit or miss.
Missing counters, cache misses or contradictory observations fail the requested reuse with artifact_reuse_failed, even when artifact files remain unchanged.
An initialization cache miss is expected and is not a reuse failure.
This proves the observed entry-dispatcher path, not all possible nested runtime behavior.
Historical protocol-5 files do not gain this evidence retroactively: the analyzer reports not_recorded for missing dispatcher observations.
Plain Python implementations under the Numba label remain python_no_dispatcher and are not treated as disk-cache hits.

After compilation, specializations containing non-cacheable lifted code or dynamic globals are recorded as `artifact_policy=numba_memory_only`.
Their initialization and same-process calls are still measured and validated.
Requested later-process observations are recorded as `reuse_unsupported`, without a duration or validation flag, and those processes are not launched.
No cold recompile is substituted for artifact reuse.
A run containing only passed calls and unsupported reuse observations has manifest status `partial` and exit code 2; actual failures still produce exit code 1.
With no later processes requested, a successful memory-only run can finish with exit code 0.
Discovery uses the implementation files actually present in the benchmark.
If an upstream Numba file exposes an ordinary Python function, it remains Python and is recorded with `artifact_policy=python`; the adapter does not add the missing decorator or claim JIT execution.
Pure CuPy library calls may need no generated cache entry and use `artifact_policy=library_cache`.
DaCe workers override inherited `DACE_default_build_folder`, `DACE_cache`, and `DACE_compiler_use_cache` with an isolated per-implementation cache, stable names, and disabled implicit compiler-cache reuse.
The manifest records the effective worker cache environment.
The artifact record stores a path relative to the isolated run directory and rejects paths outside it.
Later workers compare native artifact hashes and modification times with the preceding worker before importing the implementation.
Missing or changed artifacts remain failures, distinct from an unsupported persistence capability.
DaCe deserialization errors retain their original cause and are recorded with `failure_stage=restore`; the loader never substitutes compilation for a failed restore.
The DaCe adapter unwraps a one-element return array only when the reference return slot is scalar.
It preserves the actual dtype; array-valued slots, wrong sizes, extra/missing returns and dtype mismatches remain validation failures.
This restores the scalar call ABI without modifying benchmark code or numerical tolerances.
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
Validation observations include `validation_failures` identifying return count, array keys, field shape/dtype, nonfinite counts, or numerical-value rejection with the unchanged tolerance policy.
Diagnostics are collected outside the timer and do not cast results, remove outputs, or relax comparisons.
A failed or aborted worker stops later processes for that implementation, while other implementations can still be collected.
No failed measurement is assigned a zero time or included as a successful sample.
A process exit before a callable starts has no call duration.
Stopping the lifecycle parent with SIGTERM or SIGINT also stops its active worker and compiler process group.

These records support lifecycle experiments; they do not by themselves establish fair resource allocation or publication-quality performance.
Run competing measurements sequentially with consistent hardware, thread counts, software configuration and inputs, and retain failure/device coverage when comparing results.

## Collections and later CoVA-only runs

See [Reproducible L collections](formal-collection.md) for the system-resource main matrix, independent initialization trials, finite resource supplements, golden reuse, resume rules, conditional analysis and portable evidence.
The main default is 3 processes with 3 calls each, including each process's first call.
Historical two-thread measurements have a different resource policy and remain diagnostic evidence.

For failures in the original pre-protocol-3 collection, retry_affected.sh selects only the baseline implementations affected by the prior cache-location, cache-capability and confirmed scalar-return fixes.
Its old selection remains historical; use a new exact selection with run_large.sh for other follow-ups.

### Explicit shared golden references

An opt-in benchmark variant may set `golden_reference` in its benchmark metadata to name a canonical benchmark.
The input/array/output arguments, function and initialization contract, and selected preset parameters must match; reference chains are rejected.
The canonical benchmark owns reference source identity and input/output generation, so the variant reuses exactly the same golden key and payload, including otherwise unseeded random inputs.
This does not change measurement code or validation tolerances; the variant must still pass the reference contract.
Use a separate reference for mathematical workload changes such as `adi_corrected`.
