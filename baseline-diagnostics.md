# Targeted baseline diagnostics

Activate the existing NPBench environment, then run from the checkout root:

```bash
./run_baseline_diagnostics.sh --dry-run
./run_baseline_diagnostics.sh all .cache/baseline-diagnostics/baseline-check
```

This is the historical functional diagnostic matrix; it is separate from the final performance campaign in [Reproducible L collections](formal-collection.md).
To reproduce its original resource budget explicitly, set NPBENCH_RESOURCE_POLICY=fixed and NPBENCH_THREADS=2.
An already activated system policy otherwise remains in effect.
It invokes the existing lifecycle runner with exact implementation lists, never CoVA.
Results go to new isolated directories; historical measurements, goldens and artifacts are not overwritten.
Existing S goldens are reused and missing ones may be prepared once.
For optional L spot checks, existing goldens are required:

```bash
./run_baseline_diagnostics.sh large_spots .cache/baseline-diagnostics/large-check
```

## Scope and interpretation

| Group | Preset / implementations | Calls and purpose |
| --- | --- | --- |
| correctness | S / 4 | 6 subsequent calls in each of 2 processes; cholesky/syrk auto_opt and parallel control |
| contracts | S / 8 | 1 initialization + 1 subsequent call; return/declared-output/extra-state diagnostics |
| restore | S / 4 | 1 subsequent call in each of 2 processes; channel_flow/softmax CPU/GPU auto_opt |
| performance | S / 12 | 2 subsequent calls in each of 3 processes; resource observations and process grouping on selected anomalous workloads |
| p2 | S / 23 | 1 initialization + 1 subsequent call; representatives of old numerical, lowering, prepare, illegal-access, termination and timeout failures |
| large_spots | L / 5, opt-in | cholesky/syrk auto_opt: 6 subsequent calls in 2 processes; floyd_warshall CPU parallel and spmv GPU fusion/parallel: 1 subsequent call in 2 processes |

`all` includes the first five groups: 51 implementation cells, up to 242 lifecycle calls if every call succeeds.
P2 is representative diagnosis, not an exhaustive rerun of every failed variant.
The known seven missing covariance2 DaCe/CuPy sources are not scheduled because execution cannot supply missing implementations.
Detailed case lists live in `scripts/baseline_diagnostic_plan.py` and are saved as cases.tsv with each group.
The default timeout is 240 seconds per worker, including compilation and validation; change NPBENCH_TIMEOUT explicitly if necessary.
No overnight completion time is promised.
Small-scale success is not proof that a former L-scale failure is resolved, nor a replacement for L performance data.

Completed group cells and direct checks, including failed receipts, are skipped on resume with the same code/configuration.
Changed code or plans require a new directory.
Expected benchmark failures do not stop later groups; the script exits 1 if any collection/check failed or was partial.
The new script is separate from retry_affected.sh, whose 282-cell selection describes the previous cache incident.

## Framework corrections and remaining verification

The generated DaCe GPU programs examined use nonblocking internal CUDA streams.
Their public CompiledSDFG call checks cudaGetLastError, without necessarily waiting for GPU work.
The old adapter synchronized only CuPy's current stream, after copying results back.
That did not establish completion of DaCe streams before observing outputs.
Region now synchronizes immediately after execution and before any host materialization; DaCe GPU synchronization waits for the device.
This is a framework boundary correction, not a workload algorithm change.
It explains a concrete race opportunity but does not prove that all observed cholesky/syrk failures are resolved without execution.

The original diagnostic records used metric_version=3 and protocol_version=4.
Current records retain metric_version=3 with protocol_version=5 for resource provenance and structured preparation failures.
Old DaCe GPU durations and successful validations remain historical observations with an insufficient completion guarantee, not certified GPU baselines.
Do not pool old and new GPU times; determine the required broader revalidation after targeted checks.
CPU and CuPy workloads are unchanged, but their new records also identify the new boundary version explicitly.
Device-wide synchronization assumes isolated benchmark activity, as enforced within a checkout by the existing collection lock; unrelated GPU jobs can still affect timing.

The bounded restore adapter first uses DaCe's official loader.
Only on its exact offset-rank error may it normalize all-zero offsets in nested Array descriptors whose strides already match shape rank.
It does not modify top-level ABI descriptors, shape, strides, nonzero offsets, original program.sdfg or executable bytes.
The change leaves the offset contribution to address calculation zero and is recorded in dace-restore.json.
Other restoration failures remain errors; no compilation fallback is allowed.
The restore probes distinguish official deserialization from bounded normalization; inspect their saved receipts and numerical observations for the specific run being analyzed.

After correctness collection, an independent direct-artifact probe runs cholesky and syrk without Region or the framework execution wrapper.
It verifies recorded artifact identity, prohibits SDFG compilation, synchronizes before reading results, and validates seven calls against the same golden.
These diagnostic timings have a different scope and are explicitly excluded from lifecycle performance samples.
After restore collection, metadata-only probes record raw and normalized official SDFG deserialization results.
Each direct probe stores its command output, inspection and terminal receipt; incomplete attempts retain separate directories.
If the collection could not create an artifact, the corresponding probe reports that fact rather than compiling a replacement.

## Validation and resource evidence

The primary strict region validation remains unchanged; no tolerances or original workload files were changed.
On a failure, validation_audit distinguishes strict declared outputs from additional input-array state checks.
It also reports the original runner's numerical output projection when output counts agree.
Different output counts are labelled incomparable_output_count rather than reproducing the original zip truncation; projection success never overrides strict failure.
This diagnostic projection is not an independent run of the native runner.

Resource observations occur outside the call timer, once per worker after its first completed call.
They include affinity, live OS thread count, configured native thread pools when threadpoolctl is available, Numba settings, CUDA driver/runtime versions, GPU identity and input-array device IDs.
The CUDA driver current-context UUID is used when CuPy does not expose UUID directly; failed probes remain explicit.
Configured pools/live thread counts do not prove active utilization, and array placement does not prove every generated operation executed on GPU.
The diagnostic script additionally enables NPBENCH_RESOURCE_PROBE=1 to record per-call Linux thread CPU-time deltas outside the call timer.
These identify observed thread work, with tick resolution, exited threads, child-process exclusion and snapshot overhead explicitly labelled.
Sub-tick kernels may show zero deltas; initialization may include runtime work unrelated to steady execution.
This option is disabled by default for ordinary collections.
No dependency is installed to obtain metadata.

After the user run, first inspect correctness versus direct-artifact results, then restoration and contract diagnostics, then resource and per-process timing distributions.
Only identified framework issues warrant general fixes and corresponding affected-case reruns.
Upstream workload fixes, if later needed, must use separate implementation files and labels so original competitors remain distinguishable.
