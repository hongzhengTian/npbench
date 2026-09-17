# Explicit workload repairs

Original upstream modules and benchmark registrations remain unchanged.
These `optional: true` registrations are excluded from the default campaign; select them by name or exact cases file.
They use the existing NPBench runner, validation and lifecycle measurements.
Each new implementation module preserves the original JIT options and differs only for the documented repair.
Do not describe measurements of these variants as unmodified upstream NPBench results.

| Registration | Affected implementations | Change | Reference |
| --- | --- | --- | --- |
| `adi_corrected` | NumPy, Numba n/np, DaCe CPU/GPU, CuPy | Canonical `b = 1 + mul1`; return the mutated `u` | New canonical formula and separate golden; never pool with original ADI |
| `correlation_return` | CuPy | Return the already-computed `corr` | Imports original NumPy reference and initializer |
| `mlp_rowmax` | Numba n/np/npr | Compute each row's maximum for stable softmax, with disjoint row writes | Imports original NumPy reference and initializer |
| `azimint_hist_private` | Numba npr | Accumulate in independently owned chunk histograms, then reduce | Imports original NumPy reference and initializer |

The registrations have separate names.
`mlp_rowmax` explicitly sets `golden_reference: "mlp"` to reuse the original benchmark's exact input/output bundle: the upstream initializer includes an unseeded random input, so re-running it would not reproduce the historical inputs.
The other registrations use their own golden keys.
This uses existing discovery and golden ownership without adding framework-specific bypasses or weakening output, dtype, finite-value or mutation checks.
The initializer inputs, L sizes and numerical tolerance remain unchanged.
`adi_corrected` is a distinct scientific workload; future CoVA comparisons must use the same corrected formula and golden.
Other repairs implement the original reference contract and can supplement their original benchmark's final comparison, provided source registration and repaired status remain explicit.

The [PolyBench/C ADI source](https://raw.githubusercontent.com/MatthiasJReisinger/PolyBenchC-4.2.1/master/stencils/adi/adi.c) gives the corrected coefficient.
An independent dense linear-system test checks both sweeps without reusing the optimized Thomas recurrence.
The histogram change addresses the shared-index write race described in [Numba's parallel-loop documentation](https://numba.readthedocs.io/en/stable/user/parallel.html#explicit-parallel-loops).
Chunk ownership does not assume a particular scheduler or map iteration indices to thread IDs.
The chunk count is the configured Numba worker budget captured as an integer, not the runtime `get_num_threads()` ctypes call, so it does not introduce process-local dynamic globals that prevent durable caching.
Correctness remains independent of how many workers execute those chunks.
The softmax test includes different row offsets and large logits, checking probabilities against the original NumPy reference.

Run focused CPU checks with `python -m unittest discover -s tests -p test_workload_repairs.py -v`.
For GPU and lifecycle coverage use `scripts/workload-repair-cases.tsv` with `NPBENCH_CASES_FILE` and `run_large.sh baselines OUTPUT` after activating the prepared environment.
Prepare new goldens explicitly (`NPBENCH_REQUIRE_GOLDEN=0`) once; subsequent collections should require them.
Use a new output directory after any source change.
Small unit-test thread limits are not measurement resource policy; performance collections use the system policy.

`golden_reference` points directly to a canonical benchmark and requires matching arguments, initialization contract and preset parameters.
Its initializer and NumPy source own golden identity and generation; a required cache hit never executes either initializer.
Reference chains or mismatched contracts are rejected, and unchanged benchmarks keep their existing cache identities.
