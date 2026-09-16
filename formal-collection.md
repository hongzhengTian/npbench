# Reproducible L collections

Use this procedure after the functional fixes and resource preflight have been reviewed.
Historical two-thread L runs and the targeted S diagnostics are functional evidence, not the full-resource performance baseline.
The numerical contract remains strict_region_v1: return count, shape, dtype, finite values, declared outputs and all array arguments are checked.
Output-contract failures are reported separately from numerical-value failures, but neither silently becomes a pass.
Original benchmark workload files are preserved; any future repaired workload needs a separate file and label.

## Environment and resource policy

Activate the prepared environment and work from the NPBench checkout root.
The optional native CPU/CUDA dependency profiles include threadpoolctl for observing loaded BLAS/OpenMP libraries.
No dependency is installed by the collection script.

The default NPBENCH_RESOURCE_POLICY=system removes inherited thread-count and affinity-control environment overrides introduced by this integration.
It does not call taskset, choose a NUMA policy, or force all libraries to use the machine's logical CPU count.
Native serial/parallel implementations and runtime policies remain intact, including any internal thread choices or oversubscription.
Scheduler affinity is preserved and checked: by default all online CPUs are expected; use NPBENCH_EXPECT_CPUS explicitly for a smaller legitimate allocation.
Known cgroup CPU quotas below the declared allocation are rejected.
The fixed policy exists for diagnostics and resource sensitivity, using explicit NPBENCH_THREADS and NPBENCH_CPUSET.

One selected GPU is used; choose CUDA_VISIBLE_DEVICES before collecting.
CUDA identity, driver/runtime versions, topology, NUMA observations and effective resource controls are recorded.
Worker launch affinity and individual thread affinities are distinguished from the calling thread's post-call affinity.
Thread-pool sizes are configuration observations, not active utilization measurements.
NPBENCH_RESOURCE_PROBE=1 adds per-call thread CPU accounting for diagnostics; it is disabled in the main matrix and enabled only for the separate resource supplement.

GPU process queries must succeed for performance collections.
When the installed NVML library mismatches the loaded kernel driver, NPBENCH_NVML_LIBRARY_DIR may name a directory containing a verified matching libnvidia-ml.so.1.
Only monitoring subprocesses receive this library path; numerical workers keep their original CUDA/driver library environment.
The library hash is frozen with the collection.
Do not install a driver, reboot, or load a different CUDA computation library as an implicit benchmark setup step.

Preflight rejects observed competing GPU compute processes or more than 10% aggregate activity on the offered CPUs in its brief sample.
Startup/end checks cannot prove exclusive allocation throughout a long run: use idle allocated resources and review any external interruptions.
The checkout lock only prevents competing collections from the same checkout.
Samples remain observed results requiring resource and variability review, not automatically certified paper data.

## Plan, main collection and complete campaign

Review the full main matrix and resource checks without running kernels:

```bash
./run_large.sh --plan all .cache/final-l/main
```

The plan freezes code, dependencies, resources and selection; changing these requires a new directory.
Preflight observations are stored separately from the stable resume contract.
Repeated search-path entries are normalized without changing their order for identity checks; raw environments remain in environment-observations.jsonl.
Existing L goldens are required by default, and every worker verifies and reuses them.
A missing golden blocks its cell; prepare it separately with run_benchmark.py --prepare-golden before the performance campaign.
Setting NPBENCH_REQUIRE_GOLDEN=0 explicitly permits the runner's separate reference preparation step for development use.
The --plan operation does not certify golden payload integrity or execute any implementation; actual golden preparation/hit receipts remain required.

To run the full agreed campaign, including separate initialization/resource supplements:

```bash
mkdir -p .cache/final-l
nohup ./run_large.sh --campaign .cache/final-l \
  > .cache/final-l.log 2>&1 &
```

The campaign fixes L, system resource policy, golden hits and 3 processes with 3 total calls each for the main matrix.
It clears inherited subset/retry and per-call diagnostic-probe selectors.
The main matrix has 908 implementation cells in the current catalog, including 7 missing original sources; the frozen plan is authoritative if the catalog changes.
NumPy, Numba, DaCe CPU/GPU and CuPy baselines run before the six CoVA routes.
Unselected NPBench frameworks, multi-node execution and multi-GPU experiments are outside this matrix.
CoVA failures are collected without requiring changes to CoVA during this baseline freeze.

For each fully successful main cell there are 1 initialization, 2 fresh_process and 6 same_process samples: 9 timed calls, not 9 repeats after initialization.
The current lifecycle CLI expresses this as --fresh-process-runs 2 -r 2.
The host-to-host timing boundary remains metric_version=3; resource/error/provenance additions use protocol_version=5.
See [Lifecycle semantics](lifecycle.md) for transfers, synchronization, input reset and artifact behavior.

The campaign then runs these checked-in, finite supplementary selections:

| Phase | Selection | Sampling |
| --- | --- | --- |
| initialization | scripts/initialization-cases.tsv: GEMM for 5 baseline routes and 6 CoVA routes | 3 independent empty-artifact trials per implementation, 1 call per trial |
| resources | scripts/resource-cases.tsv: 8 dense, graph, sparse and layout-sensitive representatives | 1 thread, one socket's physical cores, all allocated physical cores; 3 processes x 3 calls |

On the current 2-socket/48-physical-core machine the resource profiles are 1, 24 and 48 threads, without applying a NUMA memory policy.
CPU IDs are obtained from topology and the actual allocation, not hard-coded socket ranges.
These supplements are separate experimental roles and do not replace the unrestricted main results.
The finite choices are not an exhaustive tuning search or a guarantee of each system's performance ceiling.
Initialization conclusions for other workloads remain descriptive unless additional independent cold trials are collected.
The campaign has at most 8,421 timed calls for the current selections; missing sources, failures and unsupported reuse reduce this number.
Golden checks, imports, compilation, validation, audits and process management add wall time; no overnight completion time is promised.
The default 1,800-second timeout covers a complete worker; configure NPBENCH_TIMEOUT and NPBENCH_GOLDEN_TIMEOUT explicitly before planning if required.

The same runner supports separate phases without a second measurement harness:

```bash
./run_large.sh all .cache/main-only
./run_large.sh --cold scripts/initialization-cases.tsv .cache/cold-only
./run_large.sh --resources scripts/resource-cases.tsv .cache/resources-only
./run_large.sh cova .cache/cova-next
```

The cova-only mode requires compatible existing goldens and uses a new result/artifact directory.
Resource policy and metric must match the baseline when interpreting comparisons.
Independent cold trials use --lifecycle -r 0 and no later processes; zero repeats remain invalid for the unchanged upstream runner.

Resume with the same command and directory only when sources/environment/configuration are unchanged.
Finished failures and partial results are preserved, not rerun on resume.
Each cell points to its selected lifecycle attempt; interrupted older attempts stay on disk and are not pooled with the selected one.
The script exits nonzero for recorded failures/partial results while continuing independent cells.
A campaign stops before supplements if main preflight or identity checks prevent normal completion of the main collection.
Do not edit NPBench, CoVA or the environment during collection.

## Analysis and portable evidence

Analysis runs automatically after each completed collection and the complete campaign.
It uses only the Python standard library and reads saved observations without invoking kernels or reference producers.

```bash
./run_large.sh --analyze .cache/final-l
./run_large.sh --analyze .cache/final-l/main .cache/cova-next --replace-cova
./run_large.sh --export .cache/final-l /path/to/new/evidence-directory
python /path/to/new/evidence-directory/analyze_collection.py \
  /path/to/new/evidence-directory --verify
python /path/to/new/evidence-directory/analyze_collection.py \
  /path/to/new/evidence-directory --output /path/to/separate/rebuilt-analysis
```

Duplicate cells are rejected unless --replace-cova explicitly selects later whole CoVA cells in the same comparison group with the same golden identity and payload hash.
Different presets and resource/metric/environment groups are kept separate, never pooled merely because benchmark names match.
The exporter includes portable compressed observations, one compressed text/source/log archive per collection, available golden metadata and the analyzer with SHA-256 checksums.
Large arrays and native binaries remain in external storage; the original paths/hashes in manifests locate them.
Compressed evidence files are limited to 50 MiB each, with oversized raw text exclusions recorded explicitly.
Frozen evidence verification is read-only; rebuilding analysis requires an output outside the archive.

The analysis directory contains:

| File | Meaning |
| --- | --- |
| summary.json | Source cell-attempt counts and separate counts after explicit CoVA replacement |
| lifecycles.csv | Eligibility, median of process medians, process distributions, resource review flags and observed wall-time components |
| failures.json | Contract/numerical/stage categories and original terminal records; stage alone is not proof of upstream responsibility |
| best-observed.csv | Fastest eligible observed baseline and post-hoc best CoVA, grouped by device, lifecycle, golden and compatible environment |
| initialization-trials.json | Independent cold samples and whether the minimum three successful trials were obtained |
| replacements.json | Explicit whole-CoVA-cell replacement audit for later comparisons |

Missing or adverse resource evidence is excluded from best-observed comparisons.
Numerical failures exclude all three phases from ranking; persistence-only failure and memory-only policies preserve explicitly qualified process-0 observations.
Incomplete attempts, absent validation, wrong identity and malformed pass records do not become valid timings.
A single initialization sample is descriptive; the two main fresh-process observations cannot establish a reliable tail distribution.
Close differences and process variability require more independent observations before claiming a small advantage.
The 15% point-estimate flag means CoVA time <= 1.15 times the fastest qualified observed baseline, not a statistical significance claim or an automatic-selection result.
No single geometric-mean speedup is generated across incompatible configurations or silently changing benchmark coverage.
Report paired sets and coverage explicitly when preparing paper aggregates.

Before publication, repeat key baseline controls contemporaneously with the final CoVA revision and review workload/device equivalence, resource interference and variability.
A frozen baseline is reusable under its conditions; it is not permanently valid after changing the environment, inputs or measurement contract.
