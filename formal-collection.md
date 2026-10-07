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
CPU workers keep the scheduler's affinity; GPU workers use the NUMA rule below.
CPU routes retain their memory policy; no library thread count is forced.
Native serial/parallel implementations and runtime policies remain intact, including any internal thread choices or oversubscription.
Scheduler affinity is preserved and checked: by default all online CPUs are expected; use NPBENCH_EXPECT_CPUS explicitly for a smaller legitimate allocation.
Known cgroup CPU quotas below the declared allocation are rejected.
The fixed policy exists for diagnostics and resource sensitivity, using explicit NPBENCH_THREADS and NPBENCH_CPUSET.

One selected GPU is used; choose CUDA_VISIBLE_DEVICES before collecting.
`NPBENCH_GPU_NUMA_BINDING=cpu-memory` is the default; `cpu` binds CPUs only, and `off` retains inherited CPU and memory policy.
It binds only CuPy, DaCe GPU, CoVA LLVM GPU and CoVA OpenMP GPU lifecycle workers.
NumPy, Numba, DaCe CPU and the four CoVA CPU routes remain unbound.
CUDA logical device zero resolves the visible ordinal/UUID mask and CUDA enumeration order to its PCI address.
The resource owner reads that PCI device's sysfs `numa_node` and `local_cpulist`.
A worker receives the intersection of the local CPUs and the controller's allocation through a standard-library-only launcher, before the worker package and numerical libraries initialize.
The launcher sets CPU affinity, calls Linux `set_mempolicy(MPOL_BIND)` for the GPU node, verifies both, and execs the original worker.
The worker verifies that the memory policy survived exec.
No numactl installation or libnuma dependency is needed.
An unknown node (-1), a single-node system, unreadable/invalid topology or an empty CPU intersection retains inherited placement and records the reason.
A failed policy setup or verification restores the original CPU and memory policy and records the failure; an unsuccessful rollback stops the worker.
MPOL_BIND limits new anonymous allocations to the selected node; shared file-cache pages can still reside elsewhere.
Each sample records node-page summaries from /proc/self/numa_maps before and after the call, outside its timer.
Anonymous VMAs and file-backed VMAs are reported separately.
The frozen collection and comparison contract include the option and per-route resolved plans; `--plan` prints these plans without starting workers.
Each lifecycle manifest records the controller's affinity and configured pools, and each worker sample records its launch affinity, per-thread masks, configured pools and calling CPU/node before and after the timed call.
The system preflight still checks the controller's full declared allocation and CPU quota.
At untimed call boundaries, the worker also clamps runtime-created threads that widened their masks and records their names and before/after masks.
This is userspace affinity, not a cpuset cgroup: a runtime can temporarily widen a new helper during its cold call.
The first DaCe GPU call exhibited this behavior; subsequent boundaries restored the node mask without moving initialization outside the timer.
Worker analysis compares launch masks against the frozen per-route CPU set, and rejects threads whose observed masks extend outside a bound GPU node.
Do not set `NPBENCH_EXPECT_CPUS=48` to bypass the controller's 96-CPU preflight on hsc-12.
GPU workers there have 48 allowed logical CPUs (24–47 and 72–95), whereas CPU workers retain 96.
Native pool defaults may change with that smaller mask: the small GEMM validation observed OpenBLAS 64 threads with binding off and 48 with binding on.
These are configured pool sizes, not observed utilization; internal library budgets remain intact.
GPU binding therefore also changes the resource offer to mixed GPU programs' host work, and GPU baselines must be recollected under the same rule.

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
The host-to-host timing boundary remains metric_version=3; resource/error/provenance and Numba reuse verification use protocol_version=6.
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
Text build configuration such as DaCe .conf, CMake .make and generated .ptx is retained; compiled binaries and numerical array payloads are not exported.
Large arrays and native binaries remain in external storage; the original paths/hashes in manifests locate them.
Compressed evidence files are limited to 50 MiB each, with oversized raw text exclusions recorded explicitly.
Frozen evidence verification is read-only; rebuilding analysis requires an output outside the archive.

The analysis directory contains:

| File | Meaning |
| --- | --- |
| summary.json | Source cell-attempt counts and separate counts after explicit CoVA replacement |
| lifecycles.csv | Eligibility, median of process medians, process distributions, resource review flags and observed wall-time components |
| failures.json | Contract/numerical/stage categories and original terminal records; stage alone is not proof of upstream responsibility |
| best-observed.csv | Fastest eligible observed baseline and post-hoc best CoVA with each candidate's eligibility, sample/process counts, variability and confirmation limits |
| qualification-summary.json | Counts of observed best candidates requiring sampling or variability review, separately by device and lifecycle |
| initialization-trials.json | Independent cold samples and whether the minimum three successful trials were obtained |
| replacements.json | Explicit whole-CoVA-cell replacement audit for later comparisons |

Missing or adverse resource evidence is excluded from best-observed comparisons.
Numerical failures exclude all three phases from ranking; persistence-only failure and memory-only policies preserve explicitly qualified process-0 observations.
Incomplete attempts, absent validation, wrong identity and malformed pass records do not become valid timings.
Between-process max/min is absent for a single process; a value of 1 is not invented to imply stability.
Process-0-only results remain explicitly qualified observations, not evidence of persistent reuse or independent-process stability.
A ratio above 1.2 is a review flag, not a confidence interval or significance test.
Resource evidence limits distinguish missing interval-wide isolation and unverified NUMA policy from adverse resource observations that exclude ranking.
No flag does not establish stability: confirmation_status remains pending_independent_review, and within_15_percent_confirmed is unset.
A single initialization sample is descriptive; the two main fresh-process observations cannot establish a reliable tail distribution.
Close differences and process variability require more independent observations before claiming a small advantage.
The 15% point-estimate flag means CoVA time <= 1.15 times the fastest qualified observed baseline, not a statistical significance claim or an automatic-selection result.
No single geometric-mean speedup is generated across incompatible configurations or silently changing benchmark coverage.
Report paired sets and coverage explicitly when preparing paper aggregates.

Before publication, repeat key baseline controls contemporaneously with the final CoVA revision and review workload/device equivalence, resource interference and variability.
Protocol 6 retains metric_version=3 and strict_region_v1, but adds explicit Numba dispatcher reuse qualification.
Older baseline records without these observations retain their original status and are labelled not_recorded; artifact integrity alone is not proof of a disk-cache hit.
Collection identity still separates protocol and measurement-source changes: a new directory is required, and protocol-5/6 results must not be silently pooled.
Any later cross-protocol baseline reuse needs an explicit compatibility review; this observational change alone does not require rerunning all historical baselines.
A frozen baseline is reusable under its conditions; it is not permanently valid after changing the environment, inputs or measurement contract.

## User-run stage A closing collection

Only the user runs this full collection after reviewing the harness change.
The command below follows the collected CoVA-only run's activation, selection and monitoring procedure.
It clears residual COVA_* and subset controls, leaves NPBENCH_COVA_ACTIVATE unset, and uses run_cova_large.sh's default activation file.
That file activates covadev and the original native environment, including OMP_TARGET_OFFLOAD=MANDATORY and CUDAHOSTCXX.
Do not replace it with an empty activation override or a partial manual environment.
Choose NPBENCH_GPU_NUMA_BINDING=off, cpu or cpu-memory before invoking the command; cpu-memory is the default.
The command freezes the plan first, then launches the same CoVA-only matrix in the background with GPU process and utilization records every ten seconds.
It does not include baseline controls or cold/resource supplements.
Outputs use a new .cache/large-runs/<UTC timestamp>-cova-L-<CoVA short revision>-<binding> directory and sibling logs.
Existing goldens must be present; a missing bundle blocks its cell rather than generating a replacement.
Save the following as a shell script and run it with bash.

```bash
#!/usr/bin/env bash
# CoVA-only collection through run_cova_large.sh's default activation file.
set -e
cd /HSC/users/tianhong/myHome/developing/npbench
unset ${!COVA_*} NPBENCH_BENCHMARKS NPBENCH_CASES_FILE NPBENCH_RETRY_FROM NPBENCH_FRAMEWORKS NPBENCH_RESOURCE_PROBE
# Unset overrides so the wrapper activates covadev and its original native environment.
unset NPBENCH_COVA_ACTIVATE NPBENCH_PYTHON
binding="${NPBENCH_GPU_NUMA_BINDING:-cpu-memory}"
case "$binding" in off|cpu|cpu-memory) ;; *) echo 'Expected off|cpu|cpu-memory' >&2; exit 2;; esac
export NPBENCH_GPU_NUMA_BINDING="$binding" CUDA_VISIBLE_DEVICES=0
version="$(git -C ../CoVA rev-parse --short=8 HEAD)"
mkdir -p .cache/large-runs
out="$(pwd)/.cache/large-runs/$(date -u +%Y%m%dT%H%M%SZ)-cova-L-$version-$binding"
# Freeze first. If configuration/resource checks fail, do not launch the collection.
./run_cova_large.sh --plan "$out" > "$out.plan.log" 2>&1
nohup bash -c '
    out=$1
    while true; do
        echo "== $(date -u +%FT%TZ)"
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
        nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
        sleep 10
    done > "$out.gpu-occupancy.log" 2>&1 &
    monitor=$!
    echo "$monitor" > "$out.monitor.pid"
    trap "kill $monitor 2>/dev/null || true; wait $monitor 2>/dev/null || true" EXIT
    trap "exit 130" INT
    trap "exit 143" TERM
    echo "started $(date -u +%FT%TZ)"
    status=0
    ./run_cova_large.sh "$out" > "$out.log" 2>&1 || status=$?
    echo "finished $(date -u +%FT%TZ) exit=$status" | tee "$out.exit.txt"
    exit "$status"
' _ "$out" > "$out.launch.log" 2>&1 &
echo "$!" > "$out.launch.pid"
echo "Background collection: $out (PID $!)"
```

The launch PID, monitoring PID, start/finish log and final exit status are recorded beside the result directory.
Do not edit code or change environments between plan freezing and collection, or before resuming that directory.
A nonzero collection exit can represent recorded workload failures; inspect the exit file and individual cells.
Monitoring errors or gaps leave occupancy unverified; identify external PIDs using worker records and review affected cells.

### Optional baseline controls, run separately

Before/after controls are optional separate collections, not part of the CoVA-only command.
Run the block below separately before or after the main collection only when requested, setting NPBENCH_CONTROL_POSITION accordingly.
It requires the same original activation environment; run_large.sh does not source run_cova_large.sh's activation file for it.
Explicitly source that file to activate covadev, CUDAHOSTCXX and OMP_TARGET_OFFLOAD=MANDATORY, and use the same GPU binding choice as the main collection.
Each invocation selects eight implementation cells; it does not run the full baseline matrix.
Keep separate GPU occupancy records for any controls used in a performance comparison and review their hot samples and variability.

```bash
cd /HSC/users/tianhong/myHome/developing/npbench
unset ${!COVA_*} NPBENCH_BENCHMARKS NPBENCH_CASES_FILE NPBENCH_RETRY_FROM NPBENCH_FRAMEWORKS NPBENCH_RESOURCE_PROBE
unset NPBENCH_COVA_ACTIVATE NPBENCH_PYTHON
export NPBENCH_RESOURCE_POLICY=system
source .cache/native-validation/cova-integration-20260912/activate-cova.sh
export NPBENCH_PYTHON="$(command -v python)"
export NPBENCH_NVML_LIBRARY_DIR="${NPBENCH_COVA_MONITOR_LIBRARY_DIR:-}"
export CUDA_VISIBLE_DEVICES=0 NPBENCH_GPU_NUMA_BINDING="${NPBENCH_GPU_NUMA_BINDING:-cpu-memory}"
export NPBENCH_PRESET=L NPBENCH_REQUIRE_GOLDEN=1 NPBENCH_FRESH_PROCESSES=2 NPBENCH_REPEATS=2
export NPBENCH_MEASUREMENT_ROLE=main
position="${NPBENCH_CONTROL_POSITION:-before}"
case "$position" in before|after) ;; *) echo 'Expected before|after' >&2; exit 2;; esac
controls="$(pwd)/.cache/large-runs/$(date -u +%Y%m%dT%H%M%SZ)-controls-$position-$NPBENCH_GPU_NUMA_BINDING"
mkdir -p "$controls"
cat > "$controls/controls.tsv" <<'CASES'
gemm	numba	nopython-mode
gemm	dace_cpu	auto_opt
gemm	dace_gpu	auto_opt
gemm	cupy	default
gesummv	numba	nopython-mode
gesummv	dace_cpu	auto_opt
gesummv	dace_gpu	auto_opt
gesummv	cupy	default
CASES
NPBENCH_CASES_FILE="$controls/controls.tsv" NPBENCH_BENCHMARKS='gemm gesummv' \
  NPBENCH_FRAMEWORKS='numba dace_cpu dace_gpu cupy' \
  bash run_large.sh baselines "$controls/results" > "$controls.log" 2>&1
```
