#!/usr/bin/env bash
# Sequential, resumable collection through NPBench's existing lifecycle CLI.
set -euo pipefail

if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
    cat <<'HELP'
Usage: ./run_large.sh [--plan] [all|baselines|cova] [RESULT_DIRECTORY]
       ./run_large.sh --campaign RESULT_DIRECTORY
       ./run_large.sh --cold CASES.tsv RESULT_DIRECTORY
       ./run_large.sh --resources CASES.tsv RESULT_DIRECTORY
       ./run_large.sh --analyze RESULT_DIRECTORY [OTHER_COLLECTION ...]
       ./run_large.sh --export RESULT_DIRECTORY EVIDENCE_DIRECTORY
Activate the prepared CoVA/NPBench environment before running this script.
Defaults: all registered benchmarks, L, 3 processes x 3 calls (9 total),
all allocated CPU resources with native runtime defaults, visible GPU 0,
1800s per complete worker/golden. Existing goldens required.
--plan freezes and checks configuration without executing benchmarks.
--cold makes 3 independent empty-artifact collections, one call per cell.
--resources compares fixed 1-thread, one-socket and all-physical-core budgets.
These supplementary modes use exact TSV selections and separate directories.
--campaign runs full L/3x3 main, the checked-in cold and resource selections,
then combines their separate analyses. It never runs an unrestricted search.
Baselines (NumPy, Numba, DaCe CPU/GPU, CuPy) run before CoVA in all mode.
Completed steps, including failures, are kept when resuming the same directory.
Use a new directory for retries, changed code/configuration, or later CoVA runs.

Optional environment:
  NPBENCH_GOLDEN_CACHE       persistent golden directory (default .cache/goldens)
  NPBENCH_RESOURCE_POLICY   system (default) or fixed (diagnostics only)
  NPBENCH_THREADS           CPU thread budget for fixed mode (default 2)
  NPBENCH_CPUSET            taskset CPU list for fixed mode only
  NPBENCH_EXPECT_CPUS       expected allocated logical CPU count (system: all online by default)
  NPBENCH_BENCHMARKS        space-separated subset (default every bench_info entry)
  NPBENCH_FRAMEWORKS        space-separated subset of the selected mode's frameworks
  NPBENCH_PRESET            S/M/L/paper (default L)
  NPBENCH_FRESH_PROCESSES   later processes per implementation (default 2)
  NPBENCH_REPEATS           subsequent calls within every process (default 2)
  NPBENCH_TIMEOUT          seconds per complete worker (default 1800)
  NPBENCH_GOLDEN_TIMEOUT   seconds per reference preparation (default 1800)
  NPBENCH_PYTHON           Python executable (default python)
  NPBENCH_NVML_LIBRARY_DIR optional matching NVML directory, used only by monitoring
  NPBENCH_REQUIRE_GOLDEN   1 (default): require existing goldens; 0: explicit preparation
  NPBENCH_CASES_FILE       exact TSV: benchmark/framework/implementation
  NPBENCH_RETRY_FROM       restrict baselines to affected failures in an old collection

cova mode requires existing goldens and never runs their producers.
No dependencies are installed. The script does not promise overnight completion
or publication-ready results; inspect failures, device placement and variability.
HELP
    exit 0
fi
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python="${NPBENCH_PYTHON:-python}"
if [[ "${1:-}" == --campaign ]]; then
    [[ $# == 2 ]] || { echo 'Expected campaign output directory' >&2; exit 2; }
    campaign_out="$(realpath -m -- "$2")"
    export NPBENCH_RESOURCE_POLICY=system NPBENCH_MEASUREMENT_ROLE=main NPBENCH_PRESET=L
    export NPBENCH_FRESH_PROCESSES=2 NPBENCH_REPEATS=2 NPBENCH_REQUIRE_GOLDEN=1
    unset NPBENCH_CASES_FILE NPBENCH_RETRY_FROM NPBENCH_BENCHMARKS NPBENCH_FRAMEWORKS NPBENCH_RESOURCE_PROBE
    campaign_failed=0
    main_rc=0
    bash "$repo/run_large.sh" all "$campaign_out/main" || main_rc=$?
    # Exit 1 means recorded workload failures; configuration/resource failures
    # use 3 and cannot be hidden by an old summary when resuming.
    [[ "$main_rc" -le 1 ]] || exit "$main_rc"
    campaign_failed="$main_rc"
    phase_rc=0
    bash "$repo/run_large.sh" --cold "$repo/scripts/initialization-cases.tsv" "$campaign_out/initialization" || phase_rc=$?
    [[ "$phase_rc" -le 1 ]] || exit "$phase_rc"
    [[ "$phase_rc" == 0 ]] || campaign_failed=1
    phase_rc=0
    bash "$repo/run_large.sh" --resources "$repo/scripts/resource-cases.tsv" "$campaign_out/resources" || phase_rc=$?
    [[ "$phase_rc" -le 1 ]] || exit "$phase_rc"
    [[ "$phase_rc" == 0 ]] || campaign_failed=1
    "$python" "$repo/scripts/analyze_collection.py" "$campaign_out" || campaign_failed=1
    exit "$campaign_failed"
elif [[ "${1:-}" == --analyze ]]; then
    shift; exec "$python" "$repo/scripts/analyze_collection.py" "$@"
elif [[ "${1:-}" == --export ]]; then
    [[ $# == 3 ]] || { echo 'Expected collection and evidence directory' >&2; exit 2; }
    exec "$python" "$repo/scripts/analyze_collection.py" "$2" --export "$3"
elif [[ "${1:-}" == --cold || "${1:-}" == --resources ]]; then
    [[ $# == 3 ]] || { echo 'Expected exact cases file and output directory' >&2; exit 2; }
    supplement="$1"; NPBENCH_CASES_FILE="$(realpath -e -- "$2")"
    export NPBENCH_CASES_FILE
    supplement_out="$(realpath -m -- "$3")"
    # One owner for measurement: invoke this runner with a frozen role/profile.
    supplement_failed=0
    if [[ "$supplement" == --cold ]]; then
        export NPBENCH_MEASUREMENT_ROLE=cold NPBENCH_REPEATS=0 NPBENCH_FRESH_PROCESSES=0
        for trial in 1 2 3; do
            phase_rc=0
            bash "$repo/run_large.sh" all "$supplement_out/trial-$trial" || phase_rc=$?
            [[ "$phase_rc" -le 1 ]] || exit "$phase_rc"
            [[ "$phase_rc" == 0 ]] || supplement_failed=1
        done
    else
        export NPBENCH_MEASUREMENT_ROLE=resource NPBENCH_RESOURCE_POLICY=fixed
        export NPBENCH_REPEATS=2 NPBENCH_FRESH_PROCESSES=2 NPBENCH_RESOURCE_PROBE=1
        # Enumerate physical cores within the actual allocation, not CPU IDs
        # assumed to match sockets or SMT siblings.
        profiles="$("$python" -c 'import os,subprocess; rows=[r.split(",") for r in subprocess.check_output(["lscpu","-p=CPU,CORE,SOCKET"],text=True).splitlines() if not r.startswith("#")]; cores={}; allowed=os.sched_getaffinity(0)
for cpu,core,socket in rows:
 if int(cpu) in allowed: cores.setdefault((socket,core),int(cpu))
first=next(iter(cores)); socket=first[0]; profiles={"one": [cores[first]], "socket": [v for k,v in cores.items() if k[0]==socket], "physical": list(cores.values())}
for name,cpus in profiles.items(): print(name, len(cpus), ",".join(map(str,sorted(cpus))),sep="\t")')"
        while IFS=$'\t' read -r profile NPBENCH_THREADS NPBENCH_CPUSET; do
            export NPBENCH_THREADS NPBENCH_CPUSET
            phase_rc=0
            bash "$repo/run_large.sh" all "$supplement_out/$profile" || phase_rc=$?
            [[ "$phase_rc" -le 1 ]] || exit "$phase_rc"
            [[ "$phase_rc" == 0 ]] || supplement_failed=1
        done <<< "$profiles"
    fi
    "$python" "$repo/scripts/analyze_collection.py" "$supplement_out" || supplement_failed=1
    exit "$supplement_failed"
fi
plan_only=0
if [[ "${1:-}" == --plan ]]; then plan_only=1; shift; fi
[[ $# -le 2 ]] || { echo "Too many arguments; see --help" >&2; exit 2; }
mode="${1:-all}"
case "$mode" in all|baselines|cova) ;; *) echo 'Expected all, baselines, or cova' >&2; exit 2;; esac
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python="${NPBENCH_PYTHON:-python}"
export NPBENCH_RESOURCE_POLICY="${NPBENCH_RESOURCE_POLICY:-system}"
export NPBENCH_MEASUREMENT_ROLE="${NPBENCH_MEASUREMENT_ROLE:-main}"
export NPBENCH_REQUIRE_GOLDEN="${NPBENCH_REQUIRE_GOLDEN:-1}"
export NPBENCH_PRESET="${NPBENCH_PRESET:-L}"
export NPBENCH_FRESH_PROCESSES="${NPBENCH_FRESH_PROCESSES:-2}"
export NPBENCH_REPEATS="${NPBENCH_REPEATS:-2}"
export NPBENCH_TIMEOUT="${NPBENCH_TIMEOUT:-1800}"
export NPBENCH_GOLDEN_TIMEOUT="${NPBENCH_GOLDEN_TIMEOUT:-1800}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONDONTWRITEBYTECODE=1
# shellcheck source=scripts/native-env.sh
source "$repo/scripts/native-env.sh"
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
NPBENCH_GOLDEN_CACHE="$(realpath -m -- "${NPBENCH_GOLDEN_CACHE:-$repo/.cache/goldens}")"
export NPBENCH_GOLDEN_CACHE
runner=("$python")
if [[ "$NPBENCH_RESOURCE_POLICY" == fixed ]]; then
    NPBENCH_CPUSET="${NPBENCH_CPUSET:-$("$python" -c 'import os; n=int(os.environ["NPBENCH_THREADS"]); a=sorted(os.sched_getaffinity(0)); assert n <= len(a), "Insufficient allowed CPUs"; print(",".join(map(str,a[:n])))')}"
    export NPBENCH_CPUSET
    runner=(taskset --cpu-list "$NPBENCH_CPUSET" "$python")
fi
out="$(realpath -m -- "${2:-$repo/.cache/large-runs/$(date -u +%Y%m%dT%H%M%SZ)-$mode}")"
mkdir -p "$repo/.cache" "$out"
# One script at a time per checkout, even when result directories differ.
exec 9>"$repo/.cache/large-run.lock"
flock -n 9 || { echo 'Another run_large.sh is active in this checkout' >&2; exit 3; }

# Freeze the plan and the environment used for resume checks. Existing result
# directories are immutable with respect to code, resources and configuration.
freeze_collection() {
"${runner[@]}" - "$repo" "$out" "$mode" <<'PY'
import hashlib, importlib.metadata, json, os, pathlib, shlex, subprocess, sys
from npbench.infrastructure import Benchmark, generate_framework
from npbench.infrastructure.lifecycle import METRIC_VERSION, PROTOCOL_VERSION, VALIDATION_CONTRACT
from npbench.infrastructure.resources import allocation_snapshot, gpu_identity, idle_observation, monitor_environment
repo, out = map(pathlib.Path, sys.argv[1:3]); mode = sys.argv[3]
def capture(command):
    try:
        return subprocess.check_output(command, stderr=subprocess.STDOUT, text=True, timeout=30,
                                       env=monitor_environment() if command[0] == 'nvidia-smi' else None).strip()
    except (OSError, subprocess.SubprocessError) as error:
        return str(error)
def revision(path):
    return capture(['git', '-C', str(path), 'rev-parse', 'HEAD'])
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()
for directory in (out, pathlib.Path(os.environ['NPBENCH_GOLDEN_CACHE'])):
    if directory.is_relative_to(repo):
        assert subprocess.run(['git', '-C', str(repo), 'check-ignore', '-q', str(directory)]).returncode == 0, 'Use an ignored directory or a path outside the checkout for results/goldens'
for key in ('NPBENCH_FRESH_PROCESSES', 'NPBENCH_REPEATS', 'NPBENCH_TIMEOUT', 'NPBENCH_GOLDEN_TIMEOUT'):
    value = int(os.environ[key])
    assert value >= (0 if key in ('NPBENCH_FRESH_PROCESSES', 'NPBENCH_REPEATS') else 1), key
assert os.environ['NPBENCH_PRESET'] in ('S', 'M', 'L', 'paper')
available = {p.stem for p in (repo/'bench_info').glob('*.json')}
benchmarks = os.environ.get('NPBENCH_BENCHMARKS', '').split() or sorted(available)
assert len(benchmarks) == len(set(benchmarks)) and set(benchmarks) <= available, 'Invalid benchmark selection'
baselines = ['numpy', 'numba', 'dace_cpu', 'dace_gpu', 'cupy']
cova = sorted(p.stem for p in (repo/'framework_info').glob('cova_*.json'))
allowed = baselines + cova if mode == 'all' else baselines if mode == 'baselines' else cova
selected = os.environ.get('NPBENCH_FRAMEWORKS', '').split() or allowed
assert len(selected) == len(set(selected)) and set(selected) <= set(allowed), 'Invalid framework selection'
frameworks = [name for name in allowed if name in selected]
exact_cases = None
if os.environ.get('NPBENCH_CASES_FILE'):
    assert not os.environ.get('NPBENCH_RETRY_FROM'), 'Exact cases cannot combine with retry selection'
    exact_cases = [tuple(line.split('\t')) for line in pathlib.Path(os.environ['NPBENCH_CASES_FILE']).read_text().splitlines() if line.strip()]
    assert exact_cases and len(exact_cases) == len(set(exact_cases)) and all(len(row) == 3 for row in exact_cases), 'Invalid or duplicate case rows'
    assert all(b in benchmarks and f in frameworks for b, f, label in exact_cases), 'Case selection is outside framework selection'
    benchmarks = [b for b in benchmarks if any(row[0] == b for row in exact_cases)]
    frameworks = [f for f in frameworks if any(row[1] == f for row in exact_cases)]
    exact_cases = sorted(exact_cases)
retry = None
if os.environ.get('NPBENCH_RETRY_FROM'):
    from scripts.retry_affected import select
    assert mode == 'baselines', 'Retry selection only supports baselines'
    retry = select(pathlib.Path(os.environ['NPBENCH_RETRY_FROM']))
    assert retry['cells'], 'No affected failures to retry'
    assert os.environ['NPBENCH_PRESET'] == retry['preset'], 'Retry preset differs from original'
    assert not out.is_relative_to(pathlib.Path(retry['source'])), 'Use a separate retry directory'
    retry_keys = {tuple(row[:3]) for row in retry['cells']}
    benchmarks = [b for b in benchmarks if any(key[0] == b for key in retry_keys)]
    frameworks = [f for f in frameworks if any(key[1] == f for key in retry_keys)]
    assert all(b in benchmarks and f in frameworks for b, f, _ in retry_keys), 'Subset overrides exclude retry cells'

if any(name in ('cupy', 'dace_gpu', 'cova_llvm_gpu', 'cova_openmp_gpu') for name in frameworks):
    import cupy
    assert cupy.cuda.runtime.getDeviceCount() > 0, 'GPU unavailable'
    gpu = gpu_identity()
    assert gpu.get('uuid'), 'CUDA GPU identity missing'
else:
    gpu = None
if any(name.startswith('cova_') for name in frameworks):
    import cova as package
    cova_root = pathlib.Path(os.environ.get('COVAPATH', pathlib.Path(package.__file__).resolve().parents[1]))
    os.environ['NPBENCH_COVA_VERSION'] = revision(cova_root)
else:
    cova_root = None
    os.environ['NPBENCH_COVA_VERSION'] = 'unselected'
versions = {name: generate_framework(name).version() for name in frameworks}
files = subprocess.check_output(['git', '-C', str(repo), 'ls-files', '-z', '--cached', '--others', '--exclude-standard']).decode().split('\0')
sources = {name: digest(repo/name) for name in sorted(set(files)) if name and (repo/name).is_file()}
environment = {k: v for k, v in os.environ.items() if k.startswith(
    ('NPBENCH_', 'COVA_', 'DACE_', 'CUDA', 'CUPY_', 'OMP_', 'OPENBLAS_', 'MKL_', 'NUMBA_', 'ROOT_NVHPC'))
    or k in ('PATH', 'PYTHONPATH', 'LD_LIBRARY_PATH', 'LIBRARY_PATH', 'CPATH', 'CMAKE_PREFIX_PATH',
             'COVAPATH', 'CC', 'CXX', 'FC', 'CMAKE_BUILD_PARALLEL_LEVEL')}
# Re-activation can repeat search entries without changing resolution.
# Preserve the raw environment separately; freeze its effective search order.
with (out/'environment-observations.jsonl').open('a') as stream:
    stream.write(json.dumps(environment)+'\n')
for key in ('PATH', 'PYTHONPATH', 'LD_LIBRARY_PATH', 'LIBRARY_PATH', 'CPATH', 'CMAKE_PREFIX_PATH'):
    if key in environment:
        environment[key] = os.pathsep.join(dict.fromkeys(environment[key].split(os.pathsep)))
contract = {'mode': mode, 'benchmarks': benchmarks, 'frameworks': frameworks,
            'measurement_role': os.environ.get('NPBENCH_MEASUREMENT_ROLE', 'main'),
            'validation_contract': VALIDATION_CONTRACT,
            'metric_version': METRIC_VERSION, 'protocol_version': PROTOCOL_VERSION, 'npbench_revision': revision(repo), 'sources': sources,
            'framework_versions': versions, 'python': sys.version, 'executable': sys.executable,
            'pip_freeze': capture([sys.executable, '-m', 'pip', 'freeze', '--all']),
            'environment': environment, 'host': os.uname().nodename,
            'affinity': sorted(os.sched_getaffinity(0)), 'gpu': gpu,
            'allocation': allocation_snapshot(),
            'nvcc': capture(['nvcc', '--version'])}
contract['toolchain'] = {name: capture(shlex.split(os.environ.get(variable, default)) + ['--version'])
    for name,variable,default in [('cc','CC','cc'), ('cxx','CXX','c++'), ('cuda_host','CUDAHOSTCXX','c++'), ('cmake','NPBENCH_CMAKE','cmake')]}
contract['toolchain']['nvcc'] = contract['nvcc']
if os.environ.get('NPBENCH_NATIVE_PREFIX'):
    prefix = pathlib.Path(os.environ['NPBENCH_NATIVE_PREFIX'])
    contract['native_libraries'] = {str(p.relative_to(prefix)): digest(p) for p in sorted(prefix.rglob('*.so*')) if p.is_file()}
else:
    contract['native_libraries'] = {}
if os.environ.get('NPBENCH_NVML_LIBRARY_DIR'):
    contract['monitoring_library_sha256'] = digest(pathlib.Path(os.environ['NPBENCH_NVML_LIBRARY_DIR'])/'libnvidia-ml.so.1')
if exact_cases is not None:
    contract['exact_cases'] = [list(row) for row in exact_cases]
if retry is not None:
    contract['retry_selection'] = retry
if cova_root:
    contract['cova_revision'] = revision(cova_root)
    contract['cova_diff'] = capture(['git', '-C', str(cova_root), 'diff', 'HEAD', '--binary'])
    extra = subprocess.check_output(['git', '-C', str(cova_root), 'ls-files', '--others', '--exclude-standard', '-z']).decode().split('\0')
    contract['cova_untracked'] = {name: digest(cova_root/name) for name in extra if name and (cova_root/name).is_file()}
    contract['cova_tools'] = {name: digest(cova_root/'build/bin'/name) for name in ('cova-opt', 'cova-translate')
                             if (cova_root/'build/bin'/name).is_file()}
if os.environ.get('NPBENCH_RESOURCE_POLICY', 'system') == 'system':
    expected = int(os.environ.get('NPBENCH_EXPECT_CPUS', os.cpu_count()))
    assert len(contract['affinity']) == expected, 'Unexpected CPU allocation: specify the actual allocated CPU count explicitly'
    for filename, value in contract['allocation']['cgroup_limits'].items():
        if filename.endswith('cpu.max'):
            quota, period = value.split()
            assert quota == 'max' or int(quota)/int(period) >= expected, 'CPU quota is below declared allocation'
# Session/cgroup path is provenance, not a performance resource identity.
allocation_raw = contract['allocation']
limits = allocation_raw['cgroup_limits']
contract['allocation'] = {k:v for k,v in allocation_raw.items() if k not in ('cgroup', 'cgroup_limits')}
contract['allocation']['limits'] = {name: sorted({value for path,value in limits.items() if path.endswith('/'+name)})
                                  for name in ('cpu.max', 'memory.max', 'cpuset.cpus.effective', 'cpuset.mems.effective')}
with (out/'allocation-observations.jsonl').open('a') as stream:
    stream.write(json.dumps(allocation_raw)+'\n')
contract['comparison_contract'] = {k: contract[k] for k in
    ('metric_version', 'protocol_version', 'validation_contract', 'host', 'affinity', 'gpu', 'allocation', 'pip_freeze', 'toolchain', 'native_libraries')}
# Selection, iteration counts and CoVA revision may change in later comparisons;
# adapters, machine/resources, dependency versions and common environment may not.
contract['comparison_contract']['measurement_sources'] = {k:v for k,v in sources.items() if k.startswith('npbench/infrastructure/')}
contract['comparison_contract']['validation_sources'] = {k:v for k,v in sources.items() if k.startswith('bench_info/')}
contract['comparison_contract']['baseline_sources'] = {k:v for k,v in sources.items() if k.startswith('npbench/benchmarks/') and '_cova_' not in k}
contract['comparison_contract']['environment'] = {k:v for k,v in environment.items() if k.startswith(
    ('OMP_', 'OPENBLAS_', 'MKL_', 'NUMBA_', 'DACE_', 'CUDA', 'CUPY_')) and 'CACHE' not in k}
contract['comparison_contract']['library_environment'] = {k:environment.get(k) for k in
    ('LD_LIBRARY_PATH', 'LIBRARY_PATH', 'CPATH', 'CMAKE_PREFIX_PATH', 'CC', 'CXX', 'CUDAHOSTCXX')}
contract['comparison_contract']['resource_policy'] = os.environ.get('NPBENCH_RESOURCE_POLICY', 'system')
path = out/'collection.json'
if path.exists():
    assert json.loads(path.read_text()) == contract, 'Code/environment/selection changed; use a new result directory'
else:
    path.write_text(json.dumps(contract, indent=2)+'\n')
    snapshot = out/'source-snapshots'
    for name in sources:
        target = snapshot/name; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((repo/name).read_bytes())
    for label, command in [('cpu', ['lscpu']), ('memory', ['free', '-h']), ('disk', ['df', '-h', str(out)]),
                           ('gpu', ['nvidia-smi']), ('numpy', [sys.executable, '-c', 'import numpy; numpy.show_config()'])]:
        (out/(label+'.txt')).write_text(capture(command)+'\n')
    if cova_root:
        with (out/'cova-source.tar').open('wb') as stream:
            subprocess.run(['git', '-C', str(cova_root), 'archive', 'HEAD'], stdout=stream, check=True)
        (out/'cova.patch').write_text(contract['cova_diff'])
        for name in contract['cova_untracked']:
            target = out/'cova-untracked'/name; target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((cova_root/name).read_bytes())
plan = []
prepared = set()
for phase in ('baselines', 'cova'):
    phase_frameworks = [name for name in frameworks if name.startswith('cova_') == (phase == 'cova')]
    if not phase_frameworks: continue
    for name in benchmarks:
        if name not in prepared:
            plan.append(('golden', name, 'numpy', 'reference', 'present')); prepared.add(name)
        bench = Benchmark(name)
        for framework_name in phase_frameworks:
            framework = generate_framework(framework_name)
            labels = framework.implementation_names(bench)
            for label in labels:
                if exact_cases is not None and (name, framework_name, label) not in exact_cases: continue
                if retry is not None and (name, framework_name, label) not in retry_keys: continue
                files = framework.impl_files(bench)
                exists = any(path.is_file() and (file_label == label or framework_name.startswith('dace_')) for path, file_label in files)
                plan.append((phase, name, framework_name, label, 'present' if exists else 'missing_source'))
if exact_cases is not None:
    assert {(b, f, label) for phase, b, f, label, _ in plan if phase != 'golden'} == set(exact_cases), 'Exact implementation unavailable'
if retry is not None:
    assert {(b, f, label) for phase, b, f, label, _ in plan if phase == 'baselines'} == retry_keys, 'Retry implementation unavailable'
(out/'plan.tsv').write_text(''.join('\t'.join(row)+'\n' for row in plan))
(out/'cova-version.txt').write_text(os.environ.get('NPBENCH_COVA_VERSION', 'unselected')+'\n')
cells = sum(row[0] != 'golden' for row in plan)
fresh, repeat = (int(os.environ[k]) for k in ('NPBENCH_FRESH_PROCESSES', 'NPBENCH_REPEATS'))
summary = {'benchmark_count': len(benchmarks), 'implementation_cells': cells,
           'missing_sources': sum(row[-1] == 'missing_source' for row in plan),
           'calls_per_complete_cell': (fresh+1)*(repeat+1), 'initialization': 1,
           'fresh_process': fresh, 'same_process': (fresh+1)*repeat,
           'resource_policy': os.environ.get('NPBENCH_RESOURCE_POLICY', 'system'),
           'cpu_affinity': contract['affinity'], 'role': contract['measurement_role'],
           'golden_required': os.environ.get('NPBENCH_REQUIRE_GOLDEN', '1') == '1',
           'worker_timeout_seconds': int(os.environ['NPBENCH_TIMEOUT'])}
(out/'plan-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
print('Collection:', out, '; planned steps:', len(plan), flush=True)
print(json.dumps(summary), flush=True)
PY
}
freeze_collection || exit 3
NPBENCH_COVA_VERSION="$(cat "$out/cova-version.txt")"
export NPBENCH_COVA_VERSION
check_resources() {
"${runner[@]}" - "$out" <<'CHECK'
import json, os, sys, time
from pathlib import Path
from npbench.infrastructure.resources import idle_observation
out=Path(sys.argv[1]); contract=json.loads((out/'collection.json').read_text())
check=idle_observation(contract['affinity'], gpu=contract['gpu'] is not None)
check['timestamp']=time.time()
with (out/'preflight.jsonl').open('a') as stream: stream.write(json.dumps(check)+'\n')
print('Resource preflight:', json.dumps(check), flush=True)
if check['observed_competition']:
    raise SystemExit('Observed competing resource use; retry on idle allocated resources')
if not check['gpu_occupancy_verified']:
    if contract['measurement_role'] in ('main', 'cold', 'resource'):
        raise SystemExit('GPU occupancy unavailable; configure a matching monitoring library or repair monitoring before performance collection')
    print('GPU occupancy unavailable: diagnostic observation only.', flush=True)
CHECK
}
check_resources || exit 3
if [[ "$plan_only" == 1 ]]; then exit 0; fi

active_pid=''
stop() {
    trap - INT TERM
    if [[ -n "$active_pid" ]]; then
        kill -TERM "$active_pid" 2>/dev/null || true
        wait "$active_pid" || true
    fi
    exit "$1"
}
trap 'stop 130' INT
trap 'stop 143' TERM
failed=0
while IFS=$'\t' read -r phase benchmark framework implementation availability; do
    cell="$out/$phase/$benchmark/$framework/$implementation"
    mkdir -p "$cell"
    if [[ -f "$cell/exit-code.txt" ]]; then
        rc="$(cat "$cell/exit-code.txt")"
        echo "Resume: $phase $benchmark $framework $implementation (exit $rc)"
    else
        step_started=$SECONDS
        reason='executed'
        command=("${runner[@]}" "$repo/run_benchmark.py" -b "$benchmark" -p "$NPBENCH_PRESET"
                 --golden-cache "$NPBENCH_GOLDEN_CACHE")
        if [[ "$phase" == golden ]]; then
            command+=(--prepare-golden)
            if [[ "$mode" == cova || -n "${NPBENCH_RETRY_FROM:-}" || "${NPBENCH_REQUIRE_GOLDEN:-0}" == 1 ]]; then command+=(--require-golden); fi
            command=(timeout --kill-after=10s "${NPBENCH_GOLDEN_TIMEOUT}s" "${command[@]}")
        else
            command+=(-f "$framework" --implementation "$implementation" --lifecycle --require-golden
                      --run-dir "$cell/runs" --fresh-process-runs "$NPBENCH_FRESH_PROCESSES"
                      -r "$NPBENCH_REPEATS" -t "$NPBENCH_TIMEOUT")
        fi
        printf '%q ' "${command[@]}" > "$cell/command.sh"
        printf '\n' >> "$cell/command.sh"
        if [[ "$availability" == missing_source ]]; then
            rc=125; reason=missing_source
        elif [[ "$phase" != golden && "$(cat "$out/golden/$benchmark/numpy/reference/exit-code.txt")" != 0 ]]; then
            rc=125; reason=golden_unavailable
        else
            echo "Run: $phase $benchmark $framework $implementation"
            (cd "$cell" && exec "${command[@]}") > "$cell/stdout.log" 2> "$cell/stderr.log" &
            active_pid=$!
            if wait "$active_pid"; then rc=0; else rc=$?; fi
            active_pid=''
        fi
        printf '%s\n' "$((SECONDS-step_started))" > "$cell/step-wall-seconds.txt"
        printf '%s\n' "$reason" > "$cell/reason.txt"
        printf '%s\n' "$rc" > "$cell/exit-code.tmp"
        mv "$cell/exit-code.tmp" "$cell/exit-code.txt"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$phase" "$benchmark" "$framework" "$implementation" "$rc" "$reason" >> "$out/progress.tsv"
    fi
    [[ "$rc" == 0 ]] || failed=$((failed + 1))
done < "$out/plan.tsv"
freeze_collection || exit 3
resource_rc=0
check_resources || resource_rc=3
"$python" "$repo/scripts/analyze_collection.py" "$out" || exit 3
[[ "$resource_rc" == 0 ]] || exit "$resource_rc"
echo "Collection complete: $out; $failed failed or blocked steps. Inspect lifecycle_results and raw logs."
[[ "$failed" == 0 ]]
