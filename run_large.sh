#!/usr/bin/env bash
# Sequential, resumable collection through NPBench's existing lifecycle CLI.
set -euo pipefail

if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
    cat <<'HELP'
Usage: ./run_large.sh [all|baselines|cova] [RESULT_DIRECTORY]
Activate the prepared CoVA/NPBench environment before running this script.
Defaults: all 54 benchmarks, L, 3 fresh processes, 5 repeats per process,
2 CPU threads on a fixed CPU set, visible GPU 0, 1800s per worker/golden.
Baselines (NumPy, Numba, DaCe CPU/GPU, CuPy) run before CoVA in all mode.
Completed steps, including failures, are kept when resuming the same directory.
Use a new directory for retries, changed code/configuration, or later CoVA runs.

Optional environment:
  NPBENCH_GOLDEN_CACHE       persistent golden directory (default .cache/goldens)
  NPBENCH_THREADS           CPU thread budget (default 2)
  NPBENCH_CPUSET            taskset CPU list (default first allowed CPUs)
  NPBENCH_BENCHMARKS        space-separated subset (default every bench_info entry)
  NPBENCH_FRAMEWORKS        space-separated subset of the selected mode's frameworks
  NPBENCH_PRESET            S/M/L/paper (default L)
  NPBENCH_FRESH_PROCESSES   later processes per implementation (default 3)
  NPBENCH_REPEATS           subsequent calls within every process (default 5)
  NPBENCH_TIMEOUT          seconds per complete worker (default 1800)
  NPBENCH_GOLDEN_TIMEOUT   seconds per reference preparation (default 1800)
  NPBENCH_PYTHON           Python executable (default python)
  NPBENCH_RETRY_FROM       restrict baselines to affected failures in an old collection

cova mode requires existing goldens and never runs their producers.
No dependencies are installed. The script does not promise overnight completion
or publication-ready results; inspect failures, device placement and variability.
HELP
    exit 0
fi
[[ $# -le 2 ]] || { echo "Too many arguments; see --help" >&2; exit 2; }
mode="${1:-all}"
case "$mode" in all|baselines|cova) ;; *) echo 'Expected all, baselines, or cova' >&2; exit 2;; esac
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python="${NPBENCH_PYTHON:-python}"
export NPBENCH_THREADS="${NPBENCH_THREADS:-2}"
export NPBENCH_PRESET="${NPBENCH_PRESET:-L}"
export NPBENCH_FRESH_PROCESSES="${NPBENCH_FRESH_PROCESSES:-3}"
export NPBENCH_REPEATS="${NPBENCH_REPEATS:-5}"
export NPBENCH_TIMEOUT="${NPBENCH_TIMEOUT:-1800}"
export NPBENCH_GOLDEN_TIMEOUT="${NPBENCH_GOLDEN_TIMEOUT:-1800}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONDONTWRITEBYTECODE=1
# shellcheck source=scripts/native-env.sh
source "$repo/scripts/native-env.sh"
export OMP_PROC_BIND=true OMP_PLACES=cores
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
NPBENCH_GOLDEN_CACHE="$(realpath -m -- "${NPBENCH_GOLDEN_CACHE:-$repo/.cache/goldens}")"
export NPBENCH_GOLDEN_CACHE
NPBENCH_CPUSET="${NPBENCH_CPUSET:-$("$python" -c 'import os; n=int(os.environ["NPBENCH_THREADS"]); a=sorted(os.sched_getaffinity(0)); assert n <= len(a), "Insufficient allowed CPUs"; print(",".join(map(str,a[:n])) )')}"
export NPBENCH_CPUSET
runner=(taskset --cpu-list "$NPBENCH_CPUSET" "$python")
out="$(realpath -m -- "${2:-$repo/.cache/large-runs/$(date -u +%Y%m%dT%H%M%SZ)-$mode}")"
mkdir -p "$repo/.cache" "$out"
# One script at a time per checkout, even when result directories differ.
exec 9>"$repo/.cache/large-run.lock"
flock -n 9 || { echo 'Another run_large.sh is active in this checkout' >&2; exit 1; }

# Freeze the plan and the environment used for resume checks. Existing result
# directories are immutable with respect to code, resources and configuration.
freeze_collection() {
"${runner[@]}" - "$repo" "$out" "$mode" <<'PY'
import hashlib, importlib.metadata, json, os, pathlib, subprocess, sys
from npbench.infrastructure import Benchmark, generate_framework
from npbench.infrastructure.lifecycle import METRIC_VERSION, PROTOCOL_VERSION
repo, out = map(pathlib.Path, sys.argv[1:3]); mode = sys.argv[3]
def capture(command):
    try:
        return subprocess.check_output(command, stderr=subprocess.STDOUT, text=True, timeout=30).strip()
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
    assert value >= (0 if key == 'NPBENCH_FRESH_PROCESSES' else 1), key
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
    gpu = cupy.cuda.runtime.getDeviceProperties(0)
    gpu = {k: str(gpu[k]) for k in ('name', 'totalGlobalMem', 'major', 'minor')}
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
contract = {'mode': mode, 'benchmarks': benchmarks, 'frameworks': frameworks,
            'metric_version': METRIC_VERSION, 'protocol_version': PROTOCOL_VERSION, 'npbench_revision': revision(repo), 'sources': sources,
            'framework_versions': versions, 'python': sys.version, 'executable': sys.executable,
            'pip_freeze': capture([sys.executable, '-m', 'pip', 'freeze', '--all']),
            'environment': environment, 'host': os.uname().nodename,
            'affinity': sorted(os.sched_getaffinity(0)), 'gpu': gpu,
            'gpu_identity': capture(['nvidia-smi', '--query-gpu=uuid,name,driver_version', '--format=csv,noheader']) if gpu else None,
            'nvcc': capture(['nvcc', '--version']) if gpu else None}
if retry is not None:
    contract['retry_selection'] = retry
if cova_root:
    contract['cova_revision'] = revision(cova_root)
    contract['cova_diff'] = capture(['git', '-C', str(cova_root), 'diff', 'HEAD', '--binary'])
    extra = subprocess.check_output(['git', '-C', str(cova_root), 'ls-files', '--others', '--exclude-standard', '-z']).decode().split('\0')
    contract['cova_untracked'] = {name: digest(cova_root/name) for name in extra if name and (cova_root/name).is_file()}
    contract['cova_tools'] = {name: digest(cova_root/'build/bin'/name) for name in ('cova-opt', 'cova-translate')
                             if (cova_root/'build/bin'/name).is_file()}
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
                if retry is not None and (name, framework_name, label) not in retry_keys: continue
                files = framework.impl_files(bench)
                exists = any(path.is_file() and (file_label == label or framework_name.startswith('dace_')) for path, file_label in files)
                plan.append((phase, name, framework_name, label, 'present' if exists else 'missing_source'))
if retry is not None:
    assert {(b, f, label) for phase, b, f, label, _ in plan if phase == 'baselines'} == retry_keys, 'Retry implementation unavailable'
(out/'plan.tsv').write_text(''.join('\t'.join(row)+'\n' for row in plan))
(out/'cova-version.txt').write_text(os.environ.get('NPBENCH_COVA_VERSION', 'unselected')+'\n')
print('Collection:', out, '; planned steps:', len(plan), flush=True)
PY
}
freeze_collection
NPBENCH_COVA_VERSION="$(cat "$out/cova-version.txt")"
export NPBENCH_COVA_VERSION

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
        reason='executed'
        command=("${runner[@]}" "$repo/run_benchmark.py" -b "$benchmark" -p "$NPBENCH_PRESET"
                 --golden-cache "$NPBENCH_GOLDEN_CACHE")
        if [[ "$phase" == golden ]]; then
            command+=(--prepare-golden)
            if [[ "$mode" == cova || -n "${NPBENCH_RETRY_FROM:-}" ]]; then command+=(--require-golden); fi
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
        printf '%s\n' "$reason" > "$cell/reason.txt"
        printf '%s\n' "$rc" > "$cell/exit-code.tmp"
        mv "$cell/exit-code.tmp" "$cell/exit-code.txt"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$phase" "$benchmark" "$framework" "$implementation" "$rc" "$reason" >> "$out/progress.tsv"
    fi
    [[ "$rc" == 0 ]] || failed=$((failed + 1))
done < "$out/plan.tsv"
freeze_collection
echo "Collection complete: $out; $failed failed or blocked steps. Inspect lifecycle_results and raw logs."
[[ "$failed" == 0 ]]
