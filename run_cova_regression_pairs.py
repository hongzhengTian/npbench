#!/usr/bin/env python3
"""Prepare or execute small ABBA CoVA lifecycle comparisons on an idle host.

Source the established NPBench CoVA environment first. Both roots must be
independent, already-built source snapshots. Default is plan-only. No checkout,
build, golden generation, competitor execution, or automatic commit occurs.
"""
import argparse
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

GROUPS = {
    'environment': [('azimint_hist_cova_helper', 'llvm_cpu'), ('softmax', 'serial_cpu')],
    'proof': [('mandelbrot1', 'llvm_cpu'), ('mandelbrot1', 'llvm_gpu'),
              ('vadv', 'llvm_cpu'), ('floyd_warshall', 'llvm_gpu')],
    'tiling': [('doitgen', 'openmp_gpu'), ('doitgen', 'llvm_cpu_serial'),
               ('hdiff', 'openmp_gpu'), ('hdiff', 'llvm_cpu')],
    'protect': [('compute', 'openmp_cpu'), ('compute', 'llvm_gpu'),
                ('jacobi_2d', 'llvm_gpu'), ('heat_3d', 'llvm_gpu'),
                ('fdtd_2d', 'llvm_gpu'), ('azimint_naive', 'openmp_gpu')],
}
THREAD_VARS = ('OMP_NUM_THREADS', 'OMP_THREAD_LIMIT', 'OMP_PROC_BIND', 'OMP_PLACES',
               'OMP_DYNAMIC', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'NUMBA_NUM_THREADS', 'COVA_LLVM_CPU_HELPERS_OPENBLAS_THREADS')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def identity(root):
    files = sorted(p for folder in ['cova', 'compiler'] for p in (root / folder).rglob('*')
                   if p.is_file() and p.suffix in {'.py', '.cpp', '.h', '.td'})
    assert files, f'No sources under {root}'
    tools = [root / 'build/bin/cova-opt', root / 'build/bin/cova-translate',
             root / 'build/python_packages/cova/mlir_cova/_mlir_libs/libCovaPythonCAPI.so']
    assert all(p.is_file() for p in tools), 'Provide a complete independent build: ' + str(root)
    file_hashes = {str(p.relative_to(root)): digest(p) for p in files + tools}
    revision = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                              capture_output=True, text=True)
    return {'root': str(root), 'revision': revision.stdout.strip() if revision.returncode == 0 else None,
            'source_and_tool_sha256': hashlib.sha256(json.dumps(file_hashes, sort_keys=True).encode()).hexdigest(),
            'files': file_hashes}

def host_state():
    def cpu():
        fields = list(map(int, Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))
        return sum(fields), fields[3] + fields[4]
    start = cpu()
    time.sleep(.5)
    end = cpu()
    busy = 1 - (end[1] - start[1]) / max(1, end[0] - start[0])
    gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                         capture_output=True, text=True)
    vmstat = dict(line.split() for line in Path('/proc/vmstat').read_text().splitlines())
    pressure = Path('/proc/pressure/memory')
    return {'timestamp': time.time(), 'host_busy_fraction': busy,
            'memory_pressure': pressure.read_text() if pressure.exists() else None,
            'memory_vmstat': {key: int(vmstat[key]) for key in
                              ('compact_stall', 'pgscan_direct', 'allocstall_movable')
                              if key in vmstat},
            'affinity': sorted(os.sched_getaffinity(0)), 'gpu_query_exit': gpu.returncode,
            'gpu_processes_present': bool(gpu.stdout.strip()), 'gpu_error': gpu.stderr.strip()}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-root', type=Path, required=True)
    p.add_argument('--candidate-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--group', choices=GROUPS, default='environment')
    p.add_argument('--case', action='append', metavar='BENCHMARK:ROUTE',
                   help='Explicit cells instead of a predefined group; repeatable')
    p.add_argument('--order', choices=['ABBA', 'AB', 'candidate'], default='ABBA',
                   help='Candidate-only is screening, not a paired comparison')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--preset', choices=['S', 'L'], default='L')
    p.add_argument('--golden-cache', type=Path)
    p.add_argument('--max-host-busy', type=float, default=.15)
    p.add_argument('--expect-cpus', type=int, default=96)
    args = p.parse_args()
    repo = Path(__file__).resolve().parent
    assert (repo / 'run_benchmark.py').is_file(), 'Place this script in NPBench root'
    roots = {k: v.resolve() for k, v in [('baseline', args.baseline_root), ('candidate', args.candidate_root)]}
    assert roots['baseline'] != roots['candidate'], 'Use independent baseline/candidate snapshots'
    assert 0 <= args.max_host_busy < 1
    initial = {k: identity(v) for k, v in roots.items()}
    for tool in ['build/bin/cova-opt', 'build/bin/cova-translate',
                 'build/python_packages/cova/mlir_cova/_mlir_libs/libCovaPythonCAPI.so']:
        assert not os.path.samefile(roots['baseline'] / tool, roots['candidate'] / tool), 'Snapshots share a native build file'
    assert initial['baseline']['source_and_tool_sha256'] != initial['candidate']['source_and_tool_sha256'], 'Identical comparison snapshots'
    output = args.output.resolve()
    assert not output.exists(), 'Use a new output directory; failed/partial rounds remain visible'
    golden = (args.golden_cache or repo / '.cache/goldens').resolve()
    jobs = []
    cells = GROUPS[args.group]
    if args.case:
        cells = [tuple(cell.split(':')) for cell in args.case]
        allowed = {'serial_cpu', 'openmp_cpu', 'llvm_cpu_serial', 'llvm_cpu',
                   'openmp_gpu', 'llvm_gpu'}
        assert all(len(cell) == 2 and cell[0] and cell[1] in allowed for cell in cells), 'Expected BENCHMARK:ROUTE'
    order = {'ABBA': ['baseline', 'candidate', 'candidate', 'baseline'],
             'AB': ['baseline', 'candidate'], 'candidate': ['candidate']}[args.order]
    for benchmark, route in cells:
        for number, variant in enumerate(order):
            destination = output / f'{benchmark}-{route}' / f'{number}-{variant}'
            command = [sys.executable, str(repo / 'run_benchmark.py'), '-b', benchmark,
                       '-f', 'cova_' + route, '--implementation', 'default', '-p', args.preset,
                       '--lifecycle', '--golden-cache', str(golden), '--require-golden',
                       '--fresh-process-runs', '2', '-r', '2', '-v', 'true', '-t', '1800',
                       '--run-dir', str(destination)]
            jobs.append({'benchmark': benchmark, 'route': route, 'variant': variant,
                         'command': command, 'destination': str(destination)})
    output.mkdir(parents=True)
    (output / 'plan.json').write_text(json.dumps({'identities': initial, 'jobs': jobs,
        'resource_policy': 'system', 'comparison_order': args.order, 'lifecycle': 'strict 3 processes x 3 calls',
        'scope': 'Warm and restored Region calls; initialization recorded separately',
        'historical_acceptance': 'Also compare 09-25 per-cell identity; doitgen repaired baseline is 3ec3dae8, not 09-24'}, indent=2))
    print(f'{len(jobs)} lifecycle jobs planned in {output}; execute={args.execute}', flush=True)
    if not args.execute:
        return
    for job in jobs:
        for variant, root in roots.items():
            assert identity(root) == initial[variant], 'Source/build changed after plan freeze'
        state = host_state()
        with (output / 'resources.jsonl').open('a') as stream:
            stream.write(json.dumps(state) + '\n')
        assert len(state['affinity']) == args.expect_cpus, 'CPU allocation does not match requested system policy'
        assert state['host_busy_fraction'] <= args.max_host_busy, 'Host busy; stop without measuring'
        if job['route'].endswith('gpu'):
            assert state['gpu_query_exit'] == 0 and not state['gpu_processes_present'], 'GPU availability/occupancy not verified'
        root = roots[job['variant']]
        env = dict(os.environ)
        for key in THREAD_VARS:
            env.pop(key, None)
        env.update(COVAPATH=str(root),
                   PYTHONPATH=os.pathsep.join([str(root / 'build/python_packages/cova'), str(root), str(repo)]),
                   PATH=str(root / 'build/bin') + os.pathsep + os.environ['PATH'],
                   NPBENCH_RESOURCE_POLICY='system', NPBENCH_MEASUREMENT_ROLE='main',
                   NPBENCH_COVA_VERSION=initial[job['variant']]['source_and_tool_sha256'],
                   PYTHONDONTWRITEBYTECODE='1')
        # Ensure the environment imports the intended snapshot, before timing.
        subprocess.run([sys.executable, '-c',
            'import cova,pathlib,os; assert pathlib.Path(cova.__file__).resolve().parents[1] == pathlib.Path(os.environ["COVAPATH"])'],
            env=env, cwd=output, check=True)
        log = Path(job['destination'] + '.log')
        log.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        with log.open('w') as stream:
            process = subprocess.Popen(job['command'], cwd=repo, env=env,
                                       stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                while process.poll() is None:
                    observation = host_state()
                    observation.update(phase='during_job', benchmark=job['benchmark'],
                                       route=job['route'], variant=job['variant'],
                                       launcher_pid=process.pid)
                    with (output / 'resources.jsonl').open('a') as samples:
                        samples.write(json.dumps(observation) + '\n')
                returncode = process.wait()
            except BaseException:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                raise
        with (output / 'progress.jsonl').open('a') as stream:
            stream.write(json.dumps({**job, 'exit_code': returncode, 'started': started, 'finished': time.time()}) + '\n')
        assert returncode == 0, 'Cell failed; preserve failure and inspect before continuing'
        assert all(identity(root) == initial[variant] for variant, root in roots.items()), 'Snapshot changed during measurement'
    print('Collection complete. Audit strict results, identities, placement and each lifecycle before comparison.')

if __name__ == '__main__':
    main()
