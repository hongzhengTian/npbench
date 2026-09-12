"""Optional process orchestration for NPBench's host-to-host call measurements.

Workloads, presets, framework implementations and numerical tolerances remain
owned by NPBench. Native `main` measurements use the original runner/table.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np

from .benchmark import Benchmark
from .framework import generate_framework
from . import golden
from .region import Region


PHASES = ('initialization', 'fresh_process', 'same_process')


def artifact_state(root):
    extensions = {'.so', '.o', '.a', '.cubin', '.ptx', '.nbc', '.nbi', '.sdfg'}
    return {str(p.relative_to(root)): {'sha256': golden.digest(p), 'mtime_ns': p.stat().st_mtime_ns}
            for p in sorted(root.rglob('*')) if p.is_file() and p.suffix in extensions}


def placement(root):
    records = []
    for path in root.glob('.cova_gen/**/*.backend.json'):
        data = json.loads(path.read_text())
        records.append({key: data.get(key) for key in
                        ('requested_device', 'actual_device', 'runtime_variant', 'fallback_reason')})
    return records


def _append(path, record):
    with path.open('a') as stream:
        stream.write(json.dumps(record) + '\n')
        stream.flush()


def worker(request_path):
    request = json.loads(Path(request_path).read_text())
    root = Path.cwd()
    bench = Benchmark(request['benchmark'])
    numpy = generate_framework('numpy')
    bundle, event = golden.load_or_create(bench, request['preset'], numpy,
                                          request['golden_cache'], required=True)
    framework = generate_framework(request['framework'])
    package_root = Path(__file__).resolve().parents[2]
    if golden.source_hashes([p for p, _ in framework.impl_files(bench)], package_root) != request['implementation_sources']:
        raise ValueError('Implementation source changed between processes')
    if framework.version() != request['version']:
        raise ValueError('Framework version changed between processes')
    # Declared inouts plus observed NumPy mutations are required host results.
    writebacks = set(bench.info.get('output_args', [])) | {
        name for name, value in bundle['arrays'].items()
        if not np.array_equal(value, bundle['inputs'][name])}
    before = artifact_state(root)
    region = Region(bench, framework, request['implementation'], writebacks,
                    restore=request['process_index'] > 0, artifacts_available=bool(before))
    for index in range(request['repeat'] + 1):
        phase = ('initialization' if request['process_index'] == 0 else 'fresh_process') if index == 0 else 'same_process'
        record = {key: request[key] for key in ('benchmark', 'framework', 'implementation', 'preset', 'version', 'process_index')}
        record.update(call_index=index, phase=phase, pid=os.getpid(), status='running',
                      time=None, validated=None, golden_key=event['key'],
                      golden_producer_executions=event['producer_executions'], placement=[])
        (root / 'current-call.json').write_text(json.dumps(record))
        data = golden.clone_data(bundle['inputs'])
        try:
            started = time.perf_counter()
            result = region(data)
            record['time'] = time.perf_counter() - started
            actual = {'returns': result, 'arrays': region.observe_arrays(data)}
            record['validated'] = golden.validate(bench, bundle, actual) if request['validate'] else None
            record['status'] = 'validation_failed' if record['validated'] is False else 'passed'
            record['placement'] = placement(root)
            record['artifact_policy'] = framework.artifact_policy(region.impl)
            after = artifact_state(root)
            record['artifacts'] = {'before': before, 'after': after,
                                   'state': 'unchanged' if before and before == after else 'changed' if after else 'none'}
            if record['status'] == 'passed' and phase != 'initialization' and record['artifact_policy'] not in ('none', 'python') and before != after:
                record['status'] = 'artifact_reuse_failed'
            before = after
        except Exception as error:
            record['status'] = 'error'
            record['failure_stage'] = region.stage
            record['error'] = type(error).__name__ + ': ' + str(error)
            traceback.print_exc()
        _append(root / request['samples_file'], record)
        print(phase, record['status'], record['time'], flush=True)
        if record['status'] != 'passed':
            return 1
    return 0


def save_results(database, run_id, records):
    # Separate table prevents unchanged upstream plotting scripts from mixing
    # host-to-host lifecycle samples with original device-resident timings.
    with sqlite3.connect(database) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS lifecycle_results (
            id INTEGER PRIMARY KEY, run_id TEXT, benchmark TEXT, framework TEXT,
            implementation TEXT, preset TEXT, version TEXT, phase TEXT,
            process_index INTEGER, call_index INTEGER, time REAL, validated INTEGER,
            status TEXT, golden_key TEXT, record_json TEXT)''')
        for row in records:
            keys = ('benchmark', 'framework', 'implementation', 'preset', 'version', 'phase',
                    'process_index', 'call_index', 'time', 'validated', 'status', 'golden_key')
            conn.execute('INSERT INTO lifecycle_results VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                         (run_id, *[row.get(key) for key in keys], json.dumps(row)))


def run(args):
    bench = Benchmark(args['benchmark'])
    cache = Path(args.get('golden_cache') or '.cache/goldens').expanduser().resolve()
    bundle, event = golden.load_or_create(bench, args['preset'], generate_framework('numpy'),
                                          cache, required=args.get('require_golden', False))
    framework = generate_framework(args['framework'])
    supported = framework.fname in ('numpy', 'numba', 'cupy', 'dace_cpu', 'dace_gpu') or framework.fname.startswith('cova_')
    if not supported:
        raise ValueError('Host-to-host lifecycle adapter is not available for ' + framework.fname)
    labels = framework.implementation_names(bench)
    selected = args.get('implementation')
    if selected:
        if selected not in labels:
            raise ValueError('Unknown implementation: ' + selected)
        labels = [selected]
    base = Path(args.get('run_dir') or '.cache/lifecycle').expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    run_root = Path(tempfile.mkdtemp(prefix=bench.bname + '-' + framework.fname + '-', dir=base))
    database = Path('npbench.db').resolve()
    package_root = str(Path(__file__).resolve().parents[2])
    env = dict(os.environ, PYTHONPATH=package_root + os.pathsep + os.environ.get('PYTHONPATH', ''))
    version = framework.version()
    manifest = {'schema': 1, 'metric': 'host_to_host_call_wall_seconds', 'run_id': run_root.name,
                'started_utc': datetime.now(timezone.utc).isoformat(), 'arguments': args,
                'golden': event, 'framework_version': version, 'python': sys.version,
                'host': os.uname().nodename,
                'implementation_sources': golden.source_hashes([p for p, _ in framework.impl_files(bench)], package_root),
                'environment': {k: v for k, v in env.items() if k.startswith(('OMP_', 'COVA_', 'NUMBA_', 'OPENBLAS_', 'MKL_', 'NPBENCH_'))},
                'processes': [], 'cancelled_processes': [], 'status': 'running'}
    manifest_file = run_root / 'manifest.json'
    manifest_file.write_text(json.dumps(manifest, indent=2) + '\n')
    all_records = []
    for label in labels:
        cell = run_root / label
        cell.mkdir()
        child_env = dict(env, NUMBA_CACHE_DIR=str(cell / 'numba-cache'), CUPY_CACHE_DIR=str(cell / 'cupy-cache'))
        for process_index in range(args['fresh_process_runs'] + 1):
            stem = 'process-' + str(process_index)
            samples_file = stem + '.jsonl'
            request = {'benchmark': bench.bname, 'framework': framework.fname, 'implementation': label,
                       'preset': args['preset'], 'repeat': args['repeat'], 'process_index': process_index,
                       'validate': args['validate'], 'golden_cache': str(cache), 'version': version,
                       'samples_file': samples_file, 'implementation_sources': manifest['implementation_sources']}
            request_path = cell / (stem + '.request.json')
            request_path.write_text(json.dumps(request))
            command = [sys.executable, '-m', 'npbench.infrastructure.lifecycle', '--worker', str(request_path)]
            (cell / 'current-call.json').unlink(missing_ok=True)
            started = time.monotonic()
            timed_out = False
            with (cell / (stem + '.stdout.log')).open('w') as out, (cell / (stem + '.stderr.log')).open('w') as err:
                process = subprocess.Popen(command, cwd=cell, env=child_env, stdout=out, stderr=err, start_new_session=True)
                try:
                    process.wait(timeout=args['timeout'])
                except subprocess.TimeoutExpired:
                    timed_out = True
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
            records = [json.loads(line) for line in (cell / samples_file).read_text().splitlines()] if (cell / samples_file).exists() else []
            if process.returncode and (not records or records[-1]['status'] == 'passed'):
                current = cell / 'current-call.json'
                row = json.loads(current.read_text()) if current.exists() else dict(request)
                # A previous process's progress must never become this failure.
                row.update(process_index=process_index, status='timeout' if timed_out else 'process_failed',
                           time=None, validated=None, error='worker exit ' + str(process.returncode))
                row.setdefault('phase', 'initialization' if process_index == 0 else 'fresh_process')
                row.setdefault('golden_key', event['key'])
                records.append(row)
                _append(cell / samples_file, row)
            if process.returncode == 0 and len(records) != args['repeat'] + 1:
                row = dict(request, phase='initialization' if process_index == 0 else 'fresh_process',
                           status='incomplete', time=None, validated=None, golden_key=event['key'],
                           error='Worker did not record every requested call')
                records.append(row)
                _append(cell / samples_file, row)
            all_records.extend(records)
            save_results(database, run_root.name, records)
            manifest['processes'].append({'implementation': label, 'index': process_index,
                                          'exit_code': process.returncode, 'timed_out': timed_out,
                                          'worker_wall_seconds': time.monotonic() - started,
                                          'samples': str((cell / samples_file).relative_to(run_root))})
            manifest_file.write_text(json.dumps(manifest, indent=2) + '\n')
            print(label, stem, 'exit', process.returncode, flush=True)
            if process.returncode or any(r['status'] != 'passed' for r in records):
                manifest['cancelled_processes'].extend(
                    {'implementation': label, 'index': index, 'reason': 'earlier worker failed'}
                    for index in range(process_index + 1, args['fresh_process_runs'] + 1))
                break
    manifest['status'] = 'passed' if all_records and all(r['status'] == 'passed' for r in all_records) else 'failed'
    manifest_file.write_text(json.dumps(manifest, indent=2) + '\n')
    print('Lifecycle results:', run_root, flush=True)
    return 0 if manifest['status'] == 'passed' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', required=True)
    sys.exit(worker(parser.parse_args().worker))
