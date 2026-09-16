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
METRIC_VERSION = 3
PROTOCOL_VERSION = 5
VALIDATION_CONTRACT = 'strict_region_v1'


def _stop_worker(process):
    """Also stop compiler descendants when a worker or its parent is stopped."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


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
    worker_started = time.perf_counter()
    from .resources import process_resources
    launch_resources = process_resources()
    event = {}
    try:
        stage = 'load_benchmark'
        bench = Benchmark(request['benchmark'])
        numpy = generate_framework('numpy')
        stage = 'load_golden'
        bundle, event = golden.load_or_create(bench, request['preset'], numpy,
                                              request['golden_cache'], required=True)
        if request.get('golden_sha256') is not None and event['sha256'] != request['golden_sha256']:
            raise ValueError('Golden payload changed between controller and worker')
        stage = 'load_framework'
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
        stage = 'artifact_integrity'
        before = artifact_state(root)
        if request.get('expected_artifacts') is not None and before != request['expected_artifacts']:
            raise ValueError('Artifact integrity changed between processes')
        stage = 'load_implementation'
        region = Region(bench, framework, request['implementation'], writebacks,
                        restore=request['process_index'] > 0, artifacts_available=bool(before),
                        scalar_returns=tuple(np.ndim(value) == 0 for value in bundle['returns']))
    except Exception as error:
        row = {k: request[k] for k in ('benchmark', 'framework', 'implementation', 'preset', 'version', 'process_index')}
        row.update(call_index=0, phase='initialization' if request['process_index'] == 0 else 'fresh_process',
                   pid=os.getpid(), status='error', time=None, validated=None, failure_stage=stage,
                   error=type(error).__name__ + ': ' + str(error), error_type=type(error).__name__,
                   golden_key=event.get('key'), golden_sha256=event.get('sha256'), metric_version=METRIC_VERSION, protocol_version=PROTOCOL_VERSION,
                   validation_contract=VALIDATION_CONTRACT, launch_resources=launch_resources)
        _append(root / request['samples_file'], row)
        traceback.print_exc()
        return 1
    preparation_seconds = time.perf_counter() - worker_started
    for index in range(request['repeat'] + 1):
        phase = ('initialization' if request['process_index'] == 0 else 'fresh_process') if index == 0 else 'same_process'
        record = {key: request[key] for key in ('benchmark', 'framework', 'implementation', 'preset', 'version', 'process_index')}
        record.update(call_index=index, phase=phase, pid=os.getpid(), status='running',
                      time=None, validated=None, golden_key=event['key'], golden_sha256=event['sha256'],
                      golden_producer_executions=event['producer_executions'], placement=[],
                      metric_version=METRIC_VERSION, protocol_version=PROTOCOL_VERSION,
                      validation_contract=VALIDATION_CONTRACT, launch_resources=launch_resources,
                      preparation_seconds=preparation_seconds if index == 0 else 0.)
        (root / 'current-call.json').write_text(json.dumps(record))
        clone_started = time.perf_counter()
        data = golden.clone_data(bundle['inputs'])
        record['input_reset_seconds'] = time.perf_counter() - clone_started
        result = actual = None
        try:
            probe_cpu = os.environ.get('NPBENCH_RESOURCE_PROBE') == '1'
            if probe_cpu:
                from .resources import thread_cpu_snapshot, thread_cpu_activity
                cpu_before = thread_cpu_snapshot()
            started = time.perf_counter()
            result = region(data)
            record['time'] = time.perf_counter() - started
            if probe_cpu:
                record['thread_cpu_activity'] = thread_cpu_activity(cpu_before, thread_cpu_snapshot())
            post_started = time.perf_counter()
            region.stage = 'validate'
            actual = {'returns': result, 'arrays': region.observe_arrays(data)}
            record['validation_failures'] = []
            record['validated'] = golden.validate(bench, bundle, actual,
                                                   diagnostics=record['validation_failures']) if request['validate'] else None
            record['status'] = 'validation_failed' if record['validated'] is False else 'passed'
            if record['validated'] is False:
                record['validation_audit'] = golden.validation_audit(bench, bundle, actual, record['validation_failures'])
            if index == 0:
                from .resources import observe_resources
                try:
                    resource_observation = observe_resources(framework, region)
                except (RuntimeError, OSError, AttributeError, ValueError) as error:
                    resource_observation = {'probe_error': type(error).__name__ + ': ' + str(error),
                                            'resources_verified': False}
            record['resource_observation'] = resource_observation
            record['validation_observation_seconds'] = time.perf_counter() - post_started
            audit_started = time.perf_counter()
            record['placement'] = placement(root)
            record['artifact_policy'] = framework.artifact_policy(region.impl)
            after = artifact_state(root)
            record['artifacts'] = {'before': before, 'after': after,
                                   'state': 'unchanged' if before and before == after else 'changed' if after else 'none'}
            if record['status'] == 'passed' and phase != 'initialization' and record['artifact_policy'] not in ('none', 'python', 'numba_memory_only') and before != after:
                record['status'] = 'artifact_reuse_failed'
            before = after
            record['artifact_audit_seconds'] = time.perf_counter() - audit_started
        except Exception as error:
            record['status'] = 'error'
            record['failure_stage'] = getattr(error, 'failure_stage', region.stage)
            record['error'] = type(error).__name__ + ': ' + str(error)
            record['error_type'] = type(error).__name__
            traceback.print_exc()
        finally:
            # Keep the compiled callable, but not the preceding call's input,
            # output or device buffers during the next allocation and timer.
            region.release()
            del data, result, actual
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
    previous = signal.getsignal(signal.SIGTERM)

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    try:
        return _run(args)
    finally:
        signal.signal(signal.SIGTERM, previous)


def _run(args):
    bench = Benchmark(args['benchmark'])
    cache = Path(args.get('golden_cache') or '.cache/goldens').expanduser().resolve()
    _, event = golden.load_or_create(bench, args['preset'], generate_framework('numpy'),
                                     cache, required=args.get('require_golden', False), load_values=False)
    # Initializer data can be cached by Benchmark after a golden miss.
    # Only workers need the actual arrays during measurements.
    bench.bdata.clear()
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
    manifest = {'schema': 1, 'metric': 'host_to_host_call_wall_seconds',
                'metric_version': METRIC_VERSION, 'protocol_version': PROTOCOL_VERSION, 'validation_contract': VALIDATION_CONTRACT, 'run_id': run_root.name,
                'started_utc': datetime.now(timezone.utc).isoformat(), 'arguments': args,
                'validation_policy': {'contract': VALIDATION_CONTRACT, 'rtol': bench.info.get('rtol', 1e-5),
                                      'atol': bench.info.get('atol', 1e-8), 'norm_error': bench.info.get('norm_error', 1e-5),
                                      'declared_output_args': bench.info.get('output_args', []), 'audit_all_array_args': True},
                'golden': event, 'framework_version': version, 'python': sys.version,
                'host': os.uname().nodename,
                'implementation_sources': golden.source_hashes([p for p, _ in framework.impl_files(bench)], package_root),
                'environment': {k: v for k, v in env.items() if k.startswith(('OMP_', 'COVA_', 'NUMBA_', 'OPENBLAS_', 'MKL_', 'NPBENCH_', 'DACE_', 'CUPY_', 'CUDA'))},
                'processes': [], 'cancelled_processes': [], 'artifact_environments': {}, 'status': 'running'}
    (Path.cwd() / 'selected-run.txt').write_text(run_root.name + '\n')
    manifest_file = run_root / 'manifest.json'
    manifest_file.write_text(json.dumps(manifest, indent=2) + '\n')
    all_records = []
    for label in labels:
        cell = run_root / label
        cell.mkdir()
        child_env = dict(env, NUMBA_CACHE_DIR=str(cell / 'numba-cache'), CUPY_CACHE_DIR=str(cell / 'cupy-cache'))
        if framework.fname.startswith('dace_'):
            # Override inherited site-wide caches before DaCe is imported in
            # the worker. Both CPU/GPU variants need isolated, stable paths.
            child_env.update(DACE_default_build_folder=str(cell / 'dace-cache'),
                             DACE_cache='name', DACE_compiler_use_cache='false')
        manifest['artifact_environments'][label] = {k: child_env[k] for k in
            ('NUMBA_CACHE_DIR', 'CUPY_CACHE_DIR', 'DACE_default_build_folder',
             'DACE_cache', 'DACE_compiler_use_cache') if k in child_env}
        expected_artifacts = None
        for process_index in range(args['fresh_process_runs'] + 1):
            stem = 'process-' + str(process_index)
            samples_file = stem + '.jsonl'
            request = {'benchmark': bench.bname, 'framework': framework.fname, 'implementation': label,
                       'preset': args['preset'], 'repeat': args['repeat'], 'process_index': process_index,
                       'validate': args['validate'], 'golden_cache': str(cache), 'golden_sha256': event['sha256'], 'version': version,
                       'samples_file': samples_file, 'implementation_sources': manifest['implementation_sources'],
                       'expected_artifacts': expected_artifacts}
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
                    _stop_worker(process)
                except BaseException:
                    _stop_worker(process)
                    raise
            records = [json.loads(line) for line in (cell / samples_file).read_text().splitlines()] if (cell / samples_file).exists() else []
            if process.returncode and (not records or records[-1]['status'] == 'passed'):
                current = cell / 'current-call.json'
                row = json.loads(current.read_text()) if current.exists() else dict(request)
                # A previous process's progress must never become this failure.
                row.update(process_index=process_index, status='timeout' if timed_out else 'process_failed',
                           time=None, validated=None, error='worker exit ' + str(process.returncode))
                row.setdefault('phase', 'initialization' if process_index == 0 else 'fresh_process')
                row.setdefault('golden_key', event['key'])
                row.update(metric_version=METRIC_VERSION, protocol_version=PROTOCOL_VERSION, validation_contract=VALIDATION_CONTRACT)
                records.append(row)
                _append(cell / samples_file, row)
            if process.returncode == 0 and len(records) != args['repeat'] + 1:
                row = dict(request, phase='initialization' if process_index == 0 else 'fresh_process',
                           status='incomplete', time=None, validated=None, golden_key=event['key'], golden_sha256=event['sha256'],
                           error='Worker did not record every requested call')
                row.update(metric_version=METRIC_VERSION, protocol_version=PROTOCOL_VERSION, validation_contract=VALIDATION_CONTRACT)
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
            if process.returncode == 0 and records and all(r['status'] == 'passed' for r in records):
                expected_artifacts = records[-1]['artifacts']['after']
                if records[-1]['artifact_policy'] == 'numba_memory_only':
                    unavailable = []
                    for index in range(process_index + 1, args['fresh_process_runs'] + 1):
                        row = {k: records[0][k] for k in
                               ('benchmark', 'framework', 'implementation', 'preset', 'version', 'golden_key')}
                        row.update(phase='fresh_process', process_index=index, call_index=0,
                                   status='reuse_unsupported', time=None, validated=None,
                                   artifact_policy='numba_memory_only', metric_version=METRIC_VERSION,
                                   protocol_version=PROTOCOL_VERSION,
                                   error='Numba specialization contains non-cacheable lifted code or dynamic globals')
                        unavailable.append(row)
                        _append(cell / ('process-' + str(index) + '.jsonl'), row)
                        manifest['cancelled_processes'].append(
                            {'implementation': label, 'index': index, 'reason': 'reuse_unsupported'})
                    all_records.extend(unavailable)
                    save_results(database, run_root.name, unavailable)
                    break
            if process.returncode or any(r['status'] != 'passed' for r in records):
                manifest['cancelled_processes'].extend(
                    {'implementation': label, 'index': index, 'reason': 'earlier worker failed'}
                    for index in range(process_index + 1, args['fresh_process_runs'] + 1))
                break
    statuses = {r['status'] for r in all_records}
    manifest['status'] = ('passed' if statuses == {'passed'} else 'partial'
                          if statuses and statuses <= {'passed', 'reuse_unsupported'} else 'failed')
    manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
    manifest_file.write_text(json.dumps(manifest, indent=2) + '\n')
    print('Lifecycle results:', run_root, flush=True)
    return {'passed': 0, 'partial': 2, 'failed': 1}[manifest['status']]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', required=True)
    sys.exit(worker(parser.parse_args().worker))
