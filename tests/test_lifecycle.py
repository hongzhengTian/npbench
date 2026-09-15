import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from npbench.infrastructure import Benchmark, generate_framework
from npbench.infrastructure.region import Region
from npbench.infrastructure import golden

ROOT = Path(__file__).resolve().parents[1]


class LifecycleTest(unittest.TestCase):
    def test_release_does_not_keep_previous_call_arrays_alive(self):
        import weakref
        bench = Benchmark('gemm')
        bench.info['parameters']['unit'] = {'NI': 4, 'NJ': 3, 'NK': 2}
        framework = generate_framework('numpy')
        data = golden.clone_data(bench.get_data('unit'))
        array = weakref.ref(data['A'])
        region = Region(bench, framework, 'default', {'C'})
        region(data)
        del data
        self.assertIsNotNone(array())
        region.release()
        self.assertIsNone(array())

    def test_parent_termination_stops_its_active_worker(self):
        import signal
        import time
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1')
            command = [sys.executable, str(ROOT / 'run_benchmark.py'), '-b', 'gemm', '-f', 'numpy',
                       '--lifecycle', '-r', '100000', '--fresh-process-runs', '0', '-t', '60',
                       '--golden-cache', str(root / 'goldens'), '--run-dir', str(root / 'runs')]
            process = subprocess.Popen(command, cwd=root, env=env, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL)
            worker_pid = None
            try:
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    samples = list((root / 'runs').glob('*/*/process-0.jsonl'))
                    if samples and samples[0].stat().st_size:
                        worker_pid = json.loads(samples[0].read_text().splitlines()[0])['pid']
                        break
                    self.assertIsNone(process.poll())
                    time.sleep(0.05)
                self.assertIsNotNone(worker_pid)
                process.terminate()
                self.assertEqual(process.wait(timeout=10), 128 + signal.SIGTERM)
                with self.assertRaises(ProcessLookupError):
                    os.kill(worker_pid, 0)
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=10)
                if worker_pid:
                    try:
                        os.killpg(worker_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_source_preparation_does_not_execute_and_calls_return_their_own_output(self):
        bench = Benchmark('gemm')
        bench.info['parameters']['unit'] = {'NI': 4, 'NJ': 3, 'NK': 2}
        framework = generate_framework('numpy')
        bundle = golden.reference(bench, 'unit', framework)
        implementation = framework.load_implementation(bench, 'default')
        from unittest.mock import Mock
        observed = Mock(wraps=implementation)
        with patch.object(framework, 'load_implementation', return_value=observed) as loader:
            region = Region(bench, framework, 'default', {'C'})
            self.assertEqual(loader.call_count, 1)
            self.assertEqual(observed.call_count, 0)
            for _ in range(3):
                data = golden.clone_data(bundle['inputs'])
                result = region(data)
                self.assertTrue(golden.validate(bench, bundle,
                                {'returns': result, 'arrays': region.observe_arrays(data)}))
            self.assertEqual(loader.call_count, 1)
            self.assertEqual(observed.call_count, 3)

    def test_dace_source_import_does_not_prepare_graph_or_restore_binary(self):
        from npbench.infrastructure.dace_framework import DaceFramework
        framework = DaceFramework('dace_cpu')
        with patch('npbench.infrastructure.dace_framework.importlib.import_module') as importer, \
                patch.object(framework, 'implementations', side_effect=AssertionError('compiled')):
            lazy = framework.load_implementation(Benchmark('gemm'), 'fusion')
            self.assertTrue(callable(lazy))
            importer.assert_called_once()
            with self.assertRaisesRegex(AssertionError, 'compiled'):
                lazy()

    def test_subprocess_phases_and_separate_result_table(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1')
            command = [sys.executable, str(ROOT / 'run_benchmark.py'), '-b', 'gemm', '-f', 'numpy',
                       '--lifecycle', '-r', '2', '--fresh-process-runs', '2', '-t', '60',
                       '--golden-cache', str(root / 'goldens'), '--run-dir', str(root / 'runs')]
            subprocess.run(command, cwd=root, env=env, check=True, capture_output=True, timeout=90)
            with sqlite3.connect(root / 'npbench.db') as conn:
                rows = [json.loads(r[0]) for r in conn.execute('SELECT record_json FROM lifecycle_results')]
                self.assertFalse(conn.execute("SELECT name FROM sqlite_master WHERE name='results'").fetchall())
            self.assertEqual(len(rows), 9)
            self.assertEqual([r['phase'] for r in rows].count('initialization'), 1)
            self.assertEqual([r['phase'] for r in rows].count('fresh_process'), 2)
            self.assertEqual([r['phase'] for r in rows].count('same_process'), 6)
            self.assertEqual(len({r['pid'] for r in rows}), 3)
            self.assertTrue(all(r['validated'] and r['status'] == 'passed' and r['time'] > 0 for r in rows))
            self.assertEqual(len(list((root / 'goldens').glob('*/golden.json'))), 1)
            again = subprocess.run([*command, '--require-golden'], cwd=root, env=env, check=True,
                                   capture_output=True, timeout=90)
            manifests = [json.loads(p.read_text()) for p in (root / 'runs').glob('*/manifest.json')]
            self.assertEqual(sorted(m['golden']['producer_executions'] for m in manifests), [0, 1])

    def test_numba_discovery_and_plain_python_implementation(self):
        bench = Benchmark('mvt')
        framework = generate_framework('numba')
        self.assertEqual(set(framework.implementation_names(bench)),
                         {name for _, name in framework.implementations(bench)})
        implementation = framework.load_implementation(bench, 'nopython-mode-parallel', restore=True)
        self.assertEqual(framework.artifact_policy(implementation), 'python')
        self.assertFalse(hasattr(implementation, 'enable_caching'))
        bench.info['parameters']['unit'] = {'N': 8}
        bundle = golden.reference(bench, 'unit', generate_framework('numpy'))
        region = Region(bench, framework, 'nopython-mode-parallel', {'x1', 'x2'},
                        restore=True, artifacts_available=False)
        data = golden.clone_data(bundle['inputs'])
        self.assertTrue(golden.validate(bench, bundle,
                        {'returns': region(data), 'arrays': region.observe_arrays(data)}))

    def test_missing_native_artifact_fails_before_executing(self):
        bench = Benchmark('gemm')
        framework = generate_framework('numpy')
        with patch.object(framework, 'artifact_policy', return_value='native'), \
                patch.object(framework, 'load_implementation', return_value=lambda *args: self.fail('executed')):
            region = Region(bench, framework, 'default', {'C'}, restore=True, artifacts_available=False)
            with self.assertRaises(FileNotFoundError):
                region({})

    def test_timeout_is_persisted_and_later_processes_are_cancelled(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1')
            command = [sys.executable, str(ROOT / 'run_benchmark.py'), '-b', 'gemm', '-f', 'numpy',
                       '--lifecycle', '-r', '1', '--fresh-process-runs', '2', '-t', '0.001',
                       '--golden-cache', str(root / 'goldens'), '--run-dir', str(root / 'runs')]
            result = subprocess.run(command, cwd=root, env=env, capture_output=True, timeout=60)
            self.assertNotEqual(result.returncode, 0)
            manifest_path, = (root / 'runs').glob('*/manifest.json')
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest['status'], 'failed')
            self.assertEqual(len(manifest['processes']), 1)
            self.assertEqual(len(manifest['cancelled_processes']), 2)
            samples, = manifest_path.parent.glob('*/*.jsonl')
            row, = [json.loads(line) for line in samples.read_text().splitlines()]
            self.assertEqual(row['status'], 'timeout')
            self.assertIsNone(row['time'])
            with sqlite3.connect(root / 'npbench.db') as conn:
                self.assertEqual(conn.execute('SELECT status, time FROM lifecycle_results').fetchall(), [('timeout', None)])

    def test_native_mode_can_reuse_golden(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1')
            command = [sys.executable, str(ROOT / 'run_benchmark.py'), '-b', 'gemm', '-f', 'numpy',
                       '-r', '1', '--golden-cache', str(root / 'goldens')]
            subprocess.run(command, cwd=root, env=env, check=True, capture_output=True, timeout=60)
            result = subprocess.run([*command, '--require-golden'], cwd=root, env=env, check=True,
                                    capture_output=True, text=True, timeout=60)
            self.assertIn("'producer_executions': 0", result.stdout)
            with sqlite3.connect(root / 'npbench.db') as conn:
                rows = conn.execute('SELECT validated FROM results').fetchall()
            self.assertEqual(rows, [(1,), (1,)])
            subprocess.run([*command, '--require-golden', '-v', 'false'], cwd=root, env=env, check=True,
                           capture_output=True, timeout=60)
            with sqlite3.connect(root / 'npbench.db') as conn:
                self.assertIsNone(conn.execute('SELECT validated FROM results ORDER BY id DESC LIMIT 1').fetchone()[0])


class ReuseClassificationTest(unittest.TestCase):
    def test_dace_restore_reports_original_deserialization_error_without_compiling(self):
        from npbench.infrastructure.dace_framework import DaceRestoreError
        framework = generate_framework('dace_cpu')
        bench = Benchmark('gemm')
        saved = json.dumps({'schema': 2, 'label': 'fusion', 'version': framework.version(),
                            'build_folder': 'dace-cache/fusion'})
        with patch('dace.sdfg.utils.load_precompiled_sdfg', side_effect=TypeError('Offset must be the same size as shape')), \
                patch('pathlib.Path.read_text', return_value=saved), \
                patch.object(framework, 'implementations', side_effect=AssertionError('compiled')), \
                patch('npbench.infrastructure.dace_framework.importlib.import_module'):
            implementation = framework.load_implementation(bench, 'fusion', restore=True)
            with self.assertRaisesRegex(DaceRestoreError, 'Offset must be the same size as shape') as caught:
                implementation()
            self.assertEqual(caught.exception.failure_stage, 'restore')


    def test_dace_scalar_abi_normalization_preserves_array_shape_and_dtype_checks(self):
        framework = generate_framework('dace_cpu')
        bench = Benchmark('crc16')
        expected = {'returns': [np.int64(42)], 'arrays': {}}
        normalized = framework.normalize_returns([np.array([42], dtype=np.int64)], (True,))
        self.assertTrue(golden.validate(bench, expected, {'returns': normalized, 'arrays': {}}))
        for values, scalar in [([np.array([42], dtype=np.int64)], (False,)),
                               ([np.array([42], dtype=np.int32)], (True,)),
                               ([np.array([42, 42], dtype=np.int64)], (True,)),
                               ([np.array([43], dtype=np.int64)], (True,)),
                               ([np.array([42], dtype=np.int64), np.int64(42)], (True,))]:
            normalized = framework.normalize_returns(values, scalar)
            self.assertFalse(golden.validate(bench, expected, {'returns': normalized, 'arrays': {}}))


    def test_dace_worker_cache_environment_ignores_inherited_shared_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shared = root/'shared'; shared.mkdir()
            env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1',
                       OPENBLAS_NUM_THREADS='1', DACE_default_build_folder=str(shared),
                       DACE_cache='single', DACE_compiler_use_cache='true')
            # Force an immediate timeout: this tests launch configuration for
            # all variants without making the unit suite compile DaCe graphs.
            command = [sys.executable, str(ROOT/'run_benchmark.py'), '-b', 'gemm', '-p', 'S',
                       '-f', 'dace_cpu', '--lifecycle', '-r', '1', '--fresh-process-runs', '0',
                       '-t', '0.001', '--golden-cache', str(root/'goldens'), '--run-dir', str(root/'runs')]
            result = subprocess.run(command, cwd=root, env=env, capture_output=True, timeout=60)
            self.assertEqual(result.returncode, 1)
            path, = (root/'runs').glob('*/manifest.json')
            manifest = json.loads(path.read_text())
            self.assertEqual(set(manifest['artifact_environments']), {'fusion', 'parallel', 'auto_opt'})
            for label, actual in manifest['artifact_environments'].items():
                self.assertEqual(actual['DACE_default_build_folder'], str(path.parent/label/'dace-cache'))
                self.assertEqual(actual['DACE_cache'], 'name')
                self.assertEqual(actual['DACE_compiler_use_cache'], 'false')
            self.assertEqual(list(shared.iterdir()), [])

    def test_numba_lifted_code_preserves_local_calls_and_marks_fresh_unsupported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1',
                       OPENBLAS_NUM_THREADS='1', NUMBA_NUM_THREADS='1', OMP_NUM_THREADS='1')
            command = [sys.executable, str(ROOT/'run_benchmark.py'), '-b', 'correlation',
                       '-p', 'S', '-f', 'numba', '--implementation', 'object-mode',
                       '--lifecycle', '-r', '1', '--fresh-process-runs', '2', '-t', '90',
                       '--golden-cache', str(root/'goldens'), '--run-dir', str(root/'runs')]
            result = subprocess.run(command, cwd=root, env=env, capture_output=True, timeout=120)
            self.assertEqual(result.returncode, 2, result.stderr.decode())
            manifest_path, = (root/'runs').glob('*/manifest.json')
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest['status'], 'partial')
            self.assertEqual(len(manifest['processes']), 1)
            self.assertEqual(len(list(manifest_path.parent.glob('*/*.request.json'))), 1)
            with sqlite3.connect(root/'npbench.db') as connection:
                rows = connection.execute('SELECT phase,status,time,validated FROM lifecycle_results').fetchall()
            self.assertEqual([r[:2] for r in rows], [('initialization', 'passed'), ('same_process', 'passed'),
                                                   ('fresh_process', 'reuse_unsupported'), ('fresh_process', 'reuse_unsupported')])
            self.assertTrue(all(r[2] is None and r[3] is None for r in rows[2:]))

    def test_worker_rejects_changed_artifacts_before_importing_implementation(self):
        from npbench.infrastructure.lifecycle import worker
        bench = Benchmark('gemm')
        request = {'benchmark': 'gemm', 'preset': 'S', 'framework': 'numpy',
                   'golden_cache': '/unused', 'implementation_sources': {},
                   'version': 'unit', 'expected_artifacts': {'missing.so': {}}}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as stream:
            json.dump(request, stream); stream.flush()
            with patch('npbench.infrastructure.lifecycle.golden.load_or_create', return_value=({'arrays': {}, 'inputs': {}}, {})), \
                    patch('npbench.infrastructure.lifecycle.golden.source_hashes', return_value={}), \
                    patch('npbench.infrastructure.lifecycle.artifact_state', return_value={}), \
                    patch('npbench.infrastructure.framework.Framework.version', return_value='unit'), \
                    patch('npbench.infrastructure.lifecycle.Region', side_effect=AssertionError('implementation imported')):
                with self.assertRaisesRegex(ValueError, 'Artifact integrity'):
                    worker(stream.name)


if __name__ == '__main__':
    unittest.main()
