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


if __name__ == '__main__':
    unittest.main()
