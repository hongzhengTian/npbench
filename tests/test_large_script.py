"""The collection shell script delegates measurement and preserves receipts."""
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LargeScriptTest(unittest.TestCase):
    def test_cpu_collection_resume_and_configuration_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env = dict(os.environ, NPBENCH_BENCHMARKS='gemm', NPBENCH_FRAMEWORKS='numpy',
                       NPBENCH_PRESET='S', NPBENCH_FRESH_PROCESSES='1', NPBENCH_REPEATS='1',
                       NPBENCH_TIMEOUT='60', NPBENCH_GOLDEN_TIMEOUT='60', NPBENCH_THREADS='1',
                       NPBENCH_CPUSET=str(min(os.sched_getaffinity(0))),
                       NPBENCH_GOLDEN_CACHE=str(root/'goldens'), NPBENCH_RESOURCE_POLICY='fixed', NPBENCH_REQUIRE_GOLDEN='0')
            command = ['bash', str(ROOT/'run_large.sh'), 'baselines', str(root/'run')]
            subprocess.run(command, env=env, check=True, capture_output=True, timeout=120)
            database = root/'run/baselines/gemm/numpy/default/npbench.db'
            with sqlite3.connect(database) as connection:
                rows = connection.execute('SELECT phase, status, validated FROM lifecycle_results').fetchall()
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(status == 'passed' and valid for _, status, valid in rows))
            self.assertEqual([phase for phase, _, _ in rows],
                             ['initialization', 'same_process', 'fresh_process', 'same_process'])
            receipts = {str(p): p.read_bytes() for p in (root/'run').rglob('exit-code.txt')}
            modified = database.stat().st_mtime_ns
            again = subprocess.run(command, env=env, check=True, capture_output=True, text=True, timeout=60)
            self.assertIn('Resume: baselines gemm numpy default', again.stdout)
            self.assertEqual(database.stat().st_mtime_ns, modified)
            # Re-activating the environment can duplicate search entries;
            # identical effective lookup must resume without rerunning calls.
            duplicate_env = dict(env, PATH=env['PATH'] + os.pathsep + env['PATH'])
            subprocess.run(command, env=duplicate_env, check=True, capture_output=True, timeout=60)
            self.assertEqual(database.stat().st_mtime_ns, modified)
            self.assertEqual(receipts, {str(p): p.read_bytes() for p in (root/'run').rglob('exit-code.txt')})
            # Synthetic old failure evidence; actual S execution uses the
            # existing golden and selects one implementation, not all Numba.
            old = root/'old'; old.mkdir()
            (old/'collection.json').write_text(json.dumps({'environment': {'NPBENCH_PRESET': 'S'}}))
            (old/'plan.tsv').write_text('baselines\tgemm\tnumba\tnopython-mode\tpresent\n')
            cell = old/'baselines/gemm/numba/nopython-mode'
            worker = cell/'runs/old/nopython-mode'; worker.mkdir(parents=True)
            (cell/'exit-code.txt').write_text('1')
            from scripts.retry_affected import MISSING
            (worker/'process-0.jsonl').write_text('\n'.join(map(json.dumps, [
                {'status': 'passed', 'phase': 'initialization'},
                {'status': 'error', 'phase': 'fresh_process', 'failure_stage': 'prepare', 'error': MISSING}]))+'\n')
            retry_env = dict(env, NPBENCH_FRAMEWORKS='numba', NPBENCH_FRESH_PROCESSES='0')
            retry_command = ['bash', str(ROOT/'retry_affected.sh'), str(old), str(root/'retry')]
            subprocess.run(retry_command, env=retry_env, check=True, capture_output=True, timeout=120)
            retry_plan = (root/'retry/plan.tsv').read_text().splitlines()
            self.assertEqual(retry_plan, ['golden\tgemm\tnumpy\treference\tpresent',
                                         'baselines\tgemm\tnumba\tnopython-mode\tpresent'])
            golden_command = (root/'retry/golden/gemm/numpy/reference/command.sh').read_text()
            self.assertIn('--require-golden', golden_command)
            retry_database = root/'retry/baselines/gemm/numba/nopython-mode/npbench.db'
            with sqlite3.connect(retry_database) as connection:
                retry_rows = connection.execute('SELECT status FROM lifecycle_results').fetchall()
            self.assertEqual(retry_rows, [('passed',), ('passed',)])
            retry_manifest = next((retry_database.parent/'runs').glob('*/manifest.json'))
            self.assertEqual(json.loads(retry_manifest.read_text())['golden']['producer_executions'], 0)
            retry_mtime = retry_database.stat().st_mtime_ns
            subprocess.run(retry_command, env=retry_env, check=True, capture_output=True, timeout=60)
            self.assertEqual(retry_database.stat().st_mtime_ns, retry_mtime)
            changed = subprocess.run(command, env=dict(env, NPBENCH_REPEATS='2'),
                                     capture_output=True, text=True, timeout=60)
            self.assertEqual(changed.returncode, 3)
            self.assertIn('Code/environment/selection changed', changed.stderr)
            self.assertEqual(database.stat().st_mtime_ns, modified)
            manifest = json.loads((root/'run/collection.json').read_text())
            self.assertEqual(manifest['metric_version'], 3)
            self.assertEqual(manifest['protocol_version'], 6)
            self.assertEqual(manifest['affinity'], [int(env['NPBENCH_CPUSET'])])


if __name__ == '__main__':
    unittest.main()
