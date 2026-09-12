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
                       NPBENCH_GOLDEN_CACHE=str(root/'goldens'))
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
            self.assertEqual(receipts, {str(p): p.read_bytes() for p in (root/'run').rglob('exit-code.txt')})
            changed = subprocess.run(command, env=dict(env, NPBENCH_REPEATS='2'),
                                     capture_output=True, text=True, timeout=60)
            self.assertNotEqual(changed.returncode, 0)
            self.assertIn('Code/environment/selection changed', changed.stderr)
            self.assertEqual(database.stat().st_mtime_ns, modified)
            manifest = json.loads((root/'run/collection.json').read_text())
            self.assertEqual(manifest['metric_version'], 2)
            self.assertEqual(manifest['affinity'], [int(env['NPBENCH_CPUSET'])])


if __name__ == '__main__':
    unittest.main()
