"""No benchmark execution: ABI ordering, metadata and validation contracts."""
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
import numpy as np

from npbench.infrastructure import golden
from npbench.infrastructure.dace_restore import normalize_nested_zero_offsets, load_artifact
from npbench.infrastructure.region import Region


class BaselineFixTest(unittest.TestCase):
    def test_exact_case_plan_and_resume_without_running_benchmarks(self):
        root = Path(__file__).resolve().parents[1]
        # Run only the collection's metadata freezer, never the shell loop,
        # golden producer, lifecycle worker or new diagnostic script.
        freeze = (root / 'run_large.sh').read_text().split("<<'PY'\n", 1)[1].split('\nPY\n', 1)[0]
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            cases = folder / 'cases.tsv'
            cases.write_text('gemm\tnumpy\tdefault\ncorrelation\tnumba\tnopython-mode\n')
            out = folder / 'collection'; out.mkdir()
            env = {key: value for key, value in os.environ.items() if not key.startswith('NPBENCH_')}
            env.update(NPBENCH_CASES_FILE=str(cases), NPBENCH_GOLDEN_CACHE=str(folder / 'goldens'),
                       NPBENCH_REPEATS='1', NPBENCH_FRESH_PROCESSES='0', NPBENCH_TIMEOUT='2',
                       NPBENCH_GOLDEN_TIMEOUT='2', NPBENCH_PRESET='S',
                       PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE='1')
            command = [sys.executable, '-', str(root), str(out), 'baselines']
            first = subprocess.run(command, input=freeze, text=True, env=env, capture_output=True, timeout=60)
            self.assertEqual(first.returncode, 0, first.stderr)
            plan = (out / 'plan.tsv').read_text().splitlines()
            self.assertEqual([line for line in plan if line.startswith('baselines\t')], [
                'baselines\tcorrelation\tnumba\tnopython-mode\tpresent',
                'baselines\tgemm\tnumpy\tdefault\tpresent'])
            again = subprocess.run(command, input=freeze, text=True, env=env, capture_output=True, timeout=60)
            self.assertEqual(again.returncode, 0, again.stderr)
            self.assertFalse((folder / 'goldens').exists())
            self.assertFalse(list(out.rglob('process-*.jsonl')))
            cases.write_text('gemm\tnumpy\tdefault\n')
            changed = subprocess.run(command, input=freeze, text=True, env=env, capture_output=True, timeout=60)
            self.assertNotEqual(changed.returncode, 0)
            self.assertIn('Code/environment/selection changed', changed.stderr)

    def test_synchronizes_before_any_host_read(self):
        ready = []
        framework = Mock()
        framework.load_implementation.return_value = lambda x: ready.append('execute') or np.ones(2)
        framework.imports.return_value = {}
        framework.exec_str.return_value = '__npb_result = __npb_impl(x)'
        framework.args.return_value = ['x']
        framework.uses_device_arrays.return_value = False
        framework.synchronize.side_effect = lambda: ready.append('sync')
        framework.normalize_returns.side_effect = lambda values, _: values
        bench = types.SimpleNamespace(info={'input_args': ['x'], 'array_args': ['x']})
        region = Region(bench, framework, 'default', [])
        with patch('npbench.infrastructure.region.to_host', side_effect=lambda x: ready.append('read') or x):
            region({'x': np.ones(2)})
        self.assertEqual(ready, ['execute', 'sync', 'read'])

    def test_dace_sync_uses_device_not_cupy_current_stream(self):
        from npbench.infrastructure.dace_framework import DaceFramework
        framework = object.__new__(DaceFramework)
        framework.info = {'arch': 'gpu'}
        fake = types.SimpleNamespace(cuda=types.SimpleNamespace(runtime=types.SimpleNamespace(deviceSynchronize=Mock())))
        with patch.dict('sys.modules', {'cupy': fake}):
            framework.synchronize()
        fake.cuda.runtime.deviceSynchronize.assert_called_once_with()

    def test_zero_offset_normalization_is_nested_and_layout_neutral(self):
        descriptor = {'type': 'Array', 'attributes': {'shape': ['N'], 'strides': ['1'], 'offset': ['0', '0']}}
        original = {'type': 'SDFG', 'attributes': {'_arrays': {'top': copy.deepcopy(descriptor)}},
                    'nodes': [{'type': 'SDFG', 'attributes': {'_arrays': {'inner': descriptor}}}]}
        saved = copy.deepcopy(original)
        fixed, changes = normalize_nested_zero_offsets(original)
        self.assertEqual(original, saved)
        self.assertEqual(fixed['attributes'], saved['attributes'])
        self.assertEqual(len(changes), 1)
        self.assertEqual(fixed['nodes'][0]['attributes']['_arrays']['inner']['attributes']['offset'], ['0'])
        for offset, strides in [(['1', '0'], ['1']), (['0', '0'], ['N', '1'])]:
            bad = copy.deepcopy(original)
            bad['nodes'][0]['attributes']['_arrays']['inner']['attributes'].update(offset=offset, strides=strides)
            self.assertEqual(normalize_nested_zero_offsets(bad)[1], [])

    def test_restoration_normalizes_only_metadata_without_compiler_or_disk_rewrite(self):
        import dace
        descriptor = {'type': 'Array', 'attributes': {'shape': ['N'], 'strides': ['1'], 'offset': ['0', '0']}}
        doc = {'type': 'SDFG', 'nodes': [{'type': 'SDFG', 'attributes': {'_arrays': {'x': descriptor}}}]}
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); path = folder / 'program.sdfg'; path.write_text(json.dumps(doc))
            before = path.read_bytes()
            with patch('dace.sdfg.utils.load_precompiled_sdfg', side_effect=TypeError('Offset must be the same size as shape')), \
                    patch.object(dace.SDFG, 'from_json', return_value=types.SimpleNamespace(name='test')), \
                    patch.object(dace.SDFG, 'compile', side_effect=AssertionError('compiled')), \
                    patch('dace.codegen.compiled_sdfg.CompiledSDFG') as compiled, \
                    patch('dace.codegen.compiled_sdfg.ReloadableDLL'):
                self.assertIs(load_artifact(folder, audit_path=folder/'audit.json'), compiled.return_value)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(json.loads((folder/'audit.json').read_text())['adapter'], 'nested_zero_offset_rank')

    def test_thread_cpu_activity_tracks_work_and_reports_exited_threads(self):
        from npbench.infrastructure.resources import thread_cpu_activity
        before = {'wall': 1., 'ticks_per_second': 100, 'threads': {
            '1': {'start': 10, 'ticks': 2}, '2': {'start': 10, 'ticks': 7}}}
        after = {'wall': 2., 'ticks_per_second': 100, 'threads': {
            '1': {'start': 10, 'ticks': 5}, '3': {'start': 20, 'ticks': 4}}}
        result = thread_cpu_activity(before, after)
        self.assertEqual(result['cpu_seconds_by_thread'], {'1': .03, '3': .04})
        self.assertEqual(result['threads_with_observed_cpu_work'], 2)
        self.assertEqual(result['exited_thread_ids'], ['2'])
        self.assertEqual(result['cpu_tick_seconds'], .01)
        self.assertIn('probe_error', thread_cpu_activity({'probe_error': 'unavailable'}, after))

    def test_validation_audit_does_not_relax_region_or_truncate_outputs(self):
        bench = types.SimpleNamespace(info={'output_args': []})
        expected = {'returns': [np.ones(2)], 'arrays': {'scratch': np.ones(2)}}
        actual = {'returns': [np.ones(2)], 'arrays': {'scratch': np.zeros(2)}}
        failures = []
        self.assertFalse(golden.validate(bench, expected, actual, diagnostics=failures))
        audit = golden.validation_audit(bench, expected, actual, failures)
        self.assertFalse(audit['strict_region_pass'])
        self.assertTrue(audit['declared_outputs_strict_pass'])
        self.assertEqual(audit['native_numeric_projection'], 'passed')
        actual['returns'] = []
        audit = golden.validation_audit(bench, expected, actual, [{'field': 'returns', 'reason': 'count'}])
        self.assertEqual(audit['native_numeric_projection'], 'incomparable_output_count')


if __name__ == '__main__':
    unittest.main()
