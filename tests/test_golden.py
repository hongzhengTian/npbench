import json
import copy
import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from npbench.infrastructure import Benchmark, generate_framework
from npbench.infrastructure import golden


class GoldenTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.bench = Benchmark('gemm')
        self.bench.info['parameters']['unit'] = {'NI': 4, 'NJ': 3, 'NK': 2}
        self.numpy = generate_framework('numpy')

    def get(self, **options):
        return golden.load_or_create(self.bench, 'unit', self.numpy, self.tmp.name, **options)

    def alias(self):
        alias = copy.copy(self.bench)
        alias.bname = 'explicit_variant'
        alias.info = copy.deepcopy(self.bench.info)
        alias.info['module_name'] = 'variant_without_a_reference_module'
        alias.info['golden_reference'] = self.bench.bname
        return alias

    def test_explicit_alias_reuses_exact_bundle_without_initializer(self):
        bundle, first = self.get()
        alias = self.alias()
        with patch.object(importlib.import_module('npbench.infrastructure.benchmark'), 'Benchmark', return_value=self.bench), \
                patch.object(alias, 'get_data', side_effect=AssertionError('alias initialized')), \
                patch.object(self.numpy, 'implementations', side_effect=AssertionError('producer executed')):
            again, event = golden.load_or_create(alias, 'unit', self.numpy, self.tmp.name, required=True)
            self.assertEqual(golden.identity(alias, 'unit', self.numpy), golden.identity(self.bench, 'unit', self.numpy))
        self.assertEqual(event['key'], first['key'])
        self.assertEqual(event['sha256'], first['sha256'])
        self.assertEqual(event['producer_executions'], 0)
        for name in bundle['inputs']:
            np.testing.assert_array_equal(bundle['inputs'][name], again['inputs'][name])

    def test_explicit_alias_rejects_different_numerical_contract(self):
        for field in ('parameters', 'input_args', 'array_args', 'output_args', 'func_name', 'init'):
            alias = self.alias()
            if field == 'parameters': alias.info[field]['unit']['NI'] += 1
            else: alias.info[field] = None
            with self.subTest(field=field), patch.object(importlib.import_module('npbench.infrastructure.benchmark'), 'Benchmark', return_value=self.bench):
                with self.assertRaisesRegex(ValueError, 'contract differs'):
                    golden.identity(alias, 'unit', self.numpy)

    def test_explicit_alias_rejects_parameter_type_change(self):
        alias = self.alias()
        alias.info['parameters']['unit']['NI'] = float(alias.info['parameters']['unit']['NI'])
        with patch.object(importlib.import_module('npbench.infrastructure.benchmark'), 'Benchmark', return_value=self.bench):
            with self.assertRaisesRegex(ValueError, 'parameters'):
                golden.identity(alias, 'unit', self.numpy)

    def test_explicit_alias_rejects_reference_chains(self):
        alias = self.alias()
        with patch.object(importlib.import_module('npbench.infrastructure.benchmark'), 'Benchmark', return_value=alias):
            with self.assertRaisesRegex(ValueError, 'directly'):
                golden.identity(alias, 'unit', self.numpy)

    def test_explicit_alias_required_miss_does_not_run_producer(self):
        alias = self.alias()
        with patch.object(importlib.import_module('npbench.infrastructure.benchmark'), 'Benchmark', return_value=self.bench), \
                patch.object(self.numpy, 'implementations', side_effect=AssertionError('producer')):
            with self.assertRaises(FileNotFoundError):
                golden.load_or_create(alias, 'unit', self.numpy, self.tmp.name, required=True)

    def test_hit_needs_no_initializer_or_reference(self):
        bundle, first = self.get()
        with patch.object(self.bench, 'get_data', side_effect=AssertionError('initializer executed')), \
                patch.object(self.numpy, 'implementations', side_effect=AssertionError('reference imported')):
            again, hit = self.get(required=True)
        self.assertEqual(first['status'], 'created')
        self.assertEqual(hit['producer_executions'], 0)
        for name in bundle['inputs']:
            np.testing.assert_array_equal(bundle['inputs'][name], again['inputs'][name])
        self.assertTrue(golden.validate(self.bench, bundle, again))

    def test_metadata_only_hit_checks_integrity_without_loading_arrays(self):
        _, first = self.get()
        with patch.object(golden.np, 'load', side_effect=AssertionError('arrays loaded')):
            bundle, hit = self.get(required=True, load_values=False)
        self.assertIsNone(bundle)
        self.assertEqual(hit['key'], first['key'])
        self.assertEqual(hit['producer_executions'], 0)
        (Path(hit['path']) / 'values.npz').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'integrity'):
            self.get(required=True, load_values=False)

    def test_policy_and_competitor_metadata_do_not_regenerate_values(self):
        bundle, event = self.get()
        self.bench.info['rtol'] = 0
        self.bench.info['cova']['return_count'] = 123
        self.bench.info['parameters']['L']['NI'] += 1
        with patch.object(self.numpy, 'implementations', side_effect=AssertionError('producer')):
            _, hit = self.get(required=True)
        self.assertEqual(event['key'], hit['key'])
        self.bench.info['parameters']['unit']['NI'] += 1
        with self.assertRaises(FileNotFoundError):
            self.get(required=True)

    def test_corruption_is_not_regenerated(self):
        _, event = self.get()
        (Path(event['path']) / 'values.npz').write_bytes(b'corrupt')
        with patch.object(self.numpy, 'implementations', side_effect=AssertionError('producer')):
            with self.assertRaisesRegex(ValueError, 'integrity'):
                self.get()

    def test_source_identity_tracks_local_reference_dependencies(self):
        root = Path(self.tmp.name)
        directory = root / 'npbench/benchmarks/demo'
        directory.mkdir(parents=True)
        source = directory / 'kernel.py'
        helper = directory / 'helper.py'
        source.write_text('from .helper import value\n')
        helper.write_text('value = 1\n')
        first = golden.source_hashes([source], root)
        helper.write_text('value = 2\n')
        self.assertNotEqual(first, golden.source_hashes([source], root))

    def test_complete_outputs_and_arrays_are_required(self):
        bundle, _ = self.get()
        actual = dict(bundle, returns=[np.zeros(1)])
        self.assertFalse(golden.validate(self.bench, bundle, actual))
        actual = dict(bundle, arrays={})
        self.assertFalse(golden.validate(self.bench, bundle, actual))
        actual = dict(bundle, arrays=golden.clone_data(bundle['arrays']))
        actual['arrays']['C'][0, 0] = np.nan
        self.assertFalse(golden.validate(self.bench, bundle, actual))

    def test_diagnostics_identify_structure_nonfinite_and_numerical_failures(self):
        bundle, _ = self.get()
        for reason, modify in [
            ('count', lambda a: a.update(returns=[np.zeros(1)])),
            ('keys', lambda a: a.update(arrays={})),
            ('shape', lambda a: a['arrays'].update(C=np.zeros(1))),
            ('dtype', lambda a: a['arrays'].update(C=a['arrays']['C'].astype(np.float32))),
            ('nonfinite', lambda a: a['arrays']['C'].fill(np.nan)),
            ('values', lambda a: a['arrays']['C'].fill(-1000)),
        ]:
            with self.subTest(reason=reason):
                actual = dict(bundle, arrays=golden.clone_data(bundle['arrays']))
                modify(actual)
                details = []
                self.assertFalse(golden.validate(self.bench, bundle, actual, diagnostics=details))
                self.assertIn(reason, {d['reason'] for d in details})
                json.dumps(details, allow_nan=False)
        details = []
        self.assertTrue(golden.validate(self.bench, bundle, bundle, diagnostics=details))
        self.assertEqual(details, [])

    def test_concurrent_requests_generate_once(self):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: self.get(), range(4)))
        self.assertEqual(sum(event['producer_executions'] for _, event in results), 1)
        self.assertEqual(len({event['key'] for _, event in results}), 1)

    def test_structure_corruption_is_rejected(self):
        _, event = self.get()
        path = Path(event['path']) / 'golden.json'
        metadata = json.loads(path.read_text())
        metadata['structure']['items']['returns'] = {'type': 'list', 'items': [{'type': 'value', 'value': 99}]}
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, 'structure integrity'):
            self.get()

    def test_missing_required_cache_does_not_execute(self):
        with patch.object(self.bench, 'get_data', side_effect=AssertionError('initializer')):
            with self.assertRaises(FileNotFoundError):
                self.get(required=True)


if __name__ == '__main__':
    unittest.main()
