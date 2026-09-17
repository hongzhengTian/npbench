"""Discovery and representative adaptation checks, without compiling CoVA."""
import ast
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from npbench.infrastructure import Benchmark, Framework, generate_framework
from npbench.infrastructure.golden import clone_data, outputs, validate


ROOT = Path(__file__).resolve().parents[1]
FRAMEWORKS = sorted(p.stem for p in (ROOT / 'framework_info').glob('cova_*.json'))


class CovaCatalogTest(unittest.TestCase):
    def test_every_registered_benchmark_has_six_discoverable_routes(self):
        self.assertEqual(len(FRAMEWORKS), 6)
        for metadata in sorted((ROOT / 'bench_info').glob('*.json')):
            bench = Benchmark(metadata.stem)
            if bench.info.get('optional') and 'repair' in bench.info:
                # Explicit baseline-only repair registrations do not claim CoVA support.
                self.assertNotIn('cova', bench.info)
                for name in FRAMEWORKS:
                    self.assertFalse(generate_framework(name).impl_files(bench)[0][0].is_file())
                continue
            contract = bench.info['cova']
            self.assertIsInstance(contract['return_count'], int)
            self.assertGreaterEqual(contract['return_count'], 0)
            self.assertEqual(len(contract['writeback_args']), len(set(contract['writeback_args'])))
            self.assertTrue(set(contract['writeback_args']) <= set(bench.info['array_args']))
            for name in FRAMEWORKS:
                with self.subTest(benchmark=bench.bname, framework=name):
                    framework = generate_framework(name)
                    (source, label), = framework.impl_files(bench)
                    self.assertEqual(framework.implementation_names(bench), [label])
                    tree = ast.parse(source.read_text())
                    entry = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                                 and n.name == bench.info['func_name'])
                    self.assertEqual(len(entry.decorator_list), 1)
                    decorator = entry.decorator_list[0]
                    self.assertIsInstance(decorator.func, ast.Name)
                    self.assertEqual(decorator.func.id, 'cova')
                    options = {kw.arg: ast.literal_eval(kw.value) for kw in decorator.keywords}
                    self.assertEqual(options, {
                        'backend': framework.info['arch'],
                        'tool': name.removeprefix('cova_').replace('_', '-')})
                    parameters = [arg.arg for arg in entry.args.args]
                    reference_file, _ = generate_framework('numpy').impl_files(bench)[0]
                    reference_entry = next(n for n in ast.parse(reference_file.read_text()).body
                                           if isinstance(n, ast.FunctionDef) and n.name == entry.name)
                    # NPBench calls positionally; Mandelbrot uses different
                    # parameter names in metadata and the upstream function.
                    self.assertEqual(parameters, [arg.arg for arg in reference_entry.args.args])
                    self.assertLessEqual(len(parameters) - len(entry.args.defaults), len(bench.info['input_args']))
                    self.assertFalse(any(isinstance(n, ast.ImportFrom) and
                                         (n.module or '').startswith('benchmarks') for n in ast.walk(tree)))

    def test_representative_returns_helpers_and_writebacks_match_reference(self):
        # Private unit inputs only; upstream S/M/L/paper definitions stay intact.
        cases = {
            'compute': {'M': 4, 'N': 5},
            'go_fast': {'N': 8},
            'crc16': {'N': 8},
            'hdiff': {'I': 4, 'J': 5, 'K': 3},
            'mlp': {'C_in': 3, 'N': 2, 'S0': 8, 'S1': 6, 'S2': 4},
            'nbody': {'N': 5, 'tEnd': 0.1},
            'channel_flow': {'ny': 5, 'nx': 5, 'nit': 2},
            'contour_integral': {'NR': 3, 'NM': 2, 'num_int_pts': 3},
            'scattering_self_energies': {'Nkz': 1, 'NE': 2, 'Nqz': 1, 'Nw': 1,
                                         'N3D': 2, 'NA': 3, 'NB': 1, 'Norb': 2},
        }
        for name, parameters in cases.items():
            bench = Benchmark(name)
            bench.info['parameters']['unit'] = dict(bench.info['parameters']['S'], **parameters)
            data = bench.get_data('unit')
            reference, _ = generate_framework('numpy').implementations(bench)[0]
            expected_data = clone_data(data)
            expected = {
                'returns': outputs(reference(*(expected_data[a] for a in bench.info['input_args']))),
                'arrays': {a: expected_data[a] for a in bench.info['array_args']},
            }
            for route in FRAMEWORKS:
                with self.subTest(benchmark=name, framework=route):
                    framework = generate_framework(route)
                    (source, label), = framework.impl_files(bench)
                    tree = ast.parse(source.read_text())
                    entry = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                                 and n.name == bench.info['func_name'])
                    entry.decorator_list = []
                    tree.body = [n for n in tree.body if not
                                 (isinstance(n, ast.ImportFrom) and n.module == 'cova')]
                    namespace = {}
                    exec(compile(tree, str(source), 'exec'), namespace)
                    with patch.object(Framework, 'implementations', return_value=[(namespace[entry.name], label)]):
                        implementation, _ = framework.implementations(bench)[0]
                    actual_data = clone_data(data)
                    actual = {
                        'returns': outputs(implementation(*(actual_data[a] for a in bench.info['input_args']))),
                        'arrays': {a: actual_data[a] for a in bench.info['array_args']},
                    }
                    self.assertTrue(validate(bench, expected, actual))
                    # Confirm the return protocol covers mutations even when
                    # the original output_args list omits them.
                    mutated = {a for a in bench.info['array_args']
                               if not np.array_equal(data[a], expected_data[a])}
                    self.assertTrue(mutated <= set(bench.info['cova']['writeback_args']))


if __name__ == '__main__':
    unittest.main()
