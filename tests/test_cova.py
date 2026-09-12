"""Test the NPBench CoVA protocol without compiling or requiring CoVA."""
import ast
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from npbench.infrastructure import Benchmark, Framework, generate_framework

ROOT = Path(__file__).resolve().parents[1]
FRAMEWORKS = sorted(path.stem for path in (ROOT / 'framework_info').glob('cova_*.json'))


class CovaProtocolTest(unittest.TestCase):
    def test_workload_results_and_mutations_match_numpy(self):
        directories = sorted((ROOT / 'npbench/benchmarks/polybench').iterdir())
        for directory in directories:
            if not list(directory.glob('*_cova_*.py')):
                continue
            bench = Benchmark(directory.name)
            # Small unit fixtures only; the public S/M/L/paper presets are unchanged.
            bench.info['parameters']['unit'] = {
                key: 3 if key in ('TSTEPS', 'TMAX') else 8
                for key in bench.info['parameters']['S']}
            data = bench.get_data('unit')
            reference, _ = generate_framework('numpy').implementations(bench)[0]
            for name in FRAMEWORKS:
                with self.subTest(benchmark=bench.bname, framework=name):
                    framework = generate_framework(name)
                    (source, label), = framework.impl_files(bench)
                    tree = ast.parse(source.read_text())
                    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
                    self.assertEqual([node.name for node in functions], ['kernel'])
                    decorator = functions[0].decorator_list[0]
                    options = {kw.arg: ast.literal_eval(kw.value) for kw in decorator.keywords}
                    self.assertEqual(options['backend'], framework.info['arch'])
                    self.assertEqual(options['tool'], name.removeprefix('cova_').replace('_', '-'))
                    # Execute the source algorithm as Python to check transplantation
                    # and the adapter independently of compiler support.
                    functions[0].decorator_list = []
                    tree.body = [node for node in tree.body
                                 if not (isinstance(node, ast.ImportFrom) and node.module == 'cova')]
                    namespace = {}
                    exec(compile(tree, str(source), 'exec'), namespace)
                    with patch.object(Framework, 'implementations', return_value=[(namespace['kernel'], label)]):
                        implementation, _ = framework.implementations(bench)[0]
                    expected_args = [np.copy(data[arg]) if arg in bench.info['array_args'] else data[arg]
                                     for arg in bench.info['input_args']]
                    actual_args = [np.copy(arg) if isinstance(arg, np.ndarray) else arg for arg in expected_args]
                    expected = reference(*expected_args)
                    actual = implementation(*actual_args)
                    self.assert_result(expected, actual)
                    for arg, x, y in zip(bench.info['input_args'], expected_args, actual_args):
                        if arg in bench.info['array_args']:
                            self.assert_result(x, y)

    def assert_result(self, expected, actual):
        if expected is None:
            self.assertIsNone(actual)
        elif isinstance(expected, (tuple, list)):
            self.assertIsInstance(actual, (tuple, list))
            self.assertEqual(len(expected), len(actual))
            for x, y in zip(expected, actual):
                self.assert_result(x, y)
        else:
            self.assertEqual(np.shape(expected), np.shape(actual))
            self.assertEqual(np.asarray(expected).dtype, np.asarray(actual).dtype)
            self.assertTrue(np.all(np.isfinite(expected)))
            self.assertTrue(np.all(np.isfinite(actual)))
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_returned_array_is_written_back_to_original_input(self):
        bench = Benchmark('gemm')
        framework = generate_framework('cova_llvm_cpu_serial')
        expected = np.ones((2, 3))
        with patch.object(Framework, 'implementations', return_value=[(lambda *args: expected, 'default')]):
            impl, _ = framework.implementations(bench)[0]
        target = np.zeros_like(expected)
        self.assertIsNone(impl(1.0, 0.0, target, None, None))
        np.testing.assert_array_equal(target, expected)

    def test_missing_or_invalid_writeback_is_rejected(self):
        bench = Benchmark('gemm')
        framework = generate_framework('cova_llvm_cpu_serial')
        for result in (None, (np.ones((2, 3)), np.ones((2, 3))), np.ones((3, 2)), np.ones((2, 3), dtype=np.float32)):
            with self.subTest(result_type=type(result).__name__):
                with patch.object(Framework, 'implementations', return_value=[(lambda *args: result, 'default')]):
                    impl, _ = framework.implementations(bench)[0]
                target = np.zeros((2, 3))
                with self.assertRaises(ValueError):
                    impl(1.0, 0.0, target, None, None)
                np.testing.assert_array_equal(target, 0.0)

    def test_source_revision_can_be_recorded_without_package_metadata(self):
        with patch.dict('os.environ', {'NPBENCH_COVA_VERSION': 'test-revision'}):
            self.assertEqual(generate_framework('cova_llvm_cpu').version(), 'test-revision')


if __name__ == '__main__':
    unittest.main()
