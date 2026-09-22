"""Helper equivalence; optional native routes also check compiler-forbidden reuse."""
import argparse
import json
from pathlib import Path
import sys
from contextlib import ExitStack
from unittest.mock import patch
import unittest
import numpy as np

from npbench.benchmarks.mandelbrot2.mandelbrot2_cova_helper_kernels import mandelbrot
from npbench.benchmarks.mandelbrot2.mandelbrot2_numpy import mandelbrot as mandelbrot_reference

from npbench.benchmarks.azimint_hist.azimint_hist_cova_helper_kernels import histogram_counts_weights
from npbench.benchmarks.stockham_fft.stockham_fft_cova_helper_kernels import stockham_fft
from npbench.benchmarks.stockham_fft.stockham_fft_numpy import stockham_fft as fft_reference

TOOL = sys.argv.pop(1) if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else None
RESTORE = len(sys.argv) > 1 and sys.argv[1] == 'restore'
if RESTORE:
    sys.argv.pop(1)


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--histogram-case", type=int, choices=range(7))
options, remaining = parser.parse_known_args(sys.argv[1:])
sys.argv[1:] = remaining


def compiled(function):
    if TOOL is None:
        return function
    from cova import cova
    return cova(backend='gpu' if 'gpu' in TOOL else 'cpu', tool=TOOL)(function)


class CoVAHelperTests(unittest.TestCase):
    def test_histogram_edges_counts_weights(self):
        function = compiled(histogram_counts_weights)
        edges = np.linspace(-1., 1., 8)
        boundary = np.concatenate([edges, np.nextafter(edges[1:-1], -np.inf),
                                   np.nextafter(edges[1:-1], np.inf)])
        cases = [(boundary, 7), (np.full(11, 2.5), 7), (np.empty(0), 5),
                 (np.array([.125]), 1), (np.resize(boundary, 65539), 7),
                 (1.e9 + boundary, 7), (np.resize(boundary, 1048591), 7)]
        selected = cases if options.histogram_case is None else [cases[options.histogram_case]]
        for values, bins in selected:
            with self.subTest(size=values.size, bins=bins):
                weights = np.random.default_rng(17).normal(size=values.size)
                original_values, original_weights = values.copy(), weights.copy()
                counts, sums, actual_edges = function(values, bins, weights)
                expected_counts, expected_edges = np.histogram(values, bins)
                expected_sums, _ = np.histogram(values, bins, weights=weights)
                np.testing.assert_array_equal(counts, expected_counts)
                np.testing.assert_allclose(sums, expected_sums, rtol=1e-12, atol=1e-12)
                np.testing.assert_array_equal(actual_edges, expected_edges)
                np.testing.assert_array_equal(values, original_values)
                np.testing.assert_array_equal(weights, original_weights)
                self.assertEqual(counts.dtype, np.dtype('int64'))
                self.assertEqual(sums.dtype, weights.dtype)
                self.assertEqual(actual_edges.dtype, values.dtype)

    def test_stockham_algorithm_and_input_writeback(self):
        function = compiled(stockham_fft)
        for radix, stages in [(2, 0), (2, 1), (2, 4), (3, 3), (4, 3)]:
            with self.subTest(radix=radix, stages=stages):
                size = radix ** stages
                rng = np.random.default_rng(23)
                values = rng.random(size) + 1j * rng.random(size)
                original = values.copy()
                actual = np.full(size, 2 + 3j)
                expected = actual.copy()
                fft_reference(size, radix, stages, values, expected)
                returned = function(size, radix, stages, values, actual)
                np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(actual, np.fft.fft(values), rtol=1e-12, atol=1e-12)
                np.testing.assert_array_equal(returned, actual)
                np.testing.assert_array_equal(values, original)
                self.assertEqual(actual.dtype, np.dtype('complex128'))

    def test_mandelbrot_shrinking_arrays(self):
        function = compiled(mandelbrot)
        for bounds, size, iterations in [((-2., .5, -1.25, 1.25), 8, 10),
                                         ((3., 4., 3., 4.), 4, 5),
                                         ((-.1, .1, -.1, .1), 4, 5),
                                         ((-2., .5, -1.25, 1.25), 4, 0)]:
            with self.subTest(bounds=bounds, size=size, iterations=iterations):
                args = (*bounds, size, size, iterations, 2.)
                actual = function(*args)
                expected = mandelbrot_reference(*args)
                self.assertEqual(len(actual), 2)
                for a, b in zip(actual, expected):
                    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
                    self.assertEqual(a.dtype, b.dtype)
                    self.assertEqual(a.shape, b.shape)

    def test_optional_registration_reuses_canonical_reference(self):
        root = Path(__file__).resolve().parents[1]
        for original in ('azimint_hist', 'stockham_fft', 'mandelbrot2'):
            canonical = json.loads((root / 'bench_info' / (original + '.json')).read_text())['benchmark']
            variant = json.loads((root / 'bench_info' / (original + '_cova_helper.json')).read_text())['benchmark']
            self.assertTrue(variant['optional'])
            self.assertEqual(variant['golden_reference'], original)
            for key in ('parameters', 'input_args', 'array_args', 'output_args', 'init', 'cova'):
                self.assertEqual(variant.get(key), canonical.get(key))


if __name__ == '__main__':
    with ExitStack() as stack:
        if RESTORE:
            from cova.orchestration.compilation import invocation
            from cova.runtime.native.extension import NativeExtensionRuntime
            for owner, name in ((invocation, 'load_or_generate_cova_module'), (NativeExtensionRuntime, 'compile')):
                stack.enter_context(patch.object(owner, name, side_effect=AssertionError('restore must not compile')))
        unittest.main()
