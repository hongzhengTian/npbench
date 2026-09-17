"""Independent small numerical checks, not performance measurements."""
import hashlib
import importlib
import json
from pathlib import Path
import unittest
import numpy as np
import numba as nb

ROOT = Path(__file__).resolve().parents[1]


def adi_dense(tsteps, u):
    """Solve each canonical ADI sweep as a dense linear system, no Thomas recurrence."""
    n = len(u)
    mul1, mul2 = 2. * n * n / tsteps, 1. * n * n / tsteps
    a, b, c = -mul1 / 2., 1. + mul1, -mul1 / 2.
    d, e, f = -mul2 / 2., 1. + mul2, -mul2 / 2.
    first = np.diag(np.full(n-2, b)) + np.diag(np.full(n-3, a), -1) + np.diag(np.full(n-3, c), 1)
    second = np.diag(np.full(n-2, e)) + np.diag(np.full(n-3, d), -1) + np.diag(np.full(n-3, f), 1)
    for _ in range(tsteps):
        v = np.ones_like(u)
        for i in range(1, n-1):
            rhs = -d*u[1:-1,i-1] + (1.+2.*d)*u[1:-1,i] - f*u[1:-1,i+1]
            rhs[0] -= a; rhs[-1] -= c
            v[1:-1,i] = np.linalg.solve(first, rhs)
        u[1:-1,0] = 1.; u[1:-1,-1] = 1.
        for i in range(1,n-1):
            rhs = -a*v[i-1,1:-1] + (1.+2.*a)*v[i,1:-1] - c*v[i+1,1:-1]
            rhs[0] -= d; rhs[-1] -= f
            u[i,1:-1] = np.linalg.solve(second,rhs)
    return u


class WorkloadRepairsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = nb.get_num_threads()
        nb.set_num_threads(min(cls.threads, 4))  # Unit check only; collection uses system defaults.

    @classmethod
    def tearDownClass(cls):
        nb.set_num_threads(cls.threads)

    def test_optional_variants_do_not_change_default_campaign(self):
        import os
        code = (ROOT/'run_large.sh').read_text().split('available = ',1)[1].split("assert len(benchmarks)",1)[0]
        code = 'available = '+code
        names={'adi_corrected','correlation_return','mlp_rowmax','azimint_hist_private'}
        from unittest.mock import patch
        with patch.dict(os.environ, {}, clear=True):
            namespace={'repo':ROOT,'json':json,'os':os}
            exec(code, namespace)
            self.assertTrue(names.isdisjoint(namespace['benchmarks']))
            self.assertIn('adi', namespace['benchmarks'])
        with patch.dict(os.environ, {'NPBENCH_CASES_FILE':'exact.tsv'}, clear=True):
            namespace={'repo':ROOT,'json':json,'os':os}
            exec(code, namespace)
            self.assertTrue(names <= set(namespace['benchmarks']))

    def test_histogram_entry_has_no_process_local_dynamic_globals(self):
        from npbench.benchmarks.azimint_hist.azimint_hist_private_numba_npr import azimint_hist
        from npbench.benchmarks.azimint_hist.azimint_hist_numpy import azimint_hist as reference
        rng = np.random.default_rng(41)
        data, radius = rng.random(2049), rng.random(2049)
        result = azimint_hist(data, radius, 5)
        np.testing.assert_allclose(result, reference(data, radius, 5), rtol=1e-10, atol=1e-10)
        self.assertTrue(azimint_hist.overloads)
        self.assertFalse(any(cres.library.has_dynamic_globals for cres in azimint_hist.overloads.values()))

    def test_original_files_unchanged(self):
        for name, expected in json.loads((ROOT/'scripts/workload-repairs-originals.json').read_text()).items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(), expected, name)

    def test_adi_against_independent_dense_solves(self):
        from npbench.benchmarks.polybench.adi.adi import initialize
        from npbench.benchmarks.polybench.adi.adi_numpy import kernel as original
        data = initialize(7)
        expected = adi_dense(3, data.copy())
        self.assertFalse(np.allclose(original(3, 7, data.copy()), expected))
        for suffix in ('numpy','numba_n','numba_np'):
            kernel = importlib.import_module('npbench.benchmarks.polybench.adi.adi_corrected_'+suffix).kernel
            actual = data.copy(); result = kernel(3, 7, actual)
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
            np.testing.assert_array_equal(result, actual)

    def test_softmax_row_shifts_and_large_logits(self):
        from npbench.benchmarks.deep_learning.mlp.mlp_numpy import softmax as reference
        data = np.array([[1000,1001,-1000],[-1000,-1001,-1002],[0,3,-2],[900,898,-900]], dtype=np.float32)
        for suffix in ('n','np','npr'):
            impl = importlib.import_module('npbench.benchmarks.deep_learning.mlp.mlp_rowmax_numba_'+suffix).softmax
            actual = impl(data.copy())
            self.assertEqual(actual.dtype, data.dtype)
            self.assertTrue(np.isfinite(actual).all())
            np.testing.assert_allclose(actual, reference(data), rtol=1e-5, atol=1e-7)
            np.testing.assert_allclose(actual.sum(axis=1), 1., rtol=1e-6)
            np.testing.assert_allclose(impl(data + np.array([[100],[-30],[20],[40]],dtype=data.dtype)), actual, rtol=1e-5, atol=1e-7)

    def test_histogram_contended_buckets_and_uneven_partitions(self):
        from npbench.benchmarks.azimint_hist.azimint_hist_private_numba_npr import histogram_prange
        rng=np.random.default_rng(7)
        for size in (3, 100003):
            values=np.resize(np.array([0., .1, .2, 1.]),size)
            weights=rng.random(size)
            expected, edges=np.histogram(values,bins=5,weights=weights)
            for threads in (1, min(4,self.threads)):
                nb.set_num_threads(threads)
                for _ in range(3):
                    actual, actual_edges=histogram_prange(values,5,weights)
                    np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-10)
                    np.testing.assert_allclose(actual_edges,edges,rtol=1e-14)

    def test_reference_and_initializer_identity_for_unchanged_algorithms(self):
        from npbench.infrastructure import Benchmark
        for alias, base in [('correlation_return','correlation'),('mlp_rowmax','mlp'),('azimint_hist_private','azimint_hist')]:
            info=Benchmark(alias).info
            original_info=Benchmark(base).info
            for field in ('parameters','input_args','array_args','output_args','init','relative_path','func_name'):
                self.assertEqual(info[field], original_info[field], (alias, field))
            prefix='npbench.benchmarks.'+info['relative_path'].replace('/','.')+'.'
            self.assertIs(importlib.import_module(prefix+alias).initialize, importlib.import_module(prefix+base).initialize)
            self.assertIs(getattr(importlib.import_module(prefix+alias+'_numpy'),info['func_name']), getattr(importlib.import_module(prefix+base+'_numpy'),info['func_name']))


if __name__ == '__main__':
    unittest.main()
