"""Corrected ADI CoVA entries against the existing independent dense oracle."""
from contextlib import ExitStack
import ast
import importlib
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
from test_workload_repairs import adi_dense
from npbench.benchmarks.polybench.adi.adi import initialize
from npbench.benchmarks.polybench.adi.adi_corrected_cova_kernels import kernel as pure

root = Path(__file__).resolve().parents[1]
source = root / 'npbench/benchmarks/polybench/adi'
reference = ast.parse((source / 'adi_corrected_numpy.py').read_text())
candidate = ast.parse((source / 'adi_corrected_cova_kernels.py').read_text())
returns = [node for node in ast.walk(candidate) if isinstance(node, ast.Return)]
assert len(returns) == 1
assert isinstance(returns[0].value, ast.Tuple)
first, second = returns[0].value.elts
assert ast.dump(first) == ast.dump(second)
returns[0].value = first
assert ast.dump(candidate) == ast.dump(reference)
info = json.loads((root / 'bench_info/adi_corrected.json').read_text())['benchmark']
assert info['optional'] and info['repair']['changes_reference']
assert info.get('golden_reference') is None
assert info['cova'] == {'return_count': 1, 'writeback_args': ['u']}
route = next((x for x in sys.argv[1:] if x != 'restore'), None)
function = pure
if route:
    function = importlib.import_module(
        'npbench.benchmarks.polybench.adi.adi_corrected_cova_' + route.replace('-', '_')).kernel
with ExitStack() as stack:
    if 'restore' in sys.argv:
        from cova.orchestration.compilation import invocation
        from cova.runtime.native.extension import NativeExtensionRuntime
        stack.enter_context(patch.object(invocation, 'load_or_generate_cova_module',
                                        side_effect=AssertionError('unexpected lowering')))
        stack.enter_context(patch.object(NativeExtensionRuntime, 'compile',
                                        side_effect=AssertionError('unexpected compilation')))
    for n, steps in ((3, 1), (7, 3), (11, 2)):
        a = initialize(n)
        expected = adi_dense(steps, a.copy())
        result, writeback = function(steps, n, a)
        for actual in (result, writeback, a):
            assert actual.shape == expected.shape and actual.dtype == expected.dtype
            assert np.isfinite(actual).all()
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
print('corrected ADI independent dense oracle and writeback passed', route or 'Python')
