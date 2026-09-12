"""Optional, framework-independent input and NumPy result bundles.

Only numerical source/contract changes invalidate values. Validation policy and
measured implementations are deliberately excluded. Bundles never use pickle.
"""
import ast
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np


SCHEMA = 1


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def source_hashes(paths, package_root):
    """Read local import dependencies without importing reference producers."""
    root = Path(package_root).resolve()
    pending = [Path(p).resolve() for p in paths]
    result = {}
    while pending:
        path = pending.pop()
        if not path.is_file() or not path.is_relative_to(root):
            continue
        name = str(path.relative_to(root))
        if name in result:
            continue
        result[name] = digest(path)
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = path.parent
                    for _ in range(node.level - 1):
                        base = base.parent
                    prefix = node.module or ''
                    for suffix in [prefix, *[prefix + '.' + a.name if prefix else a.name for a in node.names]]:
                        pending.append(base.joinpath(*suffix.split('.')).with_suffix('.py'))
                else:
                    modules = [node.module or '', *[(node.module + '.' + a.name) for a in node.names if node.module]]
            for module in modules:
                if module.startswith('npbench.'):
                    candidate = root.joinpath(*module.split('.'))
                    pending.extend([candidate.with_suffix('.py'), candidate / '__init__.py'])
    return dict(sorted(result.items()))


def identity(bench, preset, numpy):
    root = Path(__file__).resolve().parents[2]
    info = bench.info
    sources = [p for p, _ in numpy.impl_files(bench)]
    if info.get('init'):
        sources.append(root / 'npbench/benchmarks' / info['relative_path'] / (info['module_name'] + '.py'))
    contract = {key: info.get(key) for key in
                ('relative_path', 'module_name', 'func_name', 'input_args', 'array_args', 'init')}
    contract['parameters'] = info['parameters'][preset]
    return {'schema': SCHEMA, 'benchmark': bench.bname, 'preset': preset,
            'contract': contract, 'sources': source_hashes(sources, root)}


def outputs(value):
    return [] if value is None else list(value) if isinstance(value, (tuple, list)) else [value]


def clone_data(data):
    return {key: value.copy(order='K') if isinstance(value, np.ndarray) else value for key, value in data.items()}


def reference(bench, preset, numpy):
    raw = bench.get_data(preset)
    names = set(bench.info['parameters'][preset]) | set(bench.info['input_args'])
    data = clone_data({name: raw[name] for name in names})
    args = clone_data(data)
    impl, _ = numpy.implementations(bench)[0]
    result = outputs(impl(*(args[name] for name in bench.info['input_args'])))
    final_arrays = {name: args[name] for name in bench.info['array_args']}
    for value in [*result, *final_arrays.values()]:
        if not np.all(np.isfinite(value)):
            raise ValueError('NumPy reference produced nonfinite values')
    return {'inputs': data, 'returns': result, 'arrays': final_arrays}


def _pack(value, arrays):
    if isinstance(value, (np.ndarray, np.generic)):
        if value.dtype.hasobject:
            raise TypeError('Object arrays are not supported in golden bundles')
        key = 'a' + str(len(arrays))
        arrays[key] = np.asarray(value)
        return {'type': 'scalar' if isinstance(value, np.generic) else 'array', 'key': key}
    if isinstance(value, dict):
        return {'type': 'dict', 'items': {k: _pack(v, arrays) for k, v in value.items()}}
    if isinstance(value, (tuple, list)):
        return {'type': 'tuple' if isinstance(value, tuple) else 'list', 'items': [_pack(v, arrays) for v in value]}
    if value is None or type(value) in (str, int, float, bool):
        return {'type': 'value', 'value': value}
    raise TypeError('Unsupported golden value: ' + str(type(value)))


def _unpack(node, arrays):
    kind = node['type']
    if kind in ('array', 'scalar'):
        value = arrays[node['key']].copy(order='K')
        return value[()] if kind == 'scalar' else value
    if kind == 'dict':
        return {k: _unpack(v, arrays) for k, v in node['items'].items()}
    if kind in ('tuple', 'list'):
        result = [_unpack(v, arrays) for v in node['items']]
        return tuple(result) if kind == 'tuple' else result
    if kind == 'value':
        return node['value']
    raise ValueError('Unknown golden value type')


@contextmanager
def _lock(path):
    # Advisory per-key lock: concurrent framework workers must not multiply
    # reference generation. All readers still verify the published bundle.
    import fcntl
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def load_or_create(bench, preset, numpy, cache, *, required=False, load_values=True):
    contract = identity(bench, preset, numpy)
    key = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    root = Path(cache).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / key
    with _lock(root / (key + '.lock')):
        created = False
        if not target.exists():
            if required:
                raise FileNotFoundError('Required golden is missing: ' + str(target))
            bundle = reference(bench, preset, numpy)
            if identity(bench, preset, numpy) != contract:
                raise ValueError('Numerical source changed while generating the golden')
            temporary = Path(tempfile.mkdtemp(prefix='.' + key + '.', dir=root))
            try:
                arrays = {}
                structure = _pack(bundle, arrays)
                np.savez(temporary / 'values.npz', **arrays)
                metadata = {'identity': contract, 'key': key, 'structure': structure,
                            'sha256': digest(temporary / 'values.npz'),
                            'structure_sha256': hashlib.sha256(json.dumps(structure, sort_keys=True).encode()).hexdigest(),
                            'created_utc': datetime.now(timezone.utc).isoformat(),
                            'numpy_version': np.__version__, 'producer_executions': 1}
                (temporary / 'golden.json').write_text(json.dumps(metadata, indent=2) + '\n')
                os.rename(temporary, target)
                created = True
                # Do not retain the producer arrays while reading the saved
                # values back, especially for multi-gigabyte L inputs.
                del bundle, arrays
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
        # Corruption is an error, never an implicit regeneration or a pass.
        metadata = json.loads((target / 'golden.json').read_text())
        if metadata['identity'] != contract or metadata['key'] != key:
            raise ValueError('Golden identity mismatch: ' + str(target))
        if hashlib.sha256(json.dumps(metadata['structure'], sort_keys=True).encode()).hexdigest() != metadata['structure_sha256']:
            raise ValueError('Golden structure integrity mismatch: ' + str(target))
        if digest(target / 'values.npz') != metadata['sha256']:
            raise ValueError('Golden integrity mismatch: ' + str(target))
        bundle = None
        if load_values:
            with np.load(target / 'values.npz', allow_pickle=False) as arrays:
                bundle = _unpack(metadata['structure'], arrays)
    event = {'key': key, 'path': str(target), 'status': 'created' if created else 'hit',
             'producer_executions': int(created), 'sha256': metadata['sha256']}
    return bundle, event


def validate(bench, expected, actual):
    """Retain NPBench tolerances, while checking complete observable structure."""
    from . import utilities as util
    if len(expected['returns']) != len(actual['returns']):
        return False
    if expected['arrays'].keys() != actual['arrays'].keys():
        return False
    ref = [*expected['returns'], *expected['arrays'].values()]
    val = [*actual['returns'], *[actual['arrays'][k] for k in expected['arrays']]]
    for x, y in zip(ref, val):
        if np.shape(x) != np.shape(y) or np.asarray(x).dtype != np.asarray(y).dtype:
            return False
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            return False
    return bool(util.validate(ref, val, rtol=bench.info.get('rtol', 1e-5),
                              atol=bench.info.get('atol', 1e-8),
                              norm_error=bench.info.get('norm_error', 1e-5)))
