#!/usr/bin/env python3
"""Independent metadata/direct-call diagnostics; never compiles or uses Region."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import tempfile
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection', type=Path, required=True)
    parser.add_argument('--benchmark', required=True)
    parser.add_argument('--implementation', default='auto_opt')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inspect-only', action='store_true')
    parser.add_argument('--calls', type=int, default=7)
    args = parser.parse_args()
    if args.calls < 1:
        parser.error('--calls must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    args.output = Path(tempfile.mkdtemp(prefix='attempt-', dir=args.output))
    cell = args.collection/'baselines'/args.benchmark/'dace_gpu'/args.implementation
    manifests = list((cell/'runs').glob('*/manifest.json'))
    if len(manifests) != 1:
        raise ValueError('Expected exactly one recorded lifecycle')
    manifest = json.loads(manifests[0].read_text())
    worker = manifests[0].parent/args.implementation
    record = json.loads((worker/'dace-artifact.json').read_text())
    folder = (worker/record['build_folder']).resolve()
    if record.get('schema') != 2 or not folder.is_relative_to(worker.resolve()):
        raise ValueError('Expected isolated schema-2 artifact')
    from npbench.infrastructure.lifecycle import artifact_state
    # The recorded initial artifact set must still match before direct loading.
    samples = [json.loads(line) for line in (worker/'process-0.jsonl').read_text().splitlines()]
    expected = next((r['artifacts']['after'] for r in reversed(samples) if r.get('artifacts')), None)
    if expected is None or artifact_state(worker) != expected:
        raise ValueError('Saved artifact integrity does not match the original observation')
    import dace
    from npbench.infrastructure.dace_restore import normalize_nested_zero_offsets, load_artifact
    original = (folder/'program.sdfg').read_bytes()
    normalized, changes = normalize_nested_zero_offsets(json.loads(original))
    inspection = {'benchmark': args.benchmark, 'implementation': args.implementation,
                  'source_sdfg_sha256': hashlib.sha256(original).hexdigest(), 'changes': changes,
                  'source_manifest': str(manifests[0]), 'mode': 'metadata_only' if args.inspect_only else 'direct_call'}
    with dace.config.set_temporary('testing', 'deserialize_exception', value=True):
        try:
            dace.SDFG.from_file(str(folder/'program.sdfg'))
            inspection['original_deserialization'] = 'passed'
        except Exception as error:
            inspection['original_deserialization'] = type(error).__name__ + ': ' + str(error)
        try:
            dace.SDFG.from_json(normalized)
            inspection['normalized_deserialization'] = 'passed'
        except Exception as error:
            inspection['normalized_deserialization'] = type(error).__name__ + ': ' + str(error)
    (args.output/'inspection.json').write_text(json.dumps(inspection, indent=2)+'\n')
    if args.inspect_only:
        return 0 if inspection['normalized_deserialization'] == 'passed' else 1
    # Explicitly forbid any compiler fallback, including inside the loader.
    def forbidden(*unused, **kwargs):
        raise RuntimeError('Compilation is forbidden in direct artifact diagnostics')
    dace.SDFG.compile = forbidden
    compiled = load_artifact(folder, audit_path=args.output/'restore.json')
    import numpy as np
    import cupy as cp
    from npbench.infrastructure import Benchmark, generate_framework, golden
    bench = Benchmark(args.benchmark)
    bundle, event = golden.load_or_create(bench, manifest['arguments']['preset'], generate_framework('numpy'),
                                         manifest['arguments']['golden_cache'], required=True)
    if event['key'] != manifest['golden']['key'] or event['sha256'] != manifest['golden']['sha256']:
        raise ValueError('Golden differs from recorded artifact test')
    for call in range(args.calls):
        data = golden.clone_data(bundle['inputs'])
        device = {k: cp.asarray(v) if k in bench.info['array_args'] else v for k,v in data.items()}
        cp.cuda.runtime.deviceSynchronize()
        start = time.perf_counter()
        returned = compiled(**device)
        cp.cuda.runtime.deviceSynchronize()
        host = lambda x: x.get() if isinstance(x, cp.ndarray) else x
        returns = [host(x) for x in golden.outputs(returned)]
        for i,value in enumerate(returns):
            if i < len(bundle['returns']) and np.ndim(bundle['returns'][i]) == 0 and isinstance(value,np.ndarray) and value.shape == (1,):
                returns[i] = value.reshape(())[()]
        actual = {'returns': returns, 'arrays': {k:host(device[k]) for k in bench.info['array_args']}}
        elapsed = time.perf_counter()-start
        failures=[]
        valid=golden.validate(bench,bundle,actual,diagnostics=failures)
        row={'call_index':call, 'status':'passed' if valid else 'validation_failed',
             'diagnostic_call_seconds':elapsed, 'not_a_lifecycle_performance_sample':True,
             'validation_failures':failures, 'golden_key':event['key'], 'golden_producer_executions':event['producer_executions']}
        with (args.output/'calls.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        if not valid:return 1
        del data, device, returned, returns, actual
    if artifact_state(worker)!=expected:raise ValueError('Direct loading changed original artifact files')
    return 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
