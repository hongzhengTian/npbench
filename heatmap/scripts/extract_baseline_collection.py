#!/usr/bin/env python3
"""Build a frozen 14-route baseline JSON from extracted scalar collections.

No benchmark execution. Repairs explicitly replace sources, never select minima.
Inputs use the cells.json.gz/collection.json.gz research extraction contract.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
from plot_baseline_heatmaps import COLUMNS, PHASES
from plot_baseline_with_cova import aggregate

DEFAULT_REPAIRS = {'adi_corrected': 'adi_corrected',
                   'azimint_hist_private': 'azimint_hist',
                   'correlation_return': 'correlation', 'mlp_rowmax': 'mlp'}


def build_payload(main, repairs, *, dataset_id, source_hashes=None,
                  repair_map=None, replace_map=None):
    repair_map = DEFAULT_REPAIRS if repair_map is None else repair_map
    replace_map = {'adi': 'adi_corrected'} if replace_map is None else replace_map
    benchmarks = sorted({replace_map.get(e['benchmark'], e['benchmark']) for e in main})
    routes = {(fw, impl) for fw, impl, _ in COLUMNS}
    selected = {}
    for cohort, entries in [('main', main), ('repairs', repairs)]:
        seen = set()
        for entry in entries:
            source = entry['benchmark']
            if cohort == 'main' and source in replace_map:
                continue  # Changed scientific formula: never reuse original ADI.
            target = repair_map.get(source, source) if cohort == 'repairs' else source
            if cohort == 'repairs' and source not in repair_map:
                raise ValueError(f'Unspecified repair source: {source}')
            if target not in benchmarks:
                raise ValueError(f'Repair target outside main matrix: {target}')
            key = target, entry['framework'], entry['implementation']
            if key[1:] not in routes:
                raise ValueError(f'Unknown route: {key}')
            if key in seen:
                raise ValueError(f'Ambiguous attempts: {key}')
            seen.add(key)
            if not entry.get('host'):
                raise ValueError('Missing host identity')
            selected[key] = dict(entry, selected_cohort=cohort)
    hosts = {e['host'] for e in selected.values()}
    if len(hosts) != 1:
        raise ValueError(f'Mixed host collections: {hosts}')
    cells = []
    for benchmark in benchmarks:
        for framework, implementation, _ in COLUMNS:
            entry = selected.get((benchmark, framework, implementation))
            for phase in PHASES:
                cell = dict(benchmark=benchmark, framework=framework,
                    implementation=implementation, phase=phase, preset='L', profile='system',
                    seconds=None, state='N/A', status='N/A', restricted=False,
                    repaired='False', samples=0, processes=0)
                if entry is not None:
                    rows = entry['rows']
                    code = entry['exit_code']
                    partial = code == 2 and framework == 'numba' and implementation.startswith('object-mode')
                    passed = code == 0 or partial
                    if passed:
                        assert rows and all(r['status'] == 'passed' and r['validated'] is True for r in rows)
                        assert all(r['metric_version'] == 3 and r['validation_contract'] == 'strict_region_v1' and r['preset'] == 'L' for r in rows)
                        assert {(r['golden_key'], r['golden_sha256']) for r in rows} == {(entry['golden']['key'], entry['golden']['sha256'])}
                    samples = [dict(process_index=r['process_index'], call_index=r['call_index'], seconds=r['time']) for r in rows if r.get('phase') == phase and passed]
                    seconds, medians = aggregate(samples)
                    state = ('observed' if seconds is not None else 'unsupported' if partial and phase == 'fresh_process' else 'missing') if passed else ('missing' if code == 125 else 'validation' if any(r['status'] == 'validation_failed' for r in rows) else 'timeout' if any(r['status'] == 'timeout' for r in rows) else 'error')
                    golden = entry.get('golden', {})
                    cell.update(seconds=seconds, state=state, status=state, restricted=True,
                        samples=len(samples), processes=len(medians), process_medians=medians,
                        raw_samples=samples, source_benchmark=entry['benchmark'],
                        source_version=entry.get('framework_version'), run_id=entry.get('run_id'),
                        selected_source=entry['selected_cohort'], host=entry['host'],
                        selection_reason='explicit_repaired_variant' if entry['selected_cohort']=='repairs' else 'complete_main_collection',
                        repaired=str(entry['selected_cohort']=='repairs'),
                        initialization_scope='single_main_collection_empty_artifact_call',
                        restriction_reasons=['reference_not_ranking', 'single_initialization_observation'] + (['process_0_only', 'no_persistent_reuse'] if partial else []),
                        golden_key=golden.get('key'), golden_sha256=golden.get('sha256'),
                        metric_version=3, protocol_version=sorted({r['protocol_version'] for r in rows if 'protocol_version' in r}),
                        validation_contract='strict_region_v1', gpu_numa_binding='off',
                        failure_details=[r for r in rows if r['status']!='passed'],
                        comparison_use='conditional_observation_only', small_margin_claim_supported=False)
                cells.append(cell)
    values = [c['seconds'] for c in cells if c['seconds'] is not None]
    return dict(dataset_id=dataset_id, evidence='extracted scalar collections',
        input_sha256=source_hashes or {}, benchmarks=benchmarks, columns=COLUMNS, cells=cells,
        shared_log_scale_seconds=([10**math.floor(math.log10(min(values))), 10**math.ceil(math.log10(max(values)))] if values else [1e-6, 1e3]),
        host=next(iter(hosts)), gpu_numa_binding='off',
        selection_owner='heatmap/scripts/extract_baseline_collection.py; explicit repairs, equal process weighting',
        selection=dict(repair_map=repair_map, replace_map=replace_map,
            raw_main_cells=len(main), raw_repair_cells=len(repairs),
            states=dict(Counter(c['state'] for c in cells))))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--main', type=Path, required=True)
    parser.add_argument('--repairs', type=Path, required=True)
    parser.add_argument('--dataset-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repair-map', type=json.loads, default=DEFAULT_REPAIRS)
    parser.add_argument('--replace-map', type=json.loads, default={'adi':'adi_corrected'})
    args = parser.parse_args()
    hashes = {}
    def load(folder):
        path = folder / 'cells.json.gz'
        hashes[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        collection_path = folder / 'collection.json.gz'
        hashes[str(collection_path.resolve())] = hashlib.sha256(collection_path.read_bytes()).hexdigest()
        collection = json.loads(gzip.decompress(collection_path.read_bytes()))
        assert collection['environment']['NPBENCH_RESOURCE_POLICY']=='system'
        assert collection['environment']['NPBENCH_PRESET']=='L'
        assert collection['environment'].get('NPBENCH_GPU_NUMA_BINDING', 'off') == 'off'
        entries = json.loads(gzip.decompress(path.read_bytes()))
        for entry in entries:
            if not entry.get('host'):
                assert entry['exit_code'] == 125, 'Only missing-source cells may inherit collection host'
                entry['host'] = collection['host']
        return entries
    payload = build_payload(load(args.main), load(args.repairs), dataset_id=args.dataset_id,
        source_hashes=hashes, repair_map=args.repair_map, replace_map=args.replace_map)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    print(json.dumps(payload['selection']))


if __name__ == '__main__':
    main()
