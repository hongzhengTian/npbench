#!/usr/bin/env python3
"""Overlay explicitly selected strict 3x3 repair cohorts on a frozen heatmap."""
import argparse
from collections import defaultdict
import copy
import csv
import hashlib
import json
from pathlib import Path
from matplotlib.colors import LogNorm
from plot_baseline_heatmaps import PHASES, render
from plot_baseline_with_cova import HELPERS, lifecycle_cells, verify_merge


def build_overlay(previous, baseline, evidence, selected, version):
    source = json.loads((evidence / 'source-identity.json').read_text())
    groups = defaultdict(list)
    for path in sorted((evidence / 'records/targets/candidate').glob('*.json')):
        record = json.loads(path.read_text())
        key = (record['benchmark'], record['framework'])
        if key in selected:
            groups[key].append((path, record))
    assert set(groups) == selected, 'Missing selected repair evidence'
    replacements = {}
    hashes = {}
    for (source_benchmark, framework), cohort in sorted(groups.items()):
        cohort.sort(key=lambda pair: (pair[1]['compile_forbidden'], pair[1]['process']))
        assert len(cohort) == 3
        assert [x['compile_forbidden'] for _, x in cohort] == [False, True, True]
        reference = cohort[0][1]
        benchmark = HELPERS.get(source_benchmark, source_benchmark)
        numpy = next(c for c in baseline['cells'] if c['benchmark'] == benchmark and c['framework'] == 'numpy' and c['phase'] == 'initialization')
        rows, devices, files = [], set(), []
        for index, (path, record) in enumerate(cohort):
            relative = str(path.relative_to(evidence))
            files.append(relative)
            hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            assert record['metric'] == 'host_to_host_region_seconds'
            assert record['preset'] == 'L' and record['resource_policy'] == 'system'
            assert record['affinity'] == list(range(96))
            assert record['semantics_version'] == source['semantics_version']
            assert record['input_hashes'] == reference['input_hashes']
            assert record['golden']['producer_executions'] == 0
            assert (record['golden']['key'], record['golden']['sha256']) == (numpy['golden_key'], numpy['golden_sha256'])
            if index:
                assert record['binary_before'] and record['binary_before'] == record['binary_after']
            assert [r['call'] for r in record['records']] == [0, 1, 2]
            for item in record['records']:
                assert item['valid'] and not item['diagnostics']
                assert item['call'] == 0 or item['compile_forbidden']
                if framework.endswith('_gpu'):
                    assert item['placement']
                    assert all(p['actual_device'] in ('gpu', 'mixed') and p['fallback_reason'] is None for p in item['placement'])
                devices.update(p['actual_device'] for p in item['placement'])
                phase = ('initialization' if index == 0 else 'fresh_process') if item['call'] == 0 else 'same_process'
                rows.append(dict(process_index=index, call_index=item['call'], phase=phase,
                                 status='passed', validated=True, time=item['seconds'], metric_version=3,
                                 validation_contract='strict_region_v1', golden_key=record['golden']['key'],
                                 golden_sha256=record['golden']['sha256']))
        metadata = dict(source_benchmark=source_benchmark, source_version=version,
                        measured_source_base=source['cova_head'], measured_patch_sha256=source['patch_sha256'],
                        selected_source='2026-09-25-regression-repairs', source_files=files,
                        golden_key=reference['golden']['key'], golden_sha256=reference['golden']['sha256'],
                        input_hashes=reference['input_hashes'], reported_devices=sorted(devices),
                        primary_gpu_verified=False, historical=False, profiled=False,
                        repaired=str(source_benchmark in HELPERS), failure_details=[],
                        baseline_numpy_golden_matches=True,
                        restriction_reasons=['reference_not_ranking', 'historical_competitors_not_rerun',
                                             'system_resources_not_exclusive', 'partial_matrix_overlay'] +
                                            (['gpu_placement_metadata_only'] if framework.endswith('_gpu') else []),
                        display_suffix='\nGPU?' if framework.endswith('_gpu') else '')
        for cell in lifecycle_cells(benchmark, framework, rows, True, metadata):
            assert (cell['samples'], cell['processes']) == {'initialization': (1, 1), 'fresh_process': (2, 2), 'same_process': (6, 3)}[cell['phase']]
            replacements[(benchmark, framework, cell['phase'])] = cell
    payload = copy.deepcopy(previous)
    changed = []
    for i, old in enumerate(previous['cells']):
        key = (old['benchmark'], old['framework'], old['phase'])
        if key in replacements:
            payload['cells'][i] = replacements.pop(key)
            changed.append(key)
    assert not replacements
    payload.update(dataset_id='2026-09-25-repair-overlay-on-2026-09-24',
                   parent_dataset_id=previous['dataset_id'], cova_source_version='mixed: 316658e2 + ' + version,
                   selection_owner='heatmap/scripts/overlay_cova_repairs.py; explicit target cohorts only',
                   overlay_source_version=version, overlay_evidence_sha256=hashes,
                   overlay_source_identity=source, overlay_cells=[list(k) for k in changed],
                   caveats=['Partial overlay, not a new full-matrix certification.',
                            'Nine selected repair cells use strict 3x3 L measurements; all other cells unchanged.',
                            'Competitors not rerun; mixed source versions and collection dates; no synchronized ranking.'])
    # The inherited digest belongs to the parent full collection, not this overlay.
    payload['parent_cova_evidence_sha256'] = payload.pop('cova_evidence_sha256', None)
    verification = verify_merge(payload, baseline)
    unchanged = [i for i, old in enumerate(previous['cells']) if (old['benchmark'], old['framework'], old['phase']) not in changed]
    assert all(payload['cells'][i] == previous['cells'][i] for i in unchanged)
    verification.update(updated_lifecycle_cells=len(changed), unchanged_lifecycle_cells=len(unchanged),
                        unchanged_log_scale=payload['shared_log_scale_seconds'] == previous['shared_log_scale_seconds'])
    return payload, verification


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ('previous', 'baseline', 'evidence', 'selection', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--source-version', required=True)
    args = parser.parse_args()
    selected = {tuple(x) for x in json.loads(args.selection.read_text())}
    payload, verification = build_overlay(json.loads(args.previous.read_text()), json.loads(args.baseline.read_text()), args.evidence, selected, args.source_version)
    verification['parent_json_sha256'] = hashlib.sha256(args.previous.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    groups = [(0, 0, 'NumPy | CPU'), (1, 6, 'Numba | CPU'), (7, 9, 'DaCe | CPU'), (10, 12, 'DaCe | GPU'), (13, 13, 'CuPy | GPU'), (14, 17, 'CoVA | CPU routes'), (18, 19, 'CoVA | GPU routes')]
    notes = ['Partial update: 9 repaired/retested cells replace 09-24 observations; all other cells remain unchanged.',
             'Updated cells: strict 3 processes x 3 calls; L, system resources, no profiler. Not a new full-matrix certification.',
             'CPU: host-only placement. GPU?: metadata without independent primary-device trace. Mixed compiler versions.',
             'Host-to-host Region timing; reset and validation excluded. Hot values: median of process medians.',
             'Original competitor observations unchanged; no synchronized ranking. Corrected ADI identity unchanged.',
             'Same color scale as 09-24. Scalar samples, source identities and selection are retained in the JSON.']
    for phase in PHASES:
        render(args.output, phase, payload['benchmarks'], payload['cells'], LogNorm(*payload['shared_log_scale_seconds']), payload,
               columns=payload['columns'], groups=groups, title='Baseline + CoVA repair overlay',
               subtitle='2026-09-25 | 9 updated cells on 09-24 full matrix | mixed versions; baseline unchanged',
               notes=notes, filename_prefix='baseline_with_cova')
    for name, value in [('heatmaps_data', payload), ('verification', verification)]:
        (args.output / f'baseline_with_cova_{name}.json').write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    fields = ['benchmark', 'framework', 'implementation', 'phase', 'state', 'seconds', 'source_benchmark', 'source_version', 'samples', 'processes', 'golden_key', 'golden_sha256', 'reported_devices', 'restriction_reasons']
    with (args.output / 'baseline_with_cova_results.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(payload['cells'])
    print(json.dumps(verification))


if __name__ == '__main__':
    main()
