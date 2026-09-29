#!/usr/bin/env python3
"""Build a reference snapshot from complete archived ABBA replay cohorts.

Result processing only. All complete cohorts contribute equally per process;
old-side controls and incomplete cohorts never supply replacement timings.
"""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
from plot_baseline_heatmaps import PHASES, render
from matplotlib.colors import LogNorm
from plot_baseline_with_cova import HELPERS, aggregate, verify_merge


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_overlay(previous, baseline, evidence):
    cohorts = [(f'main_cases/{i}', item) for i, item in enumerate(evidence['main_cases'])]
    assert len({item['case'] for _, item in cohorts}) == len(cohorts) == 9
    cohorts.append(('mandelbrot2_cpu_confirm', evidence['mandelbrot2_cpu_confirm']))
    grouped = {}
    for source, item in cohorts:
        assert item['resource_policy'] == 'system'
        assert len(item['preflight_host_busy_fraction']) == 4
        assert all(0 <= x < .15 for x in item['preflight_host_busy_fraction'])
        assert item['preflight_gpu_external_processes'] == [False] * 4
        assert [p['position'] for p in item['cells']['current']['positions']] == [1, 2]
        grouped.setdefault(item['case'], []).append((source, item))
    payload = copy.deepcopy(previous)
    previous_cells, changes = [], []
    for index, old in enumerate(previous['cells']):
        case = old.get('source_benchmark', old['benchmark']) + ':' + old['framework'].removeprefix('cova_')
        if not old['framework'].startswith('cova_') or case not in grouped or old['phase'] == 'initialization':
            continue
        assert old['state'] == 'observed' and old['source_version'] == previous['cova_source_version']
        assert old['source_version'].startswith('f929db7f')
        samples, lineage, devices, hashes = [], [], set(), None
        for cohort_index, (source, item) in enumerate(grouped[case]):
            benchmark, route = item['case'].split(':')
            assert HELPERS.get(benchmark, benchmark) == old['benchmark']
            assert item['golden_sha256'] == old['golden_sha256']
            current_hashes = item['compiled_binary_sha256']['current']
            assert current_hashes and (hashes is None or hashes == current_hashes)
            hashes = current_hashes
            for position in item['cells']['current']['positions']:
                assert len(position['hot_seconds']) == item['calls_per_position'] - 1
                process = f'{cohort_index}:{position["position"]}'
                values = [position['fresh_seconds']] if old['phase'] == 'fresh_process' else position['hot_seconds']
                for call, seconds in enumerate(values, 0 if old['phase'] == 'fresh_process' else 1):
                    samples.append(dict(process_index=process, call_index=call, seconds=seconds))
                devices.update(p[0] for p in position['placement'])
            lineage.append('replay_evidence.json#/' + source)
        assert sorted(devices) == old['reported_devices']
        seconds, medians = aggregate(samples)
        cell = copy.deepcopy(old)
        cell.update(seconds=seconds, raw_samples=samples, process_medians=medians,
                    samples=len(samples), processes=len(medians),
                    selected_source='2026-09-29-complete-artifact-replay-cohorts',
                    source_files=lineage, replay_binary_sha256=hashes,
                    implementation_sources=grouped[case][0][1]['implementation_sources'],
                    source_date='2026-09-29', reference_overlay=True,
                    validation_provenance='Strict validation and unchanged artifacts audited in archived probe; no new execution',
                    selection_reason='All complete current-side cohorts; equal process weights; incomplete cohort excluded')
        cell['restriction_reasons'] += ['partial_matrix_overlay', 'mixed_sampling_dates', 'replay_only_no_initialization']
        if old['benchmark'] in ('mandelbrot1', 'mandelbrot2') and old['framework'] == 'cova_llvm_cpu':
            cell['restriction_reasons'].append('residual_variability_unresolved')
        cell['display_suffix'] = old.get('display_suffix', '') + (' R' if old.get('display_suffix') else '\nR')
        previous_cells.append(copy.deepcopy(old))
        changes.append(dict(benchmark=old['benchmark'], framework=old['framework'], phase=old['phase'],
                            previous_seconds=old['seconds'], seconds=seconds, samples=len(samples), processes=len(medians)))
        payload['cells'][index] = cell
    assert len(changes) == 18
    payload.update(dataset_id='2026-09-29-reference-baseline-replay-overlay',
                   parent_dataset_id=previous['dataset_id'], overlay_previous_cells=previous_cells,
                   overlay_changes=changes, overlay_policy='All complete current-side cohorts, median of process medians; no fastest selection',
                   excluded_cohorts=[evidence['incomplete_followup']],
                   selection_owner='heatmap/scripts/overlay_cova_replays.py',
                   caveats=['Mixed sampling reference snapshot, not a new full-matrix certification or synchronized ranking.',
                            'Nine routes updated only in fresh and hot phases; 09-28 initialization and every other cell unchanged.',
                            'Mandelbrot helper CPU uses both complete replay cohorts with equal process weights; incomplete M1 followup excluded.',
                            'System resources and historical competitors; preflight is not continuous occupancy monitoring.',
                            'R marks replay replacements. Existing CPU/GPU? placement restrictions remain.'])
    payload['parent_cova_evidence_sha256'] = payload.pop('cova_evidence_sha256')
    verification = verify_merge(payload, baseline)
    unchanged = sum(a == b for a, b in zip(previous['cells'], payload['cells']))
    assert unchanged == len(previous['cells']) - 18
    assert all(a == b for a, b in zip(previous['cells'], payload['cells']) if a['phase'] == 'initialization')
    verification.update(updated_lifecycle_cells=18, unchanged_lifecycle_cells=unchanged,
                        initialization_unchanged=True, original_failures_unchanged=True,
                        all_complete_cohorts_used=True, incomplete_cohorts_excluded=True,
                        shared_scale_unchanged=payload['shared_log_scale_seconds'] == previous['shared_log_scale_seconds'])
    return payload, verification


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    previous, baseline, evidence = [json.loads(path.read_text()) for path in (args.previous, args.baseline, args.evidence)]
    payload, verification = build_overlay(previous, baseline, evidence)
    payload['overlay_input_sha256'] = {name: digest(path) for name, path in [('parent_json', args.previous), ('baseline_json', args.baseline), ('replay_evidence', args.evidence)]}
    verification['input_sha256'] = payload['overlay_input_sha256']
    args.output.mkdir(parents=True, exist_ok=True)
    groups = [(0,0,'NumPy | CPU'),(1,6,'Numba | CPU'),(7,9,'DaCe | CPU'),(10,12,'DaCe | GPU'),(13,13,'CuPy | GPU'),(14,17,'CoVA | CPU routes'),(18,19,'CoVA | GPU routes')]
    notes = ['Reference baseline: 09-28 full collection plus complete 09-29 artifact replay cohorts; mixed sampling dates.',
             'R: replay replacement in nine routes, fresh/hot only. Initialization and all other cells unchanged.',
             'Hot: median of per-process medians; every complete current-side process included, no fastest selection.',
             'Usually 2 processes x (1 fresh + 2 hot); Mandelbrot helper CPU: 4 processes, 14 hot samples from two complete cohorts.',
             'Incomplete resource-gated cohort excluded. Residual Mandelbrot CPU variability remains unresolved.',
             'Historical 14 competitor columns unchanged. No synchronized ranking or new full-matrix certification.',
             '* Restricted reference; dagger: corrected/helper golden. CPU/GPU?: original placement limits; no continuous occupancy trace.',
             'Exact lineage, previous values, identities and samples: baseline_with_cova_heatmaps_data.json; R evidence: replay_evidence.json.']
    for phase in PHASES:
        render(args.output, phase, payload['benchmarks'], payload['cells'], LogNorm(*payload['shared_log_scale_seconds']), payload,
               columns=payload['columns'], groups=groups, title='Baseline reference | mixed sampling',
               subtitle='CoVA f929db7f | 09-28 full matrix + 09-29 replay overlay | R = rechecked | historical competitors unchanged',
               notes=notes, filename_prefix='baseline_with_cova')
    for name, data in [('baseline_with_cova_heatmaps_data.json',payload),('baseline_with_cova_verification.json',verification),('replay_evidence.json',evidence)]:
        (args.output/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    fields=['benchmark','framework','implementation','phase','state','seconds','selected_source','source_version','samples','processes','golden_sha256','reported_devices','restriction_reasons']
    with (args.output/'baseline_with_cova_results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(payload['cells'])
    print(json.dumps(verification))


if __name__ == '__main__':
    main()
