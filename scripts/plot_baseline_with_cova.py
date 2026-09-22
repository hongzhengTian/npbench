#!/usr/bin/env python3
"""Add persisted CoVA L observations to an unchanged baseline JSON; no execution."""
import argparse
from collections import Counter, defaultdict
import copy
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import tarfile

from plot_baseline_heatmaps import PHASES, render
from matplotlib.colors import LogNorm

COVA_COLUMNS = [
    ('cova_serial_cpu', 'default', 'EmitC\nserial'),
    ('cova_openmp_cpu', 'default', 'EmitC\nOpenMP'),
    ('cova_llvm_cpu_serial', 'default', 'LLVM\nserial'),
    ('cova_llvm_cpu', 'default', 'LLVM\nOpenMP'),
    ('cova_openmp_gpu', 'default', 'OpenMP\ntarget'),
    ('cova_llvm_gpu', 'default', 'LLVM\nCUDA'),
]
HELPERS = {name + '_cova_helper': name for name in
           ('azimint_hist', 'stockham_fft', 'mandelbrot2')}
HISTORY = '2026-09-21-npbench-cova-L'
CLOSURE = '2026-09-22-npbench-cova-closure'
LATEST = '2026-09-22-nbody-azimint-final/2026-09-22-azimint-stage4-5-evidence'


def aggregate(samples):
    """Equal weight per process, never pooled across processes or batches."""
    groups = defaultdict(list)
    for sample in samples:
        value = sample['seconds']
        assert math.isfinite(value) and value > 0
        groups[str(sample['process_index'])].append(value)
    medians = {key: statistics.median(values) for key, values in groups.items()}
    return (statistics.median(medians.values()) if medians else None), medians


class Evidence:
    def __init__(self, root):
        self.root = root
        self.hashes = {}

    def read(self, name):
        data = (self.root / name).read_bytes()
        self.hashes[name] = hashlib.sha256(data).hexdigest()
        return gzip.decompress(data) if name.endswith('.gz') else data

    def json(self, name):
        return json.loads(self.read(name))

    def archive(self, name):
        # Record archive identity without extracting files into either repository.
        self.hashes[name] = hashlib.sha256((self.root / name).read_bytes()).hexdigest()
        return tarfile.open(self.root / name)


def lifecycle_cells(benchmark, framework, rows, passed, metadata, failure=None):
    if passed:
        assert rows and all(row['status'] == 'passed' and row['validated'] is True for row in rows), (benchmark, framework, metadata['selected_source'], metadata['source_files'])
        assert all(row['metric_version'] == 3 and row['validation_contract'] == 'strict_region_v1' for row in rows)
        assert len({(r['golden_key'], r['golden_sha256']) for r in rows}) == 1
    state = 'observed' if passed else ('validation' if failure == 'numerical_failure' or
            any(r['status'] == 'validation_failed' for r in rows) else
            'timeout' if failure == 'timeout' or any(r['status'] == 'timeout' for r in rows) else 'error')
    result = []
    for phase in PHASES:
        samples = [dict(process_index=r['process_index'], call_index=r['call_index'],
                        seconds=r['time']) for r in rows if r.get('phase') == phase and
                   r['status'] == 'passed' and r.get('validated') is True and passed]
        seconds, medians = aggregate(samples)
        cell = dict(metadata, benchmark=benchmark, framework=framework, implementation='default',
                    phase=phase, preset='L', profile='system', seconds=seconds,
                    state=(state if not passed else 'observed' if seconds is not None else 'missing'),
                    restricted=True, samples=len(samples), processes=len(medians),
                    process_medians=medians, raw_samples=samples,
                    comparison_use='reference_only', small_margin_claim_supported=False)
        cell['status'] = cell['state']
        result.append(cell)
    return result


def select_cova(evidence, benchmarks):
    selected = {}
    lineage = defaultdict(list)

    def put(cells):
        for cell in cells:
            key = (cell['benchmark'], cell['framework'], cell['phase'])
            if key in selected:
                old = selected[key]
                lineage[key].append({k: old.get(k) for k in
                                     ('selected_source', 'source_version', 'source_benchmark', 'seconds', 'state')})
            selected[key] = cell

    for entry in evidence.json(HISTORY + '/cells/index.json'):
        if entry['benchmark'] not in benchmarks:
            continue  # Original ADI must never populate the adi_corrected row.
        rows = [json.loads(line) for name in entry['rows']
                for line in evidence.read(HISTORY + '/' + name).splitlines() if line]
        metadata = dict(source_benchmark=entry['benchmark'], source_version=entry['source_version'],
                        selected_source=HISTORY, source_files=entry['rows'],
                        golden_key=entry['golden_key'], golden_sha256=entry['golden_sha256'],
                        reported_devices=entry['reported_devices'], primary_gpu_verified=False,
                        repaired='False', historical=True, profiled=False,
                        restriction_reasons=['historical_build', 'reference_not_ranking',
                                             'gpu_placement_metadata_only'] if entry['framework'].endswith('_gpu') else
                                            ['historical_build', 'reference_not_ranking'])
        put(lifecycle_cells(entry['benchmark'], entry['framework'], rows, entry['status'] == 'passed', metadata))

    devices = {x['run_id']: x for x in evidence.json(CLOSURE + '/final/device-audit.json')}
    for step in range(1, 7):
        archive_name = f'{CLOSURE}/stages/step{step}.tar.gz'
        with evidence.archive(archive_name) as archive:
            audit = json.load(archive.extractfile('audit.json'))
            names = archive.getnames()
            used = set()
            for entry in audit['final']:
                source_benchmark = entry['benchmark']
                benchmark = HELPERS.get(source_benchmark, source_benchmark)
                if entry['preset'] != 'L' or benchmark not in benchmarks:
                    continue
                key = (benchmark, entry['framework'])
                assert key not in used, ('ambiguous batch', step, key)
                used.add(key)
                root = 'cells/' if step <= 2 else 'runs/'
                prefix = root + entry['path'] + '/default/'
                paths = sorted(n for n in names if n.startswith(prefix) and
                               Path(n).name.startswith('process-') and n.endswith('.jsonl.gz'))
                rows = [json.loads(line) for name in paths for line in
                        gzip.decompress(archive.extractfile(name).read()).splitlines() if line]
                device = devices.get(entry['run_id'], {})
                metadata = dict(source_benchmark=source_benchmark, source_version=entry['version'],
                                selected_source=f'closure-step{step}', source_archive=archive_name,
                                source_files=paths, run_id=entry['run_id'],
                                golden_key=entry['golden']['key'], golden_sha256=entry['golden']['sha256'],
                                reported_devices=sorted({p['actual_device'] for r in rows for p in r.get('placement', [])}),
                                primary_gpu_verified=bool(device.get('gpu') and device.get('primary_device_qualified')),
                                forbidden_compile_restore=device.get('forbidden_compile_restore'),
                                repaired=str(source_benchmark in HELPERS or source_benchmark == 'adi_corrected'),
                                historical=True, profiled=False,
                                restriction_reasons=['historical_build', 'reference_not_ranking'])
                put(lifecycle_cells(benchmark, entry['framework'], rows, entry['passed'], metadata,
                                    entry.get('failure_category')))

    # Explicit final candidate cohorts, never the old paired baselines or S diagnostics.
    cohorts = [
        ('azimint_naive', 'cova_llvm_cpu', ['p0'], False),
        ('azimint_naive', 'cova_llvm_gpu', ['p0', 'p1', 'p2'], True),
        ('gemm', 'cova_llvm_gpu', ['paired', 'r1', 'r2'], False),
        ('mandelbrot1', 'cova_llvm_gpu', ['paired', 'r1', 'r2'], False),
    ]
    for benchmark, framework, processes, profiled in cohorts:
        folder = f'{LATEST}/final/{benchmark}-{framework}-L'
        paths = [folder + '/bootstrap.json'] + [folder + '/' + p + '.json' for p in processes]
        records = [evidence.json(path) for path in paths]
        reference = records[0]
        rows = []
        for index, record in enumerate(records):
            assert (record['benchmark'], record['framework'], record['preset']) == (benchmark, framework, 'L')
            assert record['golden']['producer_executions'] == 0
            assert record['input_hashes'] == reference['input_hashes']
            assert record['golden']['sha256'] == reference['golden']['sha256']
            assert record['compile_forbidden'] == (index > 0)
            assert len(record['records']) == (1 if index == 0 else 3)
            for item in record['records']:
                assert item['valid'] is True and not item['diagnostics']
                phase = 'initialization' if index == 0 else 'fresh_process' if item['call'] == 0 else 'same_process'
                rows.append(dict(process_index=index, call_index=item['call'], phase=phase,
                                 status='passed', validated=True, time=item['seconds'],
                                 metric_version=3, validation_contract='strict_region_v1',
                                 golden_key=record['golden']['key'], golden_sha256=record['golden']['sha256']))
        metadata = dict(source_benchmark=benchmark, source_version='e01a2c3f735d4ed56cd47b76e567f3ecf25ff7af',
                        selected_source='final-optimization-candidate', source_files=paths,
                        golden_key=reference['golden']['key'], golden_sha256=reference['golden']['sha256'],
                        input_hashes=reference['input_hashes'], historical=False, profiled=profiled,
                        repaired='False', reported_devices=(['gpu'] if benchmark == 'azimint_naive' else []) if framework.endswith('_gpu') else ['cpu'],
                        primary_gpu_verified=benchmark == 'azimint_naive' and framework.endswith('_gpu'), forbidden_compile_restore=True,
                        metric_origin='diagnostic Region driver with canonical strict validation; lifecycle mapped from call indices',
                        restriction_reasons=['reference_not_ranking', 'diagnostic_driver',
                                             'single_initialization_trial'] + (['profiled_restores'] if profiled else []))
        cells = lifecycle_cells(benchmark, framework, rows, True, metadata)
        for cell in cells:
            cell['profiled'] = profiled and cell['phase'] != 'initialization'
        put(cells)

    for benchmark in benchmarks:
        for framework, _, _ in COVA_COLUMNS:
            for phase in PHASES:
                key = (benchmark, framework, phase)
                if key not in selected:
                    selected[key] = dict(benchmark=benchmark, framework=framework, implementation='default',
                                         phase=phase, state='missing', status='missing', seconds=None,
                                         restricted=False, restriction_reasons=['no_matching_L_evidence'],
                                         source_benchmark=benchmark, raw_samples=[], samples=0, processes=0)
                cell = selected[key]
                cell['superseded_observations'] = lineage[key]
                marks = []
                if cell.get('historical'):
                    marks.append('H')
                if cell.get('profiled'):
                    marks.append('P')
                if framework.endswith('_gpu') and cell['seconds'] is not None and not cell.get('primary_gpu_verified'):
                    marks.append('CPU' if cell.get('reported_devices') == ['cpu'] else 'GPU?')
                cell['display_suffix'] = ('\n' + ' '.join(marks)) if marks else ''
    return list(selected.values())


def build_payload(baseline_path, evidence_root):
    baseline_bytes = baseline_path.read_bytes()
    baseline = json.loads(baseline_bytes)
    evidence = Evidence(evidence_root)
    cova = select_cova(evidence, baseline['benchmarks'])
    references = {c['benchmark']: c for c in baseline['cells']
                  if c['framework'] == 'numpy' and c['phase'] == 'initialization'}
    for cell in cova:
        if cell['seconds'] is not None:
            ref = references[cell['benchmark']]
            assert (cell['golden_key'], cell['golden_sha256']) == (ref['golden_key'], ref['golden_sha256']), cell
            cell['baseline_numpy_golden_matches'] = True
    payload = copy.deepcopy(baseline)
    payload.update(dataset_id=baseline['dataset_id'] + '-with-cova-reference',
                   baseline_dataset_id=baseline['dataset_id'],
                   baseline_json_sha256=hashlib.sha256(baseline_bytes).hexdigest(),
                   columns=baseline['columns'] + COVA_COLUMNS, cells=baseline['cells'] + cova,
                   cova_evidence_root=str(evidence_root), cova_source_sha256=evidence.hashes,
                   selection_owner='scripts/plot_baseline_with_cova.py; chronological whole-cell overrides, no best-time selection',
                   cova_aggregation='median of process medians; initialization and first restored calls separate',
                   caveats=['Sparse historical observations, not a full matrix on current CoVA.',
                            'Failed cells have no timings; missing corrected ADI routes are not filled from original ADI.',
                            'H: historical build; P: profiled call; CPU/GPU?: GPU route lacks primary GPU verification.',
                            'CoVA values are reference observations, not certified cross-system performance rankings.'])
    values = [c['seconds'] for c in payload['cells'] if c['seconds'] is not None]
    payload['shared_log_scale_seconds'] = [10 ** math.floor(math.log10(min(values))),
                                          10 ** math.ceil(math.log10(max(values)))]
    return payload


def verify_merge(payload, baseline):
    count = len(baseline['cells'])
    assert payload['cells'][:count] == baseline['cells']
    assert payload['columns'][:len(baseline['columns'])] == baseline['columns']
    assert payload['benchmarks'] == baseline['benchmarks']
    expected = len(payload['benchmarks']) * len(payload['columns']) * len(PHASES)
    keys = [(c['benchmark'], c['framework'], c['implementation'], c['phase']) for c in payload['cells']]
    assert len(keys) == len(set(keys)) == expected
    for cell in payload['cells'][count:]:
        assert (cell['seconds'] is not None) == (cell['state'] == 'observed')
        if cell['seconds'] is not None:
            assert cell['baseline_numpy_golden_matches']
            assert aggregate(cell['raw_samples'])[0] == cell['seconds']
        if cell['benchmark'] == 'adi_corrected':
            assert cell['source_benchmark'] == 'adi_corrected'
    return dict(baseline_cells_preserved=count, total_cells=expected, cova_cells=expected-count,
                cova_phase_states=dict(Counter(c['state'] for c in payload['cells'][count:])),
                checks=['baseline cells unchanged', 'unique rectangular matrix', 'strict status gating',
                        'matching golden identities', 'equal process weighting', 'corrected ADI identity'])


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--baseline', type=Path, default=Path('baseline_heatmaps_data.json'))
    parser.add_argument('--evidence-root', type=Path, required=True, help='Research reports/evidence directory')
    parser.add_argument('--output', type=Path, default=Path('.'))
    args = parser.parse_args()
    payload = build_payload(args.baseline, args.evidence_root)
    verification = verify_merge(payload, json.loads(args.baseline.read_text()))
    verification['original_file_sha256'] = {name: hashlib.sha256((args.baseline.parent / name).read_bytes()).hexdigest()
        for name in [args.baseline.name] + [f'baseline_{phase}_heatmap.png' for phase in PHASES]
        if (args.baseline.parent / name).exists()}
    args.output.mkdir(parents=True, exist_ok=True)
    prefix = 'baseline_with_cova'
    groups = [(0, 0, 'NumPy | CPU'), (1, 6, 'Numba | CPU'), (7, 9, 'DaCe | CPU'),
              (10, 12, 'DaCe | GPU'), (13, 13, 'CuPy | GPU'),
              (14, 17, 'CoVA | CPU routes'), (18, 19, 'CoVA | GPU routes')]
    notes = [
        'Original 14 baseline columns are unchanged. Six CoVA routes use existing L evidence only; no benchmark was rerun.',
        'CoVA selection: complete L matrix, successive closure final cohorts, then final optimization candidates; never choose the fastest batch.',
        'H: historical CoVA build (not revalidated on current code). P: profiled restore. CPU / GPU?: requested GPU route lacks verified primary GPU execution.',
        '* Restricted/reference observation. † Explicit repair or CoVA helper. adi_corrected uses only its corrected formula and matching golden.',
        'Failed validations/errors/timeouts have no timings. Missing means no matching observation; it does not assert compiler support or lack of support.',
        'Host-to-host Region calls; input reset and strict checks excluded. Same-process values are medians of per-process medians, not pooled calls.',
        'Latest azimint GPU L is profiled; CPU L has one restored process. Other CoVA cells may come from older code. Not a synchronized performance ranking.',
        'Source identities, sample counts, raw scalar times, device qualifications and superseded observations: baseline_with_cova_heatmaps_data.json',
    ]
    for phase in PHASES:
        render(args.output, phase, payload['benchmarks'], payload['cells'],
               LogNorm(*payload['shared_log_scale_seconds']), payload,
               columns=payload['columns'], groups=groups, title='Baseline + CoVA reference',
               subtitle='54 workloads | 14 baseline + 6 CoVA routes | Existing L observations, mixed collection dates and compiler versions',
               notes=notes, filename_prefix=prefix)
    (args.output / f'{prefix}_heatmaps_data.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    columns = ['benchmark', 'framework', 'implementation', 'phase', 'state', 'seconds', 'source_benchmark',
               'selected_source', 'source_version', 'historical', 'profiled', 'samples', 'processes',
               'restricted', 'repaired', 'golden_key', 'golden_sha256', 'primary_gpu_verified', 'restriction_reasons']
    with (args.output / f'{prefix}_results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for cell in payload['cells']:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in cell.items()})
    (args.output / f'{prefix}_verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    print(json.dumps(dict(figures=3, workloads=len(payload['benchmarks']), routes=len(payload['columns']),
                         cova_states=Counter(c['state'] for c in payload['cells'] if c['framework'].startswith('cova_')))))


if __name__ == '__main__':
    main()
