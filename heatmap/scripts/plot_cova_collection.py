#!/usr/bin/env python3
"""Render a complete CoVA collection alongside unchanged baseline observations."""
import argparse
from collections import Counter
import copy
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
from plot_baseline_with_cova import COVA_COLUMNS, HELPERS, lifecycle_cells, verify_merge
from plot_baseline_heatmaps import PHASES, render
from matplotlib.colors import LogNorm


def build_payload(baseline, evidence):
    entries = json.loads(gzip.decompress((evidence / 'cells.json.gz').read_bytes()))
    summary = json.loads((evidence / 'summary.json').read_text())
    if baseline.get('host') and summary.get('host'):
        assert baseline['host'] == summary['host'], 'Cross-host absolute times cannot be merged'
    registered = {(e['benchmark'], e['framework']): e for e in entries}
    references = {c['benchmark']: c for c in baseline['cells'] if c['framework']=='numpy' and c['phase']=='initialization'}
    cells = []
    for benchmark in baseline['benchmarks']:
        for framework, _, _ in COVA_COLUMNS:
            # Explicit helper routes, irrespective of outcome: never hide their
            # failures by retaining older successes or selecting a faster source.
            helper = benchmark + '_cova_helper'
            source = helper if helper in HELPERS and (helper, framework) in registered else benchmark
            entry = registered[(source, framework)]
            passed = entry['exit_code'] == 0
            rows = entry['rows']
            devices = sorted({p['actual_device'] for r in rows for p in r.get('placement',[])})
            golden = entry.get('golden', {})
            metadata = dict(source_benchmark=source, source_version=entry.get('framework_version', summary['source_version']),
                selected_source=summary['collection_id'], source_files=entry['source_files'],
                gpu_numa_binding=summary.get('gpu_numa_binding', 'unknown') if framework.endswith('_gpu') else 'off', host=summary.get('host'),
                golden_key=golden.get('key'), golden_sha256=golden.get('sha256'),
                reported_devices=devices, primary_gpu_verified=False, historical=False, profiled=False,
                repaired=str(source in HELPERS or benchmark=='adi_corrected'),
                restriction_reasons=['reference_not_ranking', 'historical_competitors_not_rerun', 'system_resources_not_exclusive'] +
                    (['gpu_placement_metadata_only'] if framework.endswith('_gpu') else []),
                failure_details=[{k:r[k] for k in ['status','failure_stage','error'] if k in r} for r in rows if r['status']!='passed'])
            selected = lifecycle_cells(benchmark, framework, rows, passed, metadata)
            for cell in selected:
                if entry['exit_code'] == 125:
                    cell.update(state='missing', status='missing', missing_reason='missing_source')
                if cell['seconds'] is not None:
                    ref = references[benchmark]
                    assert (cell['golden_key'],cell['golden_sha256']) == (ref['golden_key'],ref['golden_sha256'])
                    cell['baseline_numpy_golden_matches'] = True
                cell['display_suffix'] = ('\nCPU' if devices==['cpu'] else '\nGPU?') if framework.endswith('_gpu') and passed else ''
            cells.extend(selected)
    payload = copy.deepcopy(baseline)
    payload.update(dataset_id=summary['collection_id']+'-with-frozen-baseline',
        baseline_dataset_id=baseline['dataset_id'], columns=baseline['columns']+COVA_COLUMNS,
        cells=baseline['cells']+cells, cova_source_version=summary['source_version'],
        cova_gpu_numa_binding=summary.get('gpu_numa_binding', 'unknown'),
        baseline_gpu_numa_binding=baseline.get('gpu_numa_binding', 'unknown'),
        cova_evidence_sha256=hashlib.sha256((evidence/'cells.json.gz').read_bytes()).hexdigest(),
        selection_owner='heatmap/scripts/plot_cova_collection.py; one complete collection; explicit helper substitution',
        caveats=['Baseline cells unchanged, not rerun; no synchronized ranking.',
                 'CoVA strict 3x3 L lifecycle, system resource policy; actual placement recorded, no independent per-cell GPU trace.',
                 'Failures replace old successes; corrected ADI uses only corrected source and golden.'])
    values=[c['seconds'] for c in payload['cells'] if c['seconds'] is not None]
    payload['shared_log_scale_seconds']=[10**math.floor(math.log10(min(values))),10**math.ceil(math.log10(max(values)))]
    return payload


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    baseline=json.loads(a.baseline.read_text())
    assert all(not c['framework'].startswith('cova_') for c in baseline['cells'])
    payload=build_payload(baseline,a.evidence)
    verification=verify_merge(payload,baseline)
    verification['baseline_json_sha256']=hashlib.sha256(a.baseline.read_bytes()).hexdigest()
    a.output.mkdir(parents=True,exist_ok=True)
    groups=[(0,0,'NumPy | CPU'),(1,6,'Numba | CPU'),(7,9,'DaCe | CPU'),(10,12,'DaCe | GPU'),(13,13,'CuPy | GPU'),(14,17,'CoVA | CPU routes'),(18,19,'CoVA | GPU routes')]
    notes=['Frozen 14 competitor columns. cpu-memory binds GPU workers to their GPU CPU/memory node; off leaves them unbound.',
           f"CoVA GPU NUMA policy: {payload['cova_gpu_numa_binding']}; competitor GPU NUMA policy: {payload['baseline_gpu_numa_binding']}. Same host, different collection conditions.",
           'CPU: requested GPU route reported host-only execution. GPU?: placement metadata without independent primary-device trace.',
           '* Reference/restricted observation; no synchronized ranking. Dagger: corrected formula or helper with matching golden.',
           'Failed cells have no timing; no historical success fallback. Explicit corrected ADI and workload repairs keep matching goldens.',
           'Host-to-host calls. Input reset and validation excluded. Hot values: median of process medians.',
           'Competitor versions and sampling conditions remain historical; small cross-collection differences are not causal speedups.',
           'Data and scalar provenance: baseline_with_cova_heatmaps_data.json']
    for phase in PHASES:
        render(a.output,phase,payload['benchmarks'],payload['cells'],LogNorm(*payload['shared_log_scale_seconds']),payload,
               columns=payload['columns'],groups=groups,title='Baseline + CoVA full collection',
               subtitle=f"{payload['dataset_id']} | CoVA {str(payload['cova_source_version'])[:8]} | frozen baseline {payload['baseline_dataset_id']}",notes=notes,filename_prefix='baseline_with_cova')
    (a.output/'baseline_with_cova_heatmaps_data.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    (a.output/'baseline_with_cova_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    fields=['benchmark','framework','implementation','phase','state','seconds','source_benchmark','source_version','samples','processes','golden_key','golden_sha256','reported_devices','restriction_reasons']
    with (a.output/'baseline_with_cova_results.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(payload['cells'])
    print(json.dumps(verification))

if __name__=='__main__':
    main()
