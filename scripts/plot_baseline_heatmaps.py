#!/usr/bin/env python3
"""Plot final-results.csv; selection belongs to the research evidence selector.

Use --evidence with the selected-results evidence directory. This renderer does
not merge batches, select alternative results, or run numerical benchmarks.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'npbench-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle


COLUMNS = [
    ('numpy', 'default', 'default'),
    ('numba', 'nopython-mode', 'nopython'),
    ('numba', 'nopython-mode-parallel', 'nopython\nparallel'),
    ('numba', 'nopython-mode-parallel-range', 'nopython\nparallel-range'),
    ('numba', 'object-mode', 'object'),
    ('numba', 'object-mode-parallel', 'object\nparallel'),
    ('numba', 'object-mode-parallel-range', 'object\nparallel-range'),
    ('dace_cpu', 'fusion', 'fusion'),
    ('dace_cpu', 'parallel', 'parallel'),
    ('dace_cpu', 'auto_opt', 'auto_opt'),
    ('dace_gpu', 'fusion', 'fusion'),
    ('dace_gpu', 'parallel', 'parallel'),
    ('dace_gpu', 'auto_opt', 'auto_opt'),
    ('cupy', 'default', 'default'),
]
PHASES = {
    'initialization': ('Initialization', 'First region call with empty implementation artifacts'),
    'fresh_process': ('New process', 'First region call in a new Python process; persistent reuse where supported'),
    'same_process': ('Same-process repeat', 'Median of process medians for subsequent calls in the same Python process'),
}


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def format_time(value):
    if value >= 1:
        return f'{value:.3g} s'
    if value >= .001:
        return f'{value * 1000:.3g} ms'
    return f'{value * 1e6:.3g} us'


def load_final(evidence):
    checks = json.loads((evidence/'SHA256SUMS.json').read_text())
    hashes = {}
    for name in ('final-results.csv', 'summary.json'):
        hashes[name] = hashlib.sha256((evidence/name).read_bytes()).hexdigest()
        assert hashes[name] == checks[name], f'Final input checksum mismatch: {name}'
    summary = json.loads((evidence/'summary.json').read_text())
    rows = read_csv(evidence/'final-results.csv')
    benchmarks = sorted({r['benchmark'] for r in rows})
    registered = {}
    for row in rows:
        ident = (row['benchmark'], row['framework'], row['implementation'], row['phase'])
        assert ident not in registered and row['profile'] == 'system'
        assert row['framework'] in {c[0] for c in COLUMNS}
        value = float(row['seconds']) if row['seconds'] else None
        assert (value is not None) == (row['status'] == 'observed')
        assert value is None or (math.isfinite(value) and value > 0)
        registered[ident] = dict(row, seconds=value, state=row['status'], restricted=row['restricted'] == 'True')
    assert len(registered) == summary['lifecycle_rows'] and len(benchmarks) == summary['benchmarks']
    cells = []
    for phase in PHASES:
        for benchmark in benchmarks:
            for framework, implementation, _ in COLUMNS:
                key = (benchmark, framework, implementation, phase)
                cells.append(registered.get(key, dict(benchmark=benchmark, framework=framework,
                    implementation=implementation, phase=phase, state='N/A', seconds=None, restricted=False)))
    assert sum(c['state'] != 'N/A' for c in cells) == len(rows)
    return benchmarks, cells, hashes, summary


def render(output, phase, benchmarks, cells, norm, summary):
    # Grouped headers, explicit statuses and labelled times follow CoVA's
    # heatmap presentation; absolute shared log colors avoid implying a ranking.
    cmap = plt.get_cmap('viridis')
    fig = plt.figure(figsize=(23, 29), facecolor='white')
    ax = fig.add_axes([.165, .122, .813, .765])
    ax.set_xlim(-.5, len(COLUMNS) - .5)
    ax.set_ylim(len(benchmarks) - .5, -.5)
    ax.set_xticks(range(len(COLUMNS)), [r[2] for r in COLUMNS], fontsize=10)
    ax.xaxis.tick_top()
    ax.tick_params(axis='both', length=0, pad=9)
    ax.set_yticks(range(len(benchmarks)), benchmarks, fontsize=10)
    for spine in ax.spines.values():
        spine.set_color('#a0aaba')
    state_colors = {'N/A': '#f3f5f7', 'missing': '#dce1e6', 'unsupported': '#ddd5ef',
                    'validation': '#f5cccc', 'error': '#f5cccc', 'failed': '#f5cccc',
                    'terminated': '#f5cccc', 'timeout': '#f5dfc0', 'limited': '#f4e6b8'}
    lookup = {(c['benchmark'], c['framework'], c['implementation']): c for c in cells if c['phase'] == phase}
    for y, benchmark in enumerate(benchmarks):
        for x, (fw, impl, _) in enumerate(COLUMNS):
            cell = lookup[(benchmark, fw, impl)]
            value = cell['seconds']
            color = cmap(norm(value)) if value is not None else matplotlib.colors.to_rgba(state_colors[cell['state']])
            ax.add_patch(Rectangle((x-.5, y-.5), 1, 1, facecolor=color, edgecolor='white', linewidth=.7))
            luminance = .2126 * color[0] + .7152 * color[1] + .0722 * color[2]
            ink = '#ffffff' if luminance < .46 else '#182331'
            label = (format_time(value) + (' *' if cell['restricted'] else '') + (' †' if cell.get('repaired') == 'True' else '')) if value is not None else cell['state']
            ax.text(x, y, label, ha='center', va='center', fontsize=10, color=ink, fontweight='medium')
    groups = [(0, 0, 'NumPy | CPU'), (1, 6, 'Numba | CPU'), (7, 9, 'DaCe | CPU'),
              (10, 12, 'DaCe | GPU'), (13, 13, 'CuPy | GPU')]
    for start, end, title in groups:
        ax.text((start+end)/2, 1.047, title, transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=13, fontweight='bold', color='#243347')
        if start:
            ax.axvline(start-.5, color='#596575', linewidth=1.5)
    fig.text(.5, .965, f'NPBench L | {PHASES[phase][0]} | Final results', ha='center', fontsize=25, fontweight='bold', color='#152536')
    fig.text(.5, .946, PHASES[phase][1], ha='center', fontsize=12, color='#455466')
    fig.text(.5, .931, f'{len(benchmarks)} workloads | 14 framework routes, explicit repairs marked † | 96 logical CPUs offered, one A100X',
             ha='center', fontsize=11, color='#455466')
    bar = fig.add_axes([.30, .093, .52, .008])
    cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=bar, orientation='horizontal')
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('Region time (seconds, shared logarithmic scale) | darker = shorter | not a certified ranking', fontsize=11)
    notes = [
        'One selected value per implementation and lifecycle. New qualified measurements replace old values; samples are never pooled across batches.',
        '* Restricted observation: variability, incomplete recheck, drift, process-0-only or unverified historical reuse. Detailed reasons are in final-results.csv.',
        'validation/error/terminated/timeout: excluded timings   unsupported: no persistent reuse   missing: source absent   N/A: variant not registered.',
        'Host-to-host region, metric 3, strict_region_v1. All timings are conditional observations; unmarked cells do not certify a ranking or a 15% advantage.',
        '† Explicit source repair; adi_corrected uses a different formula/golden. Original ADI is retained in historical evidence.',
        'Initialization: independent cold trials where recorded; other cells are single observations. See CSV sampling/provenance.',
        'System resource policy only. Resource-tuned diagnostics remain separate in the evidence, including the faster one-thread durbin observation.',
        f"Dataset: {summary['dataset_id']} | Selected data and provenance: baseline_heatmaps_data.json",
    ]
    for i, line in enumerate(notes):
        fig.text(.055, .061 - i*.0074, line, fontsize=10, color='#455466')
    fig.savefig(output / f'baseline_{phase}_heatmap.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--evidence', required=True, type=Path, help='Research evidence containing final-results.csv')
    parser.add_argument('--output', type=Path, default=Path('.'))
    args = parser.parse_args()
    evidence = args.evidence.resolve()
    benchmarks, cells, hashes, summary = load_final(evidence)
    values = [c['seconds'] for c in cells if c['seconds'] is not None]
    limits = [10**math.floor(math.log10(min(values))), 10**math.ceil(math.log10(max(values)))]
    args.output.mkdir(parents=True, exist_ok=True)
    for phase in PHASES:
        render(args.output, phase, benchmarks, cells, LogNorm(*limits), summary)
    payload = dict(dataset_id=summary['dataset_id'], evidence=str(evidence), input_sha256=hashes,
                   shared_log_scale_seconds=limits, benchmarks=benchmarks, columns=COLUMNS, cells=cells,
                   selection_owner='research evidence select_results.py; renderer consumes final-results.csv unchanged')
    (args.output/'baseline_heatmaps_data.json').write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'figures': 3, 'benchmarks': len(benchmarks), 'columns': len(COLUMNS),
                      'planned_cells_per_phase': summary['registered_cells'], 'colored_cells': len(values), 'shared_scale': limits}))


if __name__ == '__main__':
    main()
