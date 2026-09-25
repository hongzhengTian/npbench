#!/usr/bin/env python3
"""Extract scalar CoVA lifecycle evidence without arrays, binaries or caches."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def extract(run, output):
    output.mkdir(parents=True, exist_ok=False)
    sources = {}
    def read(path):
        sources[str(path.relative_to(run))] = digest(path)
        return json.loads(path.read_text())
    collection = read(run / 'collection.json')
    assert collection['mode'] == 'cova'
    assert collection['environment']['NPBENCH_RESOURCE_POLICY'] == 'system'
    assert collection['environment']['NPBENCH_PRESET'] == 'L'
    cells = []
    for line in (run / 'progress.tsv').read_text().splitlines():
        parts = line.split('\t')
        if len(parts) < 7 or parts[1] != 'cova':
            continue
        _, _, benchmark, framework, implementation, code, *_ = parts
        folder = run / 'cova' / benchmark / framework / implementation
        manifests = list(folder.glob('runs/*/manifest.json'))
        if len(manifests) > 1:
            raise ValueError(f'Multiple attempts require explicit selection: {folder}')
        cell = dict(benchmark=benchmark, framework=framework, implementation=implementation,
                    exit_code=int(code), rows=[], source_files=[])
        if manifests:
            path = manifests[0]
            manifest = read(path)
            cell.update({k: manifest[k] for k in ['golden', 'framework_version', 'implementation_sources',
                         'validation_policy', 'environment', 'status', 'run_id', 'arguments',
                         'started_utc', 'finished_utc'] if k in manifest})
            cell['source_files'].append(str(path.relative_to(run)))
            cell['processes'] = manifest['processes']
            for process in manifest['processes']:
                row_path = path.parent / process['samples']
                sources[str(row_path.relative_to(run))] = digest(row_path)
                cell['source_files'].append(str(row_path.relative_to(run)))
                for line in row_path.read_text().splitlines():
                    raw = json.loads(line)
                    # Preserve scalar observations, identity, failure and placement;
                    # artifact identities are compact digests plus unchanged verdicts.
                    row = {k: v for k, v in raw.items() if k not in
                           ['resource_observation', 'launch_resources', 'artifacts']}
                    resource = raw.get('resource_observation', {})
                    row['resource_summary'] = {k: v for k, v in resource.items()
                        if k in ['affinity', 'affinity_scope', 'numa_policy', 'cpu_count', 'gpu_count']}
                    affinities = resource.get('thread_affinities', {})
                    row['resource_summary']['thread_count_observed'] = len(affinities)
                    row['resource_summary']['thread_affinity_sets'] = sorted({tuple(v) for v in affinities.values()})
                    artifacts = raw.get('artifacts', {})
                    before, after = artifacts.get('before', {}), artifacts.get('after', {})
                    row['artifact_summary'] = dict(
                        before_manifest_sha256=hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest(),
                        after_manifest_sha256=hashlib.sha256(json.dumps(after, sort_keys=True).encode()).hexdigest(),
                        files_before=len(before), files_after=len(after), unchanged=before == after)
                    cell['rows'].append(row)
        else:
            assert code == '125', (folder, code)
            cell['status'] = 'missing_source'
        if code == '0':
            assert len(cell['rows']) == 9
            assert all(x['status'] == 'passed' and x['validated'] is True for x in cell['rows'])
            assert Counter(x['phase'] for x in cell['rows']) == {'initialization':1, 'fresh_process':2, 'same_process':6}
        cells.append(cell)
    assert len({(c['benchmark'],c['framework'],c['implementation']) for c in cells}) == len(cells)
    (output / 'cells.json.gz').write_bytes(gzip.compress(json.dumps(cells, separators=(',', ':'), allow_nan=False).encode(), mtime=0))
    # Collection contains software/resource declarations, not workload data.
    dump(output / 'collection.json', collection)
    for name in ['plan-summary.json', 'plan.tsv', 'progress.tsv', 'cova-version.txt', 'preflight.jsonl']:
        source = run / name
        sources[name] = digest(source)
        shutil.copyfile(source, output / name)
    (output / 'analysis').mkdir()
    for name in ['summary.json', 'lifecycles.csv']:
        source = run / 'analysis' / name
        sources['analysis/' + name] = digest(source)
        shutil.copyfile(source, output / 'analysis' / name)
    failures = read(run / 'analysis' / 'failures.json')
    dump(output / 'analysis' / 'failures.json', [{k:x[k] for k in ['benchmark','framework','exit_code','failures','device','role'] if k in x} for x in failures])
    dump(output / 'provenance.json', dict(source_collection=str(run.resolve()), source_sha256=sources,
         extraction='Selected scalar fields; artifact inventories hashed; duplicate thread affinities collapsed. No arrays/binaries copied.'))
    dump(output / 'summary.json', dict(collection_id=run.name, raw_cells=len(cells),
         exit_counts=dict(Counter(str(c['exit_code']) for c in cells)),
         source_version=collection['cova_revision'], resource_policy='system',
         metric='host-to-host; median of process medians; phases separate',
         qualification='reference_only; GPU placement metadata is not an independent trace'))
    print(json.dumps({'cells': len(cells), 'output':str(output)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    extract(args.run_dir, args.output)
