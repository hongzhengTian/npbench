#!/usr/bin/env python3
"""Portable NPBench lifecycle analysis/export; standard library only, no kernels."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import tarfile

PHASES = ('initialization', 'fresh_process', 'same_process')
TEXT_SUFFIXES = {'.log', '.txt', '.tsv', '.csv', '.json', '.jsonl', '.sh', '.md',
                 '.patch', '.py', '.c', '.cpp', '.cu', '.cuh', '.h', '.hpp',
                 '.sdfg', '.ll', '.mlir', '.cmake', '.conf', '.make', '.ptx',
                 '.yaml', '.yml', '.toml', '.ini', '.cfg', '.rst'}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def succeeded(row):
    value = row.get('time')
    return (row.get('status') == 'passed' and row.get('validated') is True
            and isinstance(value, (int, float)) and math.isfinite(value) and value > 0)


def eligibility(rows, manifest, exit_code):
    result = dict.fromkeys(PHASES, 'unavailable')
    if not rows or not manifest: return result
    if any(row.get('status') == 'validation_failed' for row in rows):
        return dict.fromkeys(PHASES, 'excluded_validation_failure')
    if any(row.get('status') == 'passed' and not succeeded(row) for row in rows):
        return dict.fromkeys(PHASES, 'invalid_pass_record')
    for row in rows:
        for field in ('metric_version', 'protocol_version', 'validation_contract', 'golden_key', 'golden_sha256'):
            expected = manifest.get('golden', {}).get('key' if field == 'golden_key' else 'sha256') if field.startswith('golden_') else manifest.get(field)
            # Early protocol-5 diagnostic records omitted the per-call hash;
            # their manifest still supplies a verified bundle fingerprint.
            if field == 'golden_sha256' and field not in row: continue
            if expected is not None and row.get(field) != expected and (row.get('status') == 'passed' or row.get(field) is not None):
                return dict.fromkeys(PHASES, 'inconsistent_identity')
    positions = [(r.get('process_index'), r.get('call_index')) for r in rows]
    if len(set(positions)) != len(positions): return dict.fromkeys(PHASES, 'duplicate_records')
    for r in rows:
        if r.get('call_index') is None: continue
        expected = ('initialization' if r['process_index'] == 0 else 'fresh_process') if r['call_index'] == 0 else 'same_process'
        if r.get('phase') != expected: return dict.fromkeys(PHASES, 'invalid_phase')
    count, fresh = manifest['arguments']['repeat'], manifest['arguments']['fresh_process_runs']
    expected = {(p, c) for p in range(fresh + 1) for c in range(count + 1)}
    if exit_code == 0 and manifest.get('status') == 'passed' and set(positions) == expected and all(map(succeeded, rows)):
        return {phase: 'observed_pass' if any(r['phase'] == phase for r in rows) else 'unavailable' for phase in PHASES}
    first = [r for r in rows if r.get('process_index') == 0]
    first_ok = (len(first) == count + 1 and {r.get('call_index') for r in first} == set(range(count + 1)) and all(map(succeeded, first)))
    errors = [r for r in rows if r.get('status') not in ('passed', 'reuse_unsupported')]
    unsupported = [r for r in rows if r.get('status') == 'reuse_unsupported']
    if first_ok and not errors and unsupported and manifest.get('status') == 'partial' and exit_code == 2:
        result.update(initialization='observed_pass', same_process='observed_pass_process0_only' if count else 'unavailable', fresh_process='unsupported')
    elif first_ok and errors and all(r.get('phase') == 'fresh_process' and r.get('failure_stage') == 'restore' for r in errors):
        result.update(initialization='observed_pass', same_process='observed_pass_process0_only' if count else 'unavailable', fresh_process='failed_restore')
    else:
        result = dict.fromkeys(PHASES, 'diagnostic_only')
    return result


def failure_categories(rows, reason, exit_code):
    result = set()
    if reason != 'executed': result.add(reason)
    for row in rows:
        status = row.get('status')
        if status == 'passed': continue
        if status == 'validation_failed':
            for failure in row.get('validation_failures', []):
                why = failure.get('reason')
                result.add('output_contract_' + why if why in ('count', 'keys', 'shape', 'dtype') else 'numerical_' + str(why))
        elif status == 'error':
            result.add('error_at_' + str(row.get('failure_stage', 'unknown')))
        else:
            result.add(str(status))
    if exit_code is None: result.add('not_completed')
    elif exit_code != 0 and not result: result.add('process_failure_without_record')
    return sorted(result)


def load_collection(root):
    contract = read_json(root / 'collection.json')
    preflight = [json.loads(line) for line in (root/'preflight.jsonl').read_text().splitlines()] if (root/'preflight.jsonl').exists() else []
    cells = []
    for line in (root / 'plan.tsv').read_text().splitlines():
        phase, benchmark, framework, impl, availability = line.split('\t')
        folder = root / phase / benchmark / framework / impl
        receipt = folder / 'exit-code.txt'
        code = int(receipt.read_text()) if receipt.exists() else None
        reason = (folder/'reason.txt').read_text().strip() if (folder/'reason.txt').exists() else ('not_started' if availability == 'present' else availability)
        manifests = list((folder/'runs').glob('*/manifest.json'))
        selected = folder/'selected-run.txt'
        if selected.exists():
            run_id = selected.read_text().strip()
            if '/' in run_id or run_id in ('', '.', '..'): raise ValueError('Invalid selected run')
            manifests = [folder/'runs'/run_id/'manifest.json']
        manifest = read_json(manifests[0]) if len(manifests) == 1 and manifests[0].exists() else None
        rows = []
        if manifest:
            for path in sorted(manifests[0].parent.glob('*/process-*.jsonl')):
                rows.extend(json.loads(line) for line in path.read_text().splitlines())
        item = {'collection': str(root), 'phase': phase, 'benchmark': benchmark, 'framework': framework,
                'implementation': impl, 'exit_code': code, 'reason': reason, 'availability': availability,
                'manifest': manifest, 'records': rows, 'attempt_count': len(list((folder/'runs').glob('*/manifest.json'))),
                'step_wall_seconds': float((folder/'step-wall-seconds.txt').read_text()) if (folder/'step-wall-seconds.txt').exists() else None}
        if len(manifests) > 1: item['reason'] = 'ambiguous_attempts'
        cells.append(item)
    return {'contract': contract, 'preflight': preflight, 'cells': cells}


def load(root):
    if (root/'evidence-index.json').exists():
        index = read_json(root/'evidence-index.json')
        for name, checksum in index['checksums'].items():
            path = (root/name).resolve()
            if not path.is_relative_to(root.resolve()) or digest(path) != checksum: raise ValueError('Evidence integrity failure: ' + name)
        return [json.loads(gzip.decompress((root/name).read_bytes())) for name in index['bundles']]
    if (root/'collection.json').exists(): return [load_collection(root)]
    children = sorted(path for path in root.rglob('collection.json') if not {'source-snapshots', 'raw', 'analysis'} & set(path.relative_to(root).parts))
    if not children: raise ValueError('No collection or evidence found: ' + str(root))
    return [load_collection(path.parent) for path in children]


def comparison_key(contract, device):
    common = dict(contract.get('comparison_contract', {}))
    if not common: return 'legacy:' + str(contract.get('npbench_revision'))
    if device == 'cpu': common.pop('gpu', None)
    return hashlib.sha256(json.dumps(common, sort_keys=True).encode()).hexdigest()


def analyze(bundles, *, replace_cova=False):
    selected = {}; replaced = []
    for bundle in bundles:
        contract = bundle['contract']; role = contract.get('measurement_role', 'legacy')
        for cell in bundle['cells']:
            if cell['phase'] == 'golden': continue
            # Independent cold trials stay independent and are summarized below.
            trial = cell['collection'] if role == 'cold' else ''
            device = 'gpu' if cell['framework'] == 'cupy' or cell['framework'].endswith('_gpu') else 'cpu'
            preset = (cell['manifest'] or {}).get('arguments', {}).get('preset', contract.get('environment', {}).get('NPBENCH_PRESET'))
            key = (role, comparison_key(contract, device), cell['benchmark'], cell['framework'], cell['implementation'], preset, trial)
            if key in selected:
                if not (replace_cova and cell['framework'].startswith('cova_')):
                    raise ValueError('Duplicate cell; select one collection or explicitly --replace-cova: ' + repr(key))
                old_golden = (selected[key][0]['manifest'] or {}).get('golden', {})
                new_golden = (cell['manifest'] or {}).get('golden', {})
                if old_golden and new_golden and any(old_golden.get(k) != new_golden.get(k) for k in ('key', 'sha256')):
                    raise ValueError('CoVA replacement changed golden inputs; use separate comparisons')
                replaced.append({'key': key, 'old': selected[key][0]['collection'], 'new': cell['collection']})
            selected[key] = (cell, contract, bundle['preflight'], device)
    table = []; raw_failures = []
    for (role, identity, *unused), (cell, contract, preflight, device) in selected.items():
        manifest, rows = cell['manifest'], cell['records']
        eligible = eligibility(rows, manifest, cell['exit_code'])
        categories = failure_categories(rows, cell['reason'], cell['exit_code'])
        base = {k: cell[k] for k in ('collection', 'benchmark', 'framework', 'implementation', 'exit_code', 'attempt_count')}
        base.update(role=role, device=device, comparison_key=identity, failures=';'.join(categories),
                    preset=manifest.get('arguments', {}).get('preset', contract.get('environment', {}).get('NPBENCH_PRESET')) if manifest else contract.get('environment', {}).get('NPBENCH_PRESET'),
                    golden_key=manifest.get('golden', {}).get('key') if manifest else None,
                    golden_sha256=manifest.get('golden', {}).get('sha256') if manifest else None,
                    metric_version=contract.get('metric_version'), protocol_version=contract.get('protocol_version'),
                    validation_contract=contract.get('validation_contract'),
                    resource_policy=contract.get('environment', {}).get('NPBENCH_RESOURCE_POLICY', 'legacy_fixed'),
                    formal_comparison_eligibility='not_certified', step_wall_seconds=cell['step_wall_seconds'])
        policies = sorted({r['artifact_policy'] for r in rows if r.get('artifact_policy')})
        base['artifact_policies'] = ';'.join(policies)
        fallback = any(p.get('actual_device') == 'cpu' for r in rows for p in r.get('placement', [])) and device == 'gpu'
        warnings = []
        if fallback: warnings.append('gpu_cpu_fallback'); eligible = dict.fromkeys(PHASES, 'excluded_device_fallback')
        if not preflight: warnings.append('missing_resource_preflight')
        if any(p.get('observed_competition') for p in preflight): warnings.append('observed_resource_competition')
        if device == 'gpu' and (not preflight or not all(p.get('gpu_occupancy_verified') for p in preflight)):
            warnings.append('gpu_occupancy_unverified')
        if any(r.get('resource_observation', {}).get('probe_error') for r in rows): warnings.append('resource_probe_error')
        launched = [r['launch_resources']['affinity'] for r in rows if r.get('launch_resources', {}).get('affinity')]
        if not launched: warnings.append('missing_worker_launch_resources')
        if launched and any(cpus != contract.get('affinity') for cpus in launched):
            warnings.append('worker_allocation_mismatch')
            eligible = dict.fromkeys(PHASES, 'excluded_resource_mismatch')
        observed_gpu = [r['resource_observation']['gpu'].get('uuid') for r in rows if r.get('resource_observation', {}).get('gpu')]
        if device == 'gpu' and not observed_gpu: warnings.append('missing_worker_gpu_identity')
        if observed_gpu and any(uuid != (contract.get('gpu') or {}).get('uuid') for uuid in observed_gpu):
            warnings.append('worker_gpu_mismatch')
            eligible = dict.fromkeys(PHASES, 'excluded_device_mismatch')
        if cell['framework'].startswith('cova_') and device == 'gpu' and not any(r.get('placement') for r in rows):
            warnings.append('missing_cova_gpu_placement')

        base['resource_review'] = ';'.join(warnings)
        base['worker_wall_seconds'] = sum(p.get('worker_wall_seconds', 0) for p in manifest.get('processes', [])) if manifest else None
        for name in ('preparation_seconds', 'input_reset_seconds', 'validation_observation_seconds', 'artifact_audit_seconds'):
            base[name] = sum(r.get(name, 0) for r in rows)
        for phase in PHASES:
            observed = [r for r in rows if r.get('phase') == phase and succeeded(r)]
            grouped = defaultdict(list)
            for row in observed: grouped[row['process_index']].append(row['time'])
            medians = {str(k): statistics.median(values) for k, values in sorted(grouped.items())}
            summary = statistics.median(medians.values()) if medians else None
            ratio = max(medians.values())/min(medians.values()) if len(medians) > 1 else None
            qualification = eligible[phase]
            if 'process0_only' in qualification:
                summary = medians.get('0')
            if cell['exit_code'] is None: qualification = 'not_completed'
            sampling_review = []
            if phase == 'initialization' and medians:
                sampling_review.append('single_initialization_observation')
            elif phase == 'fresh_process' and len(medians) < 2:
                sampling_review.append('fewer_than_two_later_processes')
            elif phase == 'same_process' and len(medians) < 3:
                sampling_review.append('fewer_than_three_processes')
            if 'process0_only' in qualification:
                sampling_review.append('process0_only')
            variability_review = bool(ratio and ratio > 1.2)
            # Availability is not interval-wide isolation, and no statistical
            # confirmation is inferred from a lack of dispersion flags.
            resource_limits = ['no_interval_isolation_evidence']
            numa = contract.get('allocation', {}).get('numa_policy')
            if not numa or str(numa).startswith('unavailable'):
                resource_limits.append('numa_policy_unverified')
            row = dict(base, phase=phase, eligibility=qualification, observed_samples=len(observed),
                       observed_processes=len(medians),
                       samples_by_process_json=json.dumps({str(k): len(v) for k, v in sorted(grouped.items())}),
                       process_medians_json=json.dumps(medians), median_seconds=summary,
                       process_max_min_ratio=ratio, process_variability_review_required=variability_review,
                       sampling_review=';'.join(sampling_review),
                       resource_evidence_limits=';'.join(resource_limits),
                       precision_review_required=bool(sampling_review or variability_review),
                       confirmation_status='pending_independent_review',
                       initialization_single_sample=phase == 'initialization')
            table.append(row)
        if categories:
            raw_failures.append(dict(base, details=[r for r in rows if r.get('status') != 'passed'],
                                     cause_attribution='unresolved_unless_independently_documented'))
    best = []
    groups = defaultdict(list)
    for row in table:
        if row['role'] != 'main' or not row['eligibility'].startswith('observed_pass') or row['median_seconds'] is None: continue
        if row['resource_review']: continue
        groups[(row['comparison_key'], row['benchmark'], row['device'], row['phase'], row['golden_key'], row['golden_sha256'], row['preset'])].append(row)
    for key, values in groups.items():
        baselines = [r for r in values if not r['framework'].startswith('cova_')]
        cova = [r for r in values if r['framework'].startswith('cova_')]
        if not baselines: continue
        reference = min(baselines, key=lambda r: r['median_seconds'])
        candidate = min(cova, key=lambda r: r['median_seconds']) if cova else None
        ratio = candidate['median_seconds']/reference['median_seconds'] if candidate else None
        selected_row = {'comparison_key':key[0], 'benchmark':key[1], 'device':key[2], 'phase':key[3], 'golden_key':key[4], 'golden_sha256':key[5], 'preset':key[6],
                     'baseline':reference['framework']+'/'+reference['implementation'], 'baseline_seconds':reference['median_seconds'],
                     'cova_posthoc_best':candidate['framework']+'/'+candidate['implementation'] if candidate else None,
                     'cova_seconds':candidate['median_seconds'] if candidate else None, 'cova_time_ratio':ratio,
                     'within_15_percent_point_estimate':ratio <= 1.15 if ratio is not None else None,
                     'within_15_percent_confirmed':None,
                     'comparison_qualification':'descriptive_only_pending_confirmation',
                     'interpretation':'observed_posthoc_best_not_automatic_selection_or_certified_ranking',
                     'resource_review':';'.join(sorted({r['resource_review'] for r in values if r['resource_review']}))}
        for prefix, selected_value in [('baseline', reference), ('cova', candidate)]:
            for field in ('eligibility', 'observed_samples', 'observed_processes',
                          'samples_by_process_json', 'process_medians_json', 'process_max_min_ratio',
                          'process_variability_review_required', 'sampling_review',
                          'precision_review_required', 'resource_evidence_limits',
                          'confirmation_status', 'formal_comparison_eligibility'):
                selected_row[prefix + '_' + field] = selected_value[field] if selected_value else None
        best.append(selected_row)
    cold = []
    trials = defaultdict(list)
    for row in table:
        if row['role'] == 'cold' and row['phase'] == 'initialization':
            trials[(row['comparison_key'],row['benchmark'],row['framework'],row['implementation'],row['golden_key'],row['golden_sha256'],row['preset'])].append(row)
    for key, rows in trials.items():
        values = [r['median_seconds'] for r in rows if r['eligibility']=='observed_pass' and r['median_seconds'] is not None]
        cold.append({'comparison_key':key[0],'benchmark':key[1],'framework':key[2],'implementation':key[3],'golden_key':key[4],'golden_sha256':key[5],'preset':key[6],
                     'attempts':len(rows),'successful_trials':len(values),'samples':values,
                     'median_seconds':statistics.median(values) if values else None,
                     'minimum_repetition_met':len(values)>=3 and len(values)==len(rows)})
    return table, raw_failures, best, cold, replaced


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        if rows:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def write_report(bundles, destination, *, replace_cova=False):
    destination.mkdir(parents=True, exist_ok=True)
    table, failures, best, cold, replaced = analyze(bundles, replace_cova=replace_cova)
    write_csv(destination/'lifecycles.csv',table);write_csv(destination/'best-observed.csv',best)
    for name,data in [('failures.json',failures),('initialization-trials.json',cold),('replacements.json',replaced)]:
        (destination/name).write_text(json.dumps(data,indent=2,allow_nan=False,sort_keys=True)+'\n')
    cells=[c for b in bundles for c in b['cells'] if c['phase']!='golden']
    counts=Counter('not_completed' if c['exit_code'] is None else 'passed' if c['exit_code']==0 else 'partial' if c['exit_code']==2 else 'failed_or_blocked' for c in cells)
    selected_cells=[r for r in table if r['phase']=='initialization']
    selected_counts=Counter('not_completed' if r['exit_code'] is None else 'passed' if r['exit_code']==0 else 'partial' if r['exit_code']==2 else 'failed_or_blocked' for r in selected_cells)
    summary={'schema':2,'collections':len(bundles),'cell_attempts':len(cells),'terminal_counts':dict(counts),
             'selected_cells':len(selected_cells),'selected_terminal_counts':dict(selected_counts),
             'aggregate_speedup':None,'note':'No cross-configuration aggregate; inspect paired sets, coverage, resources and process grouping.'}
    (destination/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    qualification = []
    for device in ('cpu', 'gpu'):
        for phase in PHASES:
            selected = [row for row in best if row['device'] == device and row['phase'] == phase]
            qualification.append({'device': device, 'phase': phase, 'observed_best_count': len(selected),
                                  'variability_review_count': sum(row['baseline_process_variability_review_required'] for row in selected),
                                  'sampling_review_count': sum(bool(row['baseline_sampling_review']) for row in selected),
                                  'process0_only_count': sum(row['baseline_eligibility'] == 'observed_pass_process0_only' for row in selected),
                                  'confirmed_count': 0})
    (destination/'qualification-summary.json').write_text(json.dumps(qualification, indent=2, sort_keys=True)+'\n')
    lines=['# NPBench collection analysis','',json.dumps(summary,ensure_ascii=False),'',
           'Functional observations and conditional timing summaries; no automatic publication certification.',
           'same_process uses the median of per-process medians; raw process summaries remain in lifecycles.csv.',
           'A numerical failure excludes all phases. Unsupported persistence and restore-only failures are scoped explicitly.',
           'best-observed.csv is a post-hoc comparison within compatible resource/metric/golden groups, not an automatic selector result.',
           'Best rows retain eligibility, process/sample counts and review flags; one process has no between-process ratio.',
           'Precision flags include insufficient sampling and observed dispersion. No flag is not statistical confirmation.',
           'Collection preflight does not establish interval-wide isolation; confirmation remains pending for all rows.',
           'GPU occupancy gaps and other resource review flags must be resolved before strong performance claims.',
           'Single initialization observations are descriptive; independent cold trials are summarized separately.',
           'Detailed attribution requires original logs; error categories do not by themselves prove an upstream cause.','']
    (destination/'README.md').write_text('\n'.join(lines))
    return summary


def export(bundles, roots, destination):
    if destination.exists(): raise ValueError('Use a new evidence directory')
    if any(destination.resolve().is_relative_to(root.resolve()) for root in roots): raise ValueError('Evidence must be outside source collections')
    destination.mkdir(parents=True)
    index={'schema':1,'bundles':[],'checksums':{},'excluded_large_or_binary':[]}
    for number,bundle in enumerate(bundles):
        name=f'collection-{number}.json.gz'
        payload=gzip.compress(json.dumps(bundle,sort_keys=True,allow_nan=False).encode(),mtime=0)
        if len(payload)>50*1024*1024: raise ValueError('Compressed observation bundle exceeds 50 MiB')
        (destination/name).write_bytes(payload);index['bundles'].append(name)
        root=Path(bundle['cells'][0]['collection']) if bundle['cells'] else None
        if root and root.is_dir():
            # Pack text into one deterministic archive per collection instead
            # of duplicating thousands of small snapshot files in research Git.
            archive=destination/f'raw-{number}.tar.gz'
            with archive.open('wb') as stream, gzip.GzipFile(fileobj=stream,mode='wb',mtime=0,filename='') as zipped, tarfile.open(fileobj=zipped,mode='w|') as tar:
                for path in sorted(root.rglob('*')):
                    if not path.is_file() or (path.suffix not in TEXT_SUFFIXES and path.name not in ('CMakeLists.txt', 'Makefile')) or 'analysis' in path.relative_to(root).parts: continue
                    if path.stat().st_size>50*1024*1024:
                        index['excluded_large_or_binary'].append({'path':str(path),'sha256':digest(path),'size':path.stat().st_size});continue
                    info=tarfile.TarInfo(str(path.relative_to(root)));info.size=path.stat().st_size
                    info.mode=0o644;info.mtime=0
                    with path.open('rb') as src: tar.addfile(info,src)
            if archive.stat().st_size>50*1024*1024:
                raise ValueError('Compressed raw-text archive exceeds 50 MiB; export smaller collections')
            golden_paths={c['manifest']['golden']['path'] for c in bundle['cells'] if c['manifest']}
            for folder in golden_paths:
                metadata=Path(folder)/'golden.json'
                if metadata.is_file():
                    target=destination/'golden-metadata'/Path(folder).name/'golden.json';target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(metadata,target)
    shutil.copyfile(Path(__file__),destination/'analyze_collection.py')
    write_report(bundles,destination/'analysis')
    for path in sorted(destination.rglob('*')):
        if path.is_file(): index['checksums'][str(path.relative_to(destination))]=digest(path)
    (destination/'evidence-index.json').write_text(json.dumps(index,indent=2)+'\n')
    print('Portable evidence:',destination)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('collections',nargs='+',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--export',type=Path)
    parser.add_argument('--verify',action='store_true',help='Verify exported evidence without writing')
    parser.add_argument('--replace-cova',action='store_true',help='Explicitly replace earlier whole CoVA cells in the same comparison group')
    args=parser.parse_args()
    bundles=[bundle for root in args.collections for bundle in load(root)]
    if args.verify:
        if not all((root/'evidence-index.json').exists() for root in args.collections): raise ValueError('--verify requires exported evidence')
        analyze(bundles,replace_cova=args.replace_cova)
        print('Evidence integrity and records verified read-only');return
    if args.export:
        export(bundles,args.collections,args.export);return
    if (args.collections[0]/'evidence-index.json').exists() and args.output is None:
        raise ValueError('Evidence verified read-only; supply --output outside the archive to rebuild analysis')
    destination=args.output or args.collections[0]/'analysis'
    if any((root/'evidence-index.json').exists() and destination.resolve().is_relative_to(root.resolve()) for root in args.collections):
        raise ValueError('Do not overwrite frozen evidence')
    print(json.dumps(write_report(bundles,destination,replace_cova=args.replace_cova)))

if __name__=='__main__': main()
