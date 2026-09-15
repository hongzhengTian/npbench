"""Conservative, evidence-based selection for the protocol-3 infrastructure retry."""
import hashlib
import json
from pathlib import Path

MISSING = 'FileNotFoundError: No artifacts remain for the fresh-process reuse measurement'


def select(source):
    source = Path(source).resolve()
    evidence = {}

    def read(path):
        data = path.read_bytes()
        evidence[str(path.relative_to(source))] = hashlib.sha256(data).hexdigest()
        return data.decode()

    collection = json.loads(read(source / 'collection.json'))
    if collection.get('protocol_version', 2) >= 3:
        raise ValueError('Expected the original pre-fix collection, not a protocol-3 retry')
    cells = []
    for line in read(source / 'plan.tsv').splitlines():
        phase, bench, framework, label, availability = line.split('\t')
        if phase != 'baselines' or framework not in ('numba', 'dace_cpu', 'dace_gpu'):
            continue
        if any(Path(part).name != part or part in ('.', '..') for part in (bench, framework, label)):
            raise ValueError('Unsafe cell path')
        cell = source / phase / bench / framework / label
        receipt = cell / 'exit-code.txt'
        if not receipt.exists() or read(receipt).strip() == '0' or availability != 'present':
            continue
        records = [json.loads(line) for path in sorted(cell.glob('runs/*/*/process-*.jsonl'))
                   for line in read(path).splitlines() if line.strip()]
        failed = [row for row in records if row['status'] != 'passed']
        reason = None
        if len(failed) == 1:
            row = failed[0]
            if (row.get('error') == MISSING and row.get('failure_stage') == 'prepare'
                    and row.get('phase') == 'fresh_process'
                    and any(x['status'] == 'passed' and x.get('phase') == 'initialization' for x in records)):
                reason = 'dace_cache_location' if framework.startswith('dace_') else 'numba_cache_capability'
            # These two scalar-return ABI failures were established by focused
            # probes. Do not broaden this to arbitrary old validation failures:
            # they lack shape diagnostics and may be algorithmic/numerical bugs.
            elif (framework.startswith('dace_') and bench in ('crc16', 'channel_flow')
                  and row['status'] == 'validation_failed' and row.get('phase') == 'initialization'):
                reason = 'dace_scalar_return_abi'
        if reason:
            cells.append([bench, framework, label, reason])
    return dict(source=str(source), preset=collection['environment']['NPBENCH_PRESET'],
                cells=cells, evidence=evidence)


if __name__ == '__main__':
    import sys
    result = select(Path(sys.argv[1]))
    for row in result['cells']:
        print('\t'.join(row))
    print(f"Selected {len(result['cells'])} failed baseline implementations; preset {result['preset']}")
