import json
from pathlib import Path
import tempfile
import unittest

from scripts.retry_affected import MISSING, select


class RetrySelectionTest(unittest.TestCase):
    def test_excludes_success_unrelated_and_cova_and_tracks_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'collection.json').write_text(json.dumps({'environment': {'NPBENCH_PRESET': 'L'}}))
            plan = []
            passed = dict(status='passed', phase='initialization')
            missing = dict(status='error', phase='fresh_process', error=MISSING, failure_stage='prepare')
            invalid = dict(status='validation_failed', phase='initialization')
            cases = [
                ('gemm', 'dace_cpu', 1, [passed, missing]),
                ('gemm', 'numba', 1, [passed, missing]),
                ('crc16', 'dace_gpu', 1, [invalid]),
                ('channel_flow', 'dace_cpu', 1, [invalid]),
                ('gemm', 'dace_gpu', 0, [passed, missing]),
                ('adi', 'dace_cpu', 1, [invalid]),
                ('crc16', 'numba', 1, [invalid]),
                ('crc16', 'cova_llvm_cpu', 1, [invalid]),
                ('nbody', 'dace_cpu', 1, [dict(missing, error='upstream error')]),
                ('mlp', 'dace_cpu', 1, [dict(missing, phase='initialization')]),
            ]
            for bench, fw, rc, rows in cases:
                phase = 'cova' if fw.startswith('cova_') else 'baselines'
                plan.append(f'{phase}\t{bench}\t{fw}\tdefault\tpresent\n')
                cell = root/phase/bench/fw/'default'
                worker = cell/'runs/run/default'; worker.mkdir(parents=True)
                (cell/'exit-code.txt').write_text(str(rc))
                (worker/'process-0.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
            (root/'plan.tsv').write_text(''.join(plan))
            result = select(root)
            self.assertEqual([(b, f) for b, f, _, _ in result['cells']],
                             [('gemm', 'dace_cpu'), ('gemm', 'numba'),
                              ('crc16', 'dace_gpu'), ('channel_flow', 'dace_cpu')])
            self.assertEqual(result['preset'], 'L')
            self.assertIn('plan.tsv', result['evidence'])
            (root/'collection.json').write_text(json.dumps({'protocol_version': 3}))
            with self.assertRaises(ValueError): select(root)
