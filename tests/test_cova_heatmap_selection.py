"""Focused checks for lifecycle aggregation and exclusion of failed observations."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from plot_baseline_with_cova import aggregate, lifecycle_cells


class HeatmapSelectionTests(unittest.TestCase):
    def test_processes_have_equal_weight(self):
        seconds, medians = aggregate([
            {'process_index': 0, 'seconds': 1.0},
            {'process_index': 0, 'seconds': 9.0},
            {'process_index': 1, 'seconds': 100.0},
        ])
        self.assertEqual(medians, {'0': 5.0, '1': 100.0})
        self.assertEqual(seconds, 52.5)

    def test_failed_cohort_does_not_publish_partial_timings(self):
        rows = [{'process_index': 0, 'call_index': 0, 'phase': 'initialization',
                 'status': 'passed', 'validated': True, 'time': 1.0},
                {'status': 'validation_failed', 'phase': 'same_process'}]
        cells = lifecycle_cells('example', 'cova_llvm_cpu', rows, False, {}, 'numerical_failure')
        self.assertTrue(all(c['state'] == 'validation' and c['seconds'] is None for c in cells))
        self.assertTrue(all(c['samples'] == 0 for c in cells))

    def test_unvalidated_time_cannot_become_an_observation(self):
        row = {'status': 'passed', 'validated': False}
        with self.assertRaises(AssertionError):
            lifecycle_cells('example', 'cova_llvm_cpu', [row], True,
                            {'selected_source': 'fixture', 'source_files': []})

    def test_absent_phase_is_missing_not_zero(self):
        row = dict(process_index=0, call_index=0, phase='initialization', status='passed',
                   validated=True, time=2.0, metric_version=3, validation_contract='strict_region_v1',
                   golden_key='key', golden_sha256='hash')
        cells = lifecycle_cells('example', 'cova_llvm_cpu', [row], True, {})
        self.assertEqual([c['seconds'] for c in cells], [2.0, None, None])
        self.assertEqual([c['state'] for c in cells], ['observed', 'missing', 'missing'])


if __name__ == '__main__':
    unittest.main()
