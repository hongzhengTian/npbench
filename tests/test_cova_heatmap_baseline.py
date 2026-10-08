"""Frozen baseline generation: repairs, failures, weighting and identities."""
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'heatmap/scripts'))
from extract_baseline_collection import build_payload


def entry(name='compute', fw='numpy', impl='default', code=0, times=None):
    rows = [dict(process_index=p, call_index=i, phase=phase, status='passed',
        validated=True, metric_version=3, protocol_version=6, preset='L',
        validation_contract='strict_region_v1', golden_key='k', golden_sha256='h', time=t)
        for p,i,phase,t in (times or [(0,0,'initialization',10.),(1,0,'fresh_process',3.),(0,1,'same_process',1.),(0,2,'same_process',3.),(1,1,'same_process',8.)])]
    return dict(benchmark=name, framework=fw, implementation=impl, exit_code=code,
        rows=rows, host='fixture', golden={'key':'k','sha256':'h'})


class BaselineTests(unittest.TestCase):
    def build(self, main, repairs=()):
        return build_payload(main,list(repairs),dataset_id='fixture')

    def test_rectangular_equal_process_weight(self):
        result=self.build([entry()])
        self.assertEqual(len(result['columns']),14)
        self.assertEqual(len(result['cells']),42)
        cell=next(c for c in result['cells'] if c['framework']=='numpy' and c['phase']=='same_process')
        self.assertEqual(cell['seconds'],5.)
        self.assertEqual(cell['initialization_scope'],'single_main_collection_empty_artifact_call')

    def test_repairs_override_without_minimum_and_adi_never_original(self):
        main=[entry('adi'),entry('correlation','cupy')]
        repair=entry('correlation_return','cupy');repair['rows'][0]['time']=100.
        result=self.build(main,[entry('adi_corrected'),repair])
        self.assertNotIn('adi',result['benchmarks'])
        corrected=[c for c in result['cells'] if c['benchmark']=='adi_corrected' and c['state']=='observed']
        self.assertTrue(all(c['source_benchmark']=='adi_corrected' for c in corrected))
        cell=next(c for c in result['cells'] if c['framework']=='cupy' and c['benchmark']=='correlation' and c['phase']=='initialization')
        self.assertEqual(cell['seconds'],100.)

    def test_failed_cell_discards_successful_prefix(self):
        e=entry(code=1);e['rows'].append(dict(status='validation_failed'))
        cells=self.build([e])['cells']
        self.assertTrue(all(c['seconds'] is None for c in cells))
        self.assertEqual({c['state'] for c in cells},{'N/A','validation'})

    def test_partial_object_mode_has_no_fresh_value(self):
        e=entry(fw='numba',impl='object-mode',code=2)
        e['rows']=[r for r in e['rows'] if r['process_index']==0]
        cells=[c for c in self.build([e])['cells'] if c['implementation']=='object-mode']
        fresh=next(c for c in cells if c['phase']=='fresh_process')
        self.assertEqual(fresh['state'],'unsupported');self.assertIsNone(fresh['seconds'])
        self.assertEqual(next(c for c in cells if c['phase']=='same_process')['seconds'],2.)

    def test_reject_ambiguous_hosts_and_wrong_golden(self):
        with self.assertRaises(ValueError): self.build([entry(),entry()])
        other=entry('atax');other['host']='other'
        with self.assertRaises(ValueError): self.build([entry(),other])
        repair=entry('correlation_return','cupy');repair['golden']['key']='wrong'
        with self.assertRaises(AssertionError):self.build([entry('correlation','cupy')],[repair])

    def test_partial_other_route_not_success(self):
        e=entry(code=2)
        self.assertTrue(all(c['seconds'] is None for c in self.build([e])['cells']))


if __name__=='__main__':unittest.main()
