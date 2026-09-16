import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import tarfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('analysis',ROOT/'scripts/analyze_collection.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


def fixture(framework='numpy',repeat=2,fresh=2):
    manifest={'arguments':{'repeat':repeat,'fresh_process_runs':fresh},'status':'passed','golden':{'key':'golden'},'metric_version':3,'protocol_version':5,'validation_contract':'strict_region_v1','processes':[]}
    rows=[]
    for p in range(fresh+1):
        for c in range(repeat+1):
            phase=('initialization' if p==0 else 'fresh_process') if c==0 else 'same_process'
            rows.append(dict(process_index=p,call_index=c,phase=phase,time=float(p+1),validated=True,status='passed',golden_key='golden',metric_version=3,protocol_version=5,validation_contract="strict_region_v1",launch_resources={"affinity":[0]}))
    cell={'collection':'/nonexistent','phase':'baselines','benchmark':'example','framework':framework,'implementation':'default','exit_code':0,'reason':'executed','availability':'present','manifest':manifest,'records':rows,'attempt_count':1,'step_wall_seconds':None}
    return {'contract':{'measurement_role':'main','metric_version':3,'protocol_version':5,'environment':{},'affinity':[0],'comparison_contract':{'cpu':'same'}},'preflight':[{'observed_competition':False,'gpu_occupancy_verified':True}],'cells':[cell]}


class AnalysisTest(unittest.TestCase):
    def test_best_observation_carries_uncertainty_and_does_not_certify(self):
        b=fixture(); candidate=fixture('cova_llvm_cpu')
        for row in candidate['cells'][0]['records']: row['time'] *= .9
        best=next(r for r in a.analyze([b,candidate])[2] if r['phase']=='same_process')
        self.assertEqual(best['baseline_observed_processes'],3)
        self.assertEqual(best['baseline_observed_samples'],6)
        self.assertTrue(best['baseline_process_variability_review_required'])
        self.assertTrue(best['within_15_percent_point_estimate'])
        self.assertIsNone(best['within_15_percent_confirmed'])
        self.assertEqual(best['baseline_confirmation_status'],'pending_independent_review')
        self.assertIn('no_interval_isolation_evidence',best['baseline_resource_evidence_limits'])

    def test_single_process_partial_is_not_reported_as_stable(self):
        b=fixture('numba');cell=b['cells'][0]
        cell['records']=cell['records'][:3]+[dict(process_index=1,call_index=0,phase='fresh_process',status='reuse_unsupported',golden_key='golden',metric_version=3,protocol_version=5)]
        cell['manifest']['status']='partial';cell['exit_code']=2
        best=next(r for r in a.analyze([b])[2] if r['phase']=='same_process')
        self.assertEqual(best['baseline_eligibility'],'observed_pass_process0_only')
        self.assertEqual(best['baseline_observed_processes'],1)
        self.assertIsNone(best['baseline_process_max_min_ratio'])
        self.assertTrue(best['baseline_precision_review_required'])
        self.assertIn('process0_only',best['baseline_sampling_review'])
        self.assertFalse(best['baseline_process_variability_review_required'])

    def test_low_dispersion_still_requires_confirmation_and_cold_repetition(self):
        b=fixture()
        for row in b['cells'][0]['records']:row['time']=1.
        best=a.analyze([b])[2]
        same=next(r for r in best if r['phase']=='same_process')
        cold=next(r for r in best if r['phase']=='initialization')
        self.assertFalse(same['baseline_precision_review_required'])
        self.assertEqual(same['baseline_confirmation_status'],'pending_independent_review')
        self.assertIn('single_initialization_observation',cold['baseline_sampling_review'])

    def test_export_retains_build_configuration_text_without_native_binaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.mkdir()
            (source/'dace.conf').write_text('compiler: cpu\n')
            (source/'build.make').write_text('target: source.cpp\n')
            (source/'lib.so').write_bytes(b'not a text artifact')
            b=fixture();b['cells'][0]['collection']=str(source)
            b['cells'][0]['manifest']['golden']['path']=str(root/'absent-golden-payload')
            a.export([b],[source],root/'evidence')
            with tarfile.open(root/'evidence/raw-0.tar.gz') as tar:
                self.assertEqual(set(tar.getnames()),{'dace.conf','build.make'})

    def test_nine_calls_grouped_by_process_and_validation_failure_excludes_early_passes(self):
        b=fixture();cell=b['cells'][0]
        table,*_=a.analyze([b]);same=next(r for r in table if r['phase']=='same_process')
        self.assertEqual(same['observed_samples'],6);self.assertEqual(same['median_seconds'],2.)
        self.assertEqual(json.loads(same['process_medians_json']),{'0':1.,'1':2.,'2':3.})
        self.assertEqual(len(a.analyze([b])[2]),3)
        missing=copy.deepcopy(b)
        for row in missing['cells'][0]['records']:row.pop('launch_resources')
        self.assertEqual(a.analyze([missing])[2],[])
        cell['records'][-1].update(status='validation_failed',validated=False,validation_failures=[{'reason':'values'}])
        cell['exit_code']=1;cell['manifest']['status']='failed'
        self.assertEqual(set(a.eligibility(cell['records'],cell['manifest'],1).values()),{'excluded_validation_failure'})
        self.assertEqual(a.analyze([b])[2],[])

    def test_reuse_unsupported_and_restore_failure_preserve_only_original_process(self):
        cell=fixture()['cells'][0]
        cell['records']=cell['records'][:3]+[dict(process_index=1,call_index=0,phase='fresh_process',status='reuse_unsupported',golden_key='golden',metric_version=3,protocol_version=5)]
        cell['manifest']['status']='partial'
        result=a.eligibility(cell['records'],cell['manifest'],2)
        self.assertEqual(result['fresh_process'],'unsupported');self.assertEqual(result['same_process'],'observed_pass_process0_only')
        cell['records'][-1].update(status='error',failure_stage='restore')
        self.assertEqual(a.eligibility(cell['records'],cell['manifest'],1)['fresh_process'],'failed_restore')

    def test_partial_attempt_wrong_identity_and_duplicate_are_not_passes(self):
        cell=fixture()['cells'][0]
        self.assertEqual(set(a.eligibility(cell['records'][:-1],cell['manifest'],0).values()),{'diagnostic_only'})
        cell['records'][0]['golden_key']='other'
        self.assertEqual(set(a.eligibility(cell['records'],cell['manifest'],0).values()),{'inconsistent_identity'})
        with self.assertRaisesRegex(ValueError,'Duplicate cell'):a.analyze([fixture(),fixture()])
        small=fixture();large=fixture()
        small['cells'][0]['manifest']['arguments']['preset']='S'
        large['cells'][0]['manifest']['arguments']['preset']='L'
        self.assertEqual(len(a.analyze([small,large])[0]),6)
        c1=fixture('cova_llvm_cpu');c2=fixture('cova_llvm_cpu')
        c1['cells'][0]['manifest']['golden']['sha256']='first'
        c2['cells'][0]['manifest']['golden']['sha256']='different'
        with self.assertRaisesRegex(ValueError,'changed golden'):a.analyze([c1,c2],replace_cova=True)

        cell=fixture()['cells'][0];cell['records'][0]['validation_contract']='relaxed'
        self.assertEqual(set(a.eligibility(cell['records'],cell['manifest'],0).values()),{'inconsistent_identity'})
        b=fixture();b['contract']['affinity']=[0,1]
        b['cells'][0]['records'][0]['launch_resources']={'affinity':[0]}
        self.assertEqual({r['eligibility'] for r in a.analyze([b])[0]}, {'excluded_resource_mismatch'})


    def test_explicit_cova_replacement_is_whole_cell_and_cold_is_not_main_ranking(self):
        old=fixture('cova_llvm_cpu');new=fixture('cova_llvm_cpu');new['cells'][0]['collection']='/later'
        result=a.analyze([old,new],replace_cova=True)
        self.assertEqual(len(result[-1]),1);self.assertEqual({r['collection'] for r in result[0]},{'/later'})
        trials=[]
        for i in range(3):
            b=fixture(repeat=0,fresh=0);b['contract']['measurement_role']='cold';b['cells'][0]['collection']='/trial'+str(i);trials.append(b)
        _,_,best,cold,_=a.analyze(trials)
        self.assertEqual(best,[]);self.assertTrue(cold[0]['minimum_repetition_met'])

    def test_portable_export_rebuild_and_corruption_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);b=fixture()
            b['cells'][0]['records'][-1].update(status='validation_failed',validated=False,validation_failures=[{'reason':'values','field':'returns[0]'}])
            b['cells'][0]['exit_code']=1;b['cells'][0]['manifest']['status']='failed'
            a.export([b],[root/'source'],root/'evidence')
            loaded=a.load(root/'evidence');self.assertEqual(a.analyze([b]),a.analyze(loaded))
            before={str(p):a.digest(p) for p in (root/'evidence').rglob('*') if p.is_file()}
            a.write_report(loaded,root/'rebuilt')
            for path in (root/'evidence/analysis').iterdir():
                self.assertEqual(path.read_bytes(),(root/'rebuilt'/path.name).read_bytes(),path.name)
            self.assertEqual(before,{str(p):a.digest(p) for p in (root/'evidence').rglob('*') if p.is_file()})
            path=root/'evidence/collection-0.json.gz';path.write_bytes(path.read_bytes()+b'bad')
            with self.assertRaisesRegex(ValueError,'integrity'):a.load(root/'evidence')

if __name__=='__main__':unittest.main()
