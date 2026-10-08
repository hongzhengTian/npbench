"""Full-collection selection must preserve failures and corrected identities."""
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'heatmap' / 'scripts'))
from plot_cova_collection import build_payload
from plot_baseline_with_cova import COVA_COLUMNS, verify_merge
from plot_baseline_heatmaps import PHASES

class CollectionTests(unittest.TestCase):
    def test_helper_failure_is_not_replaced_by_successful_original(self):
        baseline = dict(dataset_id='fixture', benchmarks=['azimint_hist'],
            columns=[['numpy','default','NumPy']], cells=[dict(benchmark='azimint_hist',framework='numpy',implementation='default',phase=p,seconds=1.,state='observed',restricted=False,golden_key='k',golden_sha256='h') for p in PHASES])
        entries=[]
        for fw,_,_ in COVA_COLUMNS:
            rows=[dict(process_index=0,call_index=i,phase=p,status='passed',validated=True,time=2.,metric_version=3,validation_contract='strict_region_v1',golden_key='k',golden_sha256='h') for i,p in enumerate(PHASES)]
            entries.append(dict(benchmark='azimint_hist',framework=fw,exit_code=0,rows=rows,source_files=[],golden={'key':'k','sha256':'h'}))
        entries.append(dict(benchmark='azimint_hist_cova_helper',framework='cova_openmp_gpu',exit_code=1,rows=[{'status':'error','failure_stage':'execute','error':'fixture'}],source_files=[]))
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            (root/'cells.json.gz').write_bytes(gzip.compress(json.dumps(entries).encode()))
            (root/'summary.json').write_text(json.dumps(dict(source_version='fixture',collection_id='fixture')))
            result=build_payload(baseline,root)
        self.assertEqual(result['cells'][:3],baseline['cells'])
        gpu=[c for c in result['cells'] if c['framework']=='cova_openmp_gpu']
        self.assertTrue(all(c['seconds'] is None and c['state']=='error' for c in gpu))
        self.assertTrue(all(c['source_benchmark']=='azimint_hist_cova_helper' for c in gpu))
        verify_merge(result,baseline)

    def test_success_with_wrong_golden_is_rejected(self):
        baseline=dict(dataset_id='fixture',benchmarks=['adi_corrected'],columns=[],cells=[dict(benchmark='adi_corrected',framework='numpy',phase='initialization',golden_key='corrected',golden_sha256='corrected')])
        entries=[]
        for fw,_,_ in COVA_COLUMNS:
            row=dict(process_index=0,call_index=0,phase='initialization',status='passed',validated=True,time=1.,metric_version=3,validation_contract='strict_region_v1',golden_key='original',golden_sha256='original')
            entries.append(dict(benchmark='adi_corrected',framework=fw,exit_code=0,rows=[row],source_files=[],golden={'key':'original','sha256':'original'}))
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            (root/'cells.json.gz').write_bytes(gzip.compress(json.dumps(entries).encode()))
            (root/'summary.json').write_text(json.dumps(dict(source_version='fixture',collection_id='fixture')))
            with self.assertRaises(AssertionError):build_payload(baseline,root)

    def test_host_identity_and_binding_provenance(self):
        baseline=dict(dataset_id='fixture',host='same',gpu_numa_binding='off',
            benchmarks=['compute'],columns=[],cells=[dict(benchmark='compute',framework='numpy',phase='initialization',seconds=1.,golden_key='k',golden_sha256='h')])
        entries=[dict(benchmark='compute',framework=fw,exit_code=125,rows=[],source_files=[]) for fw,_,_ in COVA_COLUMNS]
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            (root/'cells.json.gz').write_bytes(gzip.compress(json.dumps(entries).encode()))
            summary=dict(source_version='fixture',collection_id='fixture',host='same',gpu_numa_binding='cpu-memory')
            (root/'summary.json').write_text(json.dumps(summary))
            result=build_payload(baseline,root)
            self.assertEqual(result['cova_gpu_numa_binding'],'cpu-memory')
            self.assertEqual(result['baseline_gpu_numa_binding'],'off')
            self.assertTrue(all(c['gpu_numa_binding']==('cpu-memory' if c['framework'].endswith('_gpu') else 'off') for c in result['cells'][1:]))
            summary['host']='different'
            (root/'summary.json').write_text(json.dumps(summary))
            with self.assertRaises(AssertionError):build_payload(baseline,root)

if __name__=='__main__':unittest.main()
