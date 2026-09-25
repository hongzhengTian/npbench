#!/usr/bin/env bash
# Bounded diagnostic only. No production modifications, profiler, or competitor runs.
# Run: bash run_ordered_prepare_L.sh
set -euo pipefail
cd -- "$(dirname -- "$(realpath -- "$0")")"
export L_TEST_SCRIPT="$(realpath -- "$0")"
source /HSC/users/tianhong/myHome/developing/npbench/.cache/native-validation/cova-integration-20260912/activate-cova.sh
export NPBENCH_COVA_VERSION=e01a2c3f735d4ed56cd47b76e567f3ecf25ff7af
export COVA_LLVM_PREBUILD=/HSC/users/tianhong/myHome/developing/CoVA/externals/llvm_prebuild/cova-llvm-22.0.0git-6b09f739-linux-x86_64-py310-r1
export PYTHONPATH=/HSC/users/tianhong/myHome/developing/npbench:$PYTHONPATH
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMBA_NUM_THREADS=8
export NPBENCH_RESOURCE_POLICY=fixed NPBENCH_THREADS=8
export PYTHONDONTWRITEBYTECODE=1
export OMP_PROC_BIND=close OMP_PLACES=cores OMP_DYNAMIC=false
export NUMBA_THREADING_LAYER=omp
python - <<'COVA_L_DIAGNOSTIC_PY'
ASSETS = {'driver.py': "import os,sys,pathlib,json,time,hashlib\nimport numpy as np\nfrom contextlib import ExitStack\nfrom unittest.mock import patch\nfrom cova.orchestration.compilation import invocation\nfrom cova.runtime.native.extension import NativeExtensionRuntime\nfrom cova.contracts import compilation as compilation_contract\nfrom npbench.infrastructure.benchmark import Benchmark\nfrom npbench.infrastructure.framework import generate_framework\nfrom npbench.infrastructure.region import Region\nfrom npbench.infrastructure import golden\nname,framework,preset,process=sys.argv[1:5]\nif os.environ.get('STAGE2_BASELINE') == '1':\n compilation_contract.FRONTEND_SEMANTICS_VERSION = int(os.environ.get('BASE_SEMANTICS', '16'))\nbase=pathlib.Path(os.environ.get('DIAG_OUTPUT_DIR','/tmp/cova-stage3-20260923/evidence'))\nartifact=pathlib.Path(os.environ.get('DIAG_ARTIFACT', '/tmp/cova-stage3-20260923/artifacts/'+name+'-'+framework+'-'+preset))\nartifact.mkdir(parents=True,exist_ok=True);os.chdir(artifact)\ndef identities():\n return {str(p.relative_to(artifact)):hashlib.sha256(p.read_bytes()).hexdigest() for p in artifact.rglob('*.so')}\nbefore=identities()\nbench=Benchmark(name);bundle,event=golden.load_or_create(bench,preset,generate_framework('numpy'),'/HSC/users/tianhong/myHome/developing/npbench/.cache/goldens',required=True)\nwith ExitStack() as stack:\n if process!='bootstrap' and framework.startswith('cova'):\n  stack.enter_context(patch.object(invocation,'load_or_generate_cova_module',side_effect=AssertionError('forbidden frontend compile')))\n  stack.enter_context(patch.object(NativeExtensionRuntime,'compile',side_effect=AssertionError('forbidden native compile')))\n writebacks=set(bench.info.get('output_args',[]))|{k for k,v in bundle['arrays'].items() if not np.array_equal(v,bundle['inputs'][k])}\n fw=generate_framework(framework)\n label=os.environ.get('DIAG_LABEL','default')\n region=Region(bench,fw,label,writebacks,restore=framework.startswith('dace') and process!='bootstrap',scalar_returns=tuple(np.asarray(x).shape==() for x in bundle['returns']))\n records=[]\n for call in range(int(os.environ.get('DIAG_CALLS',1 if process=='bootstrap' else 3))):\n  cache_before=fw.cache_snapshot(region.impl) if framework=='numba' else None\n  data=golden.clone_data(bundle['inputs']);cpu_start=time.process_time();start=time.perf_counter();ret=region(data);elapsed=time.perf_counter()-start;cpu_seconds=time.process_time()-cpu_start\n  actual={'returns':[np.asarray(x).copy() for x in ret],'arrays':{k:v.copy() for k,v in region.observe_arrays(data).items()}};region.release()\n  diagnostic=[];valid=golden.validate(bench,bundle,actual,diagnostics=diagnostic)\n  records.append({'call':call,'seconds':elapsed,'process_cpu_seconds':cpu_seconds,'valid':valid,'diagnostics':diagnostic,'numba_cache_after':fw.cache_snapshot(region.impl) if framework=='numba' else None})\n  result={'semantics_version':compilation_contract.FRONTEND_SEMANTICS_VERSION,'candidate_source_id':os.environ.get('STAGE2_SOURCE_ID'),'baseline_artifact':os.environ.get('STAGE2_BASELINE')=='1','benchmark':name,'implementation':label,'affinity':sorted(os.sched_getaffinity(0)),'framework':framework,'preset':preset,'process':process,'golden':event,'input_hashes':{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in bundle['inputs'].items() if isinstance(v,np.ndarray)},'records':records,'compile_forbidden':process!='bootstrap' and framework.startswith('cova'),'artifact':str(artifact),'binary_before':before,'binary_after':identities(),'input_shapes':{k:[list(v.shape),str(v.dtype)] for k,v in bundle['inputs'].items() if isinstance(v,np.ndarray)},'scalar_inputs':{k:v for k,v in bundle['inputs'].items() if isinstance(v,(int,float,str))}}\n  (base/f'{name}-{framework}-{preset}-{process}.json').write_text(json.dumps(result,indent=2)+'\\n');print(records[-1],flush=True)\n  if not valid:raise SystemExit(2)\n", 'transform.py': '"""Diagnostic source ablation only. Never installed in CoVA."""\nimport re\ndef prepare(text,mode=\'prepare\'):\n start=text.index(\'      for (size_t v61 = v10; v61 < v16; v61 += v8) {\')\n end=text.index(\'      double v87 = v59;\',start)\n old=text[start:end]\n producer=old[old.index(\'        size_t v64\'):old.index(\'        double v79\')]\n producer=producer.replace(\'v61 % v16\',\'(cova_base + cova_j) % v16\')\n consumer=old[old.index(\'        double v79\'):old.index(\'      };\')]\n # Both immutable read-only arrays, no stores in preparation; every consume remains ordered.\n prep=producer+\'\'\'\n        cova_values[cova_j] = v78;\n        cova_masks[cova_j] = v76;\n\'\'\'\n cons=\'\'\'        double v62 = v59;\n        int64_t v63 = v60;\n        double v78 = cova_values[cova_j];\n        bool v76 = cova_masks[cova_j];\n\'\'\'+consumer\n new=\'\'\'      double cova_values[16];\n      bool cova_masks[16];\n      for (size_t cova_base = v10; cova_base < v16; cova_base += 16) {\n        #pragma unroll\n        for (size_t cova_j = 0; cova_j < 16; ++cova_j) {\n          if (cova_base + cova_j < v16) {\n\'\'\'+prep+\'\'\'          }\n        }\n        #pragma unroll\n        for (size_t cova_j = 0; cova_j < 16; ++cova_j) {\n          if (cova_base + cova_j < v16) {\n\'\'\'+cons+\'\'\'          }\n        }\n      }\n\'\'\'\n if mode==\'pipeline\':\n  nextprep=prep.replace(\'(cova_base + cova_j)\', \'(cova_base + 16 + cova_j)\').replace(\'cova_values[\',\'cova_next_values[\').replace(\'cova_masks[\',\'cova_next_masks[\')\n  initial=prep.replace(\'(cova_base + cova_j)\', \'(cova_j)\')\n  new = "      double cova_values[16], cova_next_values[16];\\n      bool cova_masks[16], cova_next_masks[16];\\n"\n  new += "      #pragma unroll\\n      for (size_t cova_j=0; cova_j<16; ++cova_j) {\\n        if(cova_j<v16) {\\n"+initial+"        }\\n      }\\n"\n  new += "      for(size_t cova_base=v10; cova_base<v16; cova_base+=16) {\\n        #pragma unroll\\n        for(size_t cova_j=0;cova_j<16;++cova_j) {\\n          if(cova_base+16+cova_j<v16) {\\n"+nextprep+"          }\\n        }\\n"\n  new += "        #pragma unroll\\n        for(size_t cova_j=0;cova_j<16;++cova_j) {\\n          if(cova_base+cova_j<v16) {\\n"+cons+"          }\\n        }\\n"\n  new += "        #pragma unroll\\n        for(size_t cova_j=0;cova_j<16;++cova_j) {\\n          if(cova_base+16+cova_j<v16) {\\n            cova_values[cova_j]=cova_next_values[cova_j];\\n            cova_masks[cova_j]=cova_next_masks[cova_j];\\n          }\\n        }\\n      }\\n"\n else:\n  assert mode==\'prepare\' \n return text[:start]+new+text[end:]\n', 'run.py': "import subprocess,os,time,json,signal,fcntl\nfrom pathlib import Path\nr=Path('/tmp/cova-stage2-20260923');e=Path(os.environ.get('DIAG_OUTPUT_DIR',str(r/'evidence')))\ndef run(fw,preset,process,artifact=None,label='default',limit=300,calls=None,name='azimint_naive'):\n key=f'{name}-{fw}-{preset}-{process}';env=os.environ.copy();env['DIAG_LABEL']=label\n if artifact:env['DIAG_ARTIFACT']=artifact\n else:env.pop('DIAG_ARTIFACT',None)\n if calls:env['DIAG_CALLS']=str(calls)\n else:env.pop('DIAG_CALLS',None)\n cmd=['taskset','-c','16-23','python',env.get('STAGE2_DRIVER_PATH',str(r/'driver.py')),name,fw,preset,process]\n start=time.time();reason=None;peak=0\n with (e/(key+'.log')).open('w') as log:\n  p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\n  while p.poll() is None:\n   try:\n    status=Path(f'/proc/{p.pid}/status').read_text();rss=int(next(x.split()[1] for x in status.splitlines() if x.startswith('VmRSS:')))*1024;peak=max(peak,rss)\n   except (OSError,StopIteration):pass\n   if time.time()-start>limit:reason='worker_timeout'\n   if peak>16*1024**3:reason='rss_limit'\n   if reason:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=10);break\n   time.sleep(1)\n result={'key':key,'command':cmd,'start_epoch':start,'elapsed_s':time.time()-start,'returncode':p.returncode,'reason':reason,'peak_rss_bytes':peak,'label':label}\n with (e/'execution.jsonl').open('a') as f:f.write(json.dumps(result)+'\\n')\n print(result,flush=True);return p.returncode==0 and not reason\nif __name__=='__main__':\n lock=open('/HSC/users/tianhong/myHome/developing/npbench/.cache/large-run.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)\n for p in ['p1','p2','p3']:\n  for fw,label,artifact in [('cova_llvm_gpu','default','/tmp/azim-stage45/final/azimint_naive-cova_llvm_gpu-L'),('cupy','default',None),('cova_llvm_cpu','default','/tmp/azim-stage45/final/azimint_naive-cova_llvm_cpu-L'),('numba','nopython-mode-parallel-range',None)]:\n   if not run(fw,'L',p,artifact,label):raise SystemExit(2)\n", 'probe.py': "import subprocess,runpy,os\nfrom pathlib import Path\nfrom transform import prepare\nreal=subprocess.run\ndef run(cmd,*a,**kw):\n result=real(cmd,*a,**kw)\n if isinstance(cmd,list) and '-emit-omp-gpu-cpp' in cmd:\n  p=Path(cmd[cmd.index('-o')+1]);p.write_text(prepare(p.read_text(),os.environ.get('PREP_MODE','prepare')))\n return result\nsubprocess.run=run\nrunpy.run_path(str(Path(__file__).with_name('driver.py')),run_name='__main__')\n"}
import fcntl, hashlib, json, os, shutil, statistics, subprocess, sys, time
from pathlib import Path
ROOT=Path.cwd()
COVA=Path('/HSC/users/tianhong/myHome/developing/CoVA')
def git(repo,*args):
    return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()
assert git(COVA,'rev-parse','HEAD')=='afbe2192ad070081fd5b47dfdf1f5406b6b3477b', 'CoVA HEAD changed; review before running'
assert git(ROOT,'rev-parse','HEAD')=='038fccb5acc000d97448a27d76fc54843bd9ca08', 'NPBench HEAD changed; review before running'
assert not git(COVA,'status','--porcelain'), 'CoVA worktree is not clean'
assert not git(ROOT,'diff','HEAD','--'), 'NPBench tracked changes need review'
assert set(range(16,24)) <= os.sched_getaffinity(0), 'CPU 16-23 unavailable'
cache=ROOT/'.cache';cache.mkdir(exist_ok=True)
lock=(cache/'large-run.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
runroot=cache/'ordered-prepare-L'/time.strftime('%Y%m%d-%H%M%S')
runroot.mkdir(parents=True,exist_ok=False)
e=runroot/'evidence';e.mkdir()
scripts=runroot/'scripts';scripts.mkdir()
for name,source in ASSETS.items(): (scripts/name).write_text(source)
shutil.copy2(os.environ['L_TEST_SCRIPT'],e/'run_ordered_prepare_L.sh')
os.environ['DIAG_OUTPUT_DIR']=str(e)
os.environ['PREP_MODE']='prepare'
os.environ.pop('STAGE2_BASELINE',None)
os.environ['OMP_TARGET_OFFLOAD']='MANDATORY'
os.environ['STAGE2_SOURCE_ID']='afbe2192:ordered-prepare16-L-diagnostic'
os.environ['NUMBA_CACHE_DIR']=str(runroot/'numba-cache')
os.environ['CUPY_CACHE_DIR']=str(runroot/'cupy-cache')
sys.path.insert(0,str(scripts))
from run import run

def save(name,value): (e/name).write_text(json.dumps(value,indent=2)+'\n')
identity={'cova_head':git(COVA,'rev-parse','HEAD'),'npbench_head':git(ROOT,'rev-parse','HEAD'),
 'npbench_status':git(ROOT,'status','--porcelain'),'python':sys.version,
 'resources':{k:v for k,v in os.environ.items() if k.startswith(('OMP_','COVA_','NPBENCH_','CUDA_VISIBLE'))},
 'script_sha256':hashlib.sha256(Path(os.environ['L_TEST_SCRIPT']).read_bytes()).hexdigest(),
 'scope':'L only; baseline vs prepare16; no profiler/competitors; bootstrap excluded; AB/BA/AB; fresh + 3 hot calls'}
for tool in [COVA/'build/bin/cova-opt',COVA/'build/bin/cova-translate']:
 if tool.exists(): identity[str(tool)]=hashlib.sha256(tool.read_bytes()).hexdigest()
identity['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True)
save('identity.json',identity)
print('Results:',e,flush=True)
print('Two isolated bootstraps, then 3 paired processes per variant, 4 calls/process.',flush=True)
artifacts={v:runroot/'artifacts'/v for v in ['baseline','prepare']}
reference=None
results={}
def worker(variant,process,calls):
 global reference
 os.environ['STAGE2_DRIVER_PATH']=str(scripts/('probe.py' if variant=='prepare' else 'driver.py'))
 if not run('cova_openmp_gpu','L',process,str(artifacts[variant]),calls=calls,limit=600):
  raise RuntimeError('Worker failed; stopping remaining tests. Inspect execution.jsonl and worker log.')
 key=f'azimint_naive-cova_openmp_gpu-L-{process}'
 result=json.loads((e/(key+'.json')).read_text())
 assert len(result['records'])==calls and all(x['valid'] for x in result['records'])
 assert result['semantics_version']==17
 current={k:result[k] for k in ['input_hashes','input_shapes','scalar_inputs']}
 if reference is None: reference=current
 assert current==reference, 'Input identity changed'
 if process!='bootstrap':
  assert result['compile_forbidden'] and result['binary_before'] and result['binary_before']==result['binary_after'], 'Restore identity mismatch'
 metas=list(artifacts[variant].rglob('*.backend.json'))
 assert metas, 'Missing GPU backend identity'
 assert any(json.loads(p.read_text()).get('actual_device')=='gpu' and not json.loads(p.read_text()).get('fallback_reason') for p in metas), 'No GPU backend evidence'
 results[variant+'-'+process]=result
 return result
try:
 for variant in ['baseline','prepare']:
  worker(variant,'bootstrap',1)
  # Separate bootstrap evidence before the next variant uses the bootstrap label.
  for suffix in ['json','log']:
   p=e/f'azimint_naive-cova_openmp_gpu-L-bootstrap.{suffix}'
   p.rename(e/f'azimint_naive-cova_openmp_gpu-L-{variant}-bootstrap.{suffix}')
  ir=e/'ir'/variant;ir.mkdir(parents=True)
  sources=list(artifacts[variant].rglob('*omp_gpu.cpp'))
  assert len(sources)==1, 'Unexpected generated scan source count'
  source=sources[0].read_text()
  assert ('cova_values[16]' in source)==(variant=='prepare'), 'Candidate transformation not confirmed'
  for p in sources+list(artifacts[variant].rglob('*.backend.json')): shutil.copy2(p,ir/p.name)
 for pair in range(1,4):
  for variant in (['baseline','prepare'] if pair%2 else ['prepare','baseline']):
   worker(variant,f'{variant}-p{pair}',4)
 med={v:[statistics.median(x['seconds'] for x in results[f'{v}-{v}-p{i}']['records'][1:]) for i in range(1,4)] for v in artifacts}
 b=statistics.median(med['baseline']);c=statistics.median(med['prepare'])
 summary={'status':'complete','hot_process_medians_s':med,'baseline_s':b,'prepare_s':c,'candidate_over_baseline':c/b,
 'improvement_percent':100*(1-c/b),'pair_wins':sum(c<b for b,c in zip(med['baseline'],med['prepare'])),
 'fresh_process_s':{v:[results[f'{v}-{v}-p{i}']['records'][0]['seconds'] for i in range(1,4)] for v in artifacts},
 'note':'26 strict calls incl 2 bootstrap; hot metric excludes bootstrap/fresh; no competitor ranking or formal five-pair acceptance'}
 save('summary.json',summary)
 print(json.dumps(summary,indent=2),flush=True)
except BaseException as exc:
 save('failure.json',{'type':type(exc).__name__,'message':str(exc)})
 raise
finally:
 # Evidence is text/scalars/source only. Artifacts remain separately for reuse.
 save('manifest.json',{str(p.relative_to(e)):hashlib.sha256(p.read_bytes()).hexdigest() for p in e.rglob('*') if p.is_file() and p.name!='manifest.json'})
 print('Share this evidence directory only:',e,flush=True)

COVA_L_DIAGNOSTIC_PY
