import json
import os
from pathlib import Path
import subprocess
import unittest

ROOT=Path(__file__).resolve().parents[1]
class ResourcePolicyTest(unittest.TestCase):
    def test_system_clears_inherited_limits_without_widening_scheduler_affinity(self):
        env=dict(os.environ,NPBENCH_RESOURCE_POLICY='system',NPBENCH_THREADS='2',NPBENCH_CPUSET='0,1',OMP_NUM_THREADS='2',OMP_PROC_BIND='true',NUMBA_THREADING_LAYER='omp',OPENBLAS_NUM_THREADS='2')
        command=['bash','-c', 'source "$1"; python -c \'import os,json; print(json.dumps({"env":dict(os.environ),"affinity":sorted(os.sched_getaffinity(0))}))\'', '--',str(ROOT/'scripts/native-env.sh')]
        result=subprocess.run(command,env=env,text=True,capture_output=True,check=True)
        data=json.loads(result.stdout)
        for key in ('NPBENCH_THREADS','NPBENCH_CPUSET','OMP_NUM_THREADS','OMP_PROC_BIND','NUMBA_THREADING_LAYER','OPENBLAS_NUM_THREADS'):
            self.assertNotIn(key,data['env'])
        self.assertEqual(data['affinity'],sorted(os.sched_getaffinity(0)))
        result=subprocess.run(command,env=dict(env,NPBENCH_RESOURCE_POLICY='fixed',NPBENCH_THREADS='3'),text=True,capture_output=True,check=True)
        data=json.loads(result.stdout);self.assertEqual(data['env']['OMP_NUM_THREADS'],'3')

if __name__=='__main__':unittest.main()
