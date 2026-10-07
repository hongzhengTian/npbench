import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from npbench.infrastructure import resources as r


class GPUNumaTest(unittest.TestCase):
    def test_option_and_cpu_lists(self):
        self.assertEqual(r.gpu_numa_option({}), 'cpu-memory')
        self.assertEqual(r.gpu_numa_option({'NPBENCH_GPU_NUMA_BINDING': 'off'}), 'off')
        for value in ('', 'on', 'auto', 'AUTO'):
            with self.assertRaises(ValueError): r.gpu_numa_option({'NPBENCH_GPU_NUMA_BINDING': value})
        self.assertEqual(r.parse_cpu_list('24-26,72-74,25'), [24,25,26,72,73,74])
        for value in ('', '3-1', '-1', '1,a', '1-2-3'):
            with self.assertRaises(ValueError): r.parse_cpu_list(value)

    def test_cuda_logical_zero_resolves_mask_and_pci_domain(self):
        for mask in ('1,0', 'GPU-test-uuid', '0'):
            with patch.dict(os.environ, CUDA_VISIBLE_DEVICES=mask):
                runtime = SimpleNamespace(deviceGetPCIBusId=lambda index: self.assertEqual(index, 0) or '00000000:88:00.0')
                with patch.dict(sys.modules, {'cupy': SimpleNamespace(cuda=SimpleNamespace(runtime=runtime))}):
                    self.assertEqual(r.selected_gpu_pci_address(), '0000:88:00.0')

    def test_topology_and_allocation_intersection(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(r, 'selected_gpu_pci_address', return_value='0000:88:00.0'):
            sysfs = Path(directory); device=sysfs/'bus/pci/devices/0000:88:00.0';device.mkdir(parents=True)
            (sysfs/'devices/system/node').mkdir(parents=True);(sysfs/'devices/system/node/online').write_text('0-1')
            (device/'numa_node').write_text('1');(device/'local_cpulist').write_text('24-47,72-95')
            for framework in r.GPU_NUMA_FRAMEWORKS:
                plan=r.gpu_numa_plan(framework, allowed=range(96), sysfs=sysfs, environment={})
                self.assertEqual(plan['allowed_cpus'], list(range(24,48))+list(range(72,96)))
                self.assertEqual(plan['status'], 'bound')
            plan=r.gpu_numa_plan('cupy', allowed=[24,25,0], sysfs=sysfs, environment={})
            self.assertEqual(plan['allowed_cpus'], [24,25])
            self.assertEqual(r.gpu_numa_plan('cupy',allowed=[0],sysfs=sysfs,environment={})['reason'], 'no_local_cpus_in_allocation')
            cpu_plan=r.gpu_numa_plan('cupy',allowed=range(96),sysfs=sysfs,environment={'NPBENCH_GPU_NUMA_BINDING':'cpu'})
            self.assertEqual(cpu_plan['memory_policy'],'unchanged')
            self.assertEqual(plan['memory_nodes'],[1])
            (sysfs/'devices/system/node/online').write_text('0')
            self.assertEqual(r.gpu_numa_plan('cupy',allowed=range(96),sysfs=sysfs,environment={})['reason'],'single_numa_node')
            (sysfs/'devices/system/node/online').write_text('0-1')
            (device/'numa_node').write_text('-1')
            self.assertEqual(r.gpu_numa_plan('cupy',allowed=range(96),sysfs=sysfs,environment={})['reason'], 'unknown_numa_node')
            (device/'numa_node').unlink()
            self.assertEqual(r.gpu_numa_plan('cupy',allowed=range(96),sysfs=sysfs,environment={})['status'], 'unbound')

    def test_cpu_and_off_do_not_probe_cuda(self):
        with patch.object(r,'selected_gpu_pci_address',side_effect=AssertionError('must not probe')):
            for framework in ('numpy','numba','dace_cpu','cova_llvm_cpu','cova_llvm_cpu_serial','cova_openmp_cpu','cova_serial_cpu'):
                self.assertEqual(r.gpu_numa_plan(framework, allowed=[1,2], environment={})['allowed_cpus'], [1,2])
            self.assertEqual(r.gpu_numa_plan('cupy',allowed=[1,2],environment={'NPBENCH_GPU_NUMA_BINDING':'off'})['reason'], 'disabled')

    def test_runtime_mask_repair_preserves_narrower_masks_and_off(self):
        cpu=min(os.sched_getaffinity(0));other=next(x for x in os.sched_getaffinity(0) if x!=cpu)
        with patch.object(r.os,'sched_getaffinity',side_effect=[{cpu,other},{cpu}]), patch.object(r.os,'sched_setaffinity') as apply, patch.object(r.Path,'iterdir',return_value=[Path('/proc/self/task/123')]), patch.object(r.Path,'read_text',return_value='helper'):
            repairs=r.maintain_worker_affinity({'status':'bound','allowed_cpus':[cpu]})
            self.assertEqual(repairs[0]['name'],'helper');apply.assert_called_once_with(123,{cpu})
        with patch.object(r.os,'sched_setaffinity',side_effect=AssertionError('must not modify')):
            self.assertEqual(r.maintain_worker_affinity({'status':'unbound'}),[])

    def test_binding_precedes_numerical_import_and_checks_every_thread(self):
        cpu=min(os.sched_getaffinity(0));plan={'status':'bound','allowed_cpus':[cpu]}
        command=[sys.executable,'-c', 'import numpy; from npbench.infrastructure.resources import process_resources; import json; print(json.dumps(process_resources()))']
        with tempfile.TemporaryDirectory() as directory:
            request=Path(directory)/'request.json'
            request.write_text(json.dumps({'gpu_numa_binding':plan,'gpu_numa_result':str(Path(directory)/'placement.json')}))
            result=subprocess.run(r.worker_launch_command(command,plan,request),text=True,capture_output=True,check=True)
        resources=json.loads(result.stdout.splitlines()[-1]);r.check_worker_affinity(resources,plan)
        resources['thread_affinities']['bad']=[cpu,cpu+1]
        with self.assertRaises(ValueError):r.check_worker_affinity(resources,plan)
        self.assertEqual(r.worker_launch_command(command,{'status':'unbound'}), command)

class MemoryBindingTest(unittest.TestCase):
    def test_apply_and_verify(self):
        from npbench.infrastructure import gpu_numa_launcher as launcher
        plan={'status':'bound','allowed_cpus':[1], 'memory_policy':'bind','numa_node':1,'memory_nodes':[1]}
        with patch.object(launcher.os,'sched_getaffinity',side_effect=[{0,1},{1}]), patch.object(launcher.os,'sched_setaffinity') as cpu, patch.object(launcher,'memory_policy',side_effect=[{'mode':0,'nodes':[]},None,{'mode':2,'nodes':[1]}]) as memory:
            result=launcher.apply_placement(plan)
            self.assertTrue(result['launcher_verified']);cpu.assert_called_once_with(0,[1])
            self.assertEqual(memory.call_args_list[1].args,(2,[1]))

    def test_failure_restores_original_placement(self):
        from npbench.infrastructure import gpu_numa_launcher as launcher
        plan={'status':'bound','allowed_cpus':[1], 'memory_policy':'bind','numa_node':1,'memory_nodes':[1]}
        for results in ([{'mode':0,'nodes':[]},OSError('denied')], [{'mode':0,'nodes':[]},None,{'mode':2,'nodes':[0]},None,{'mode':0,'nodes':[]}]):
            with patch.object(launcher.os,'sched_getaffinity',side_effect=[{0,1},{1},{1},{0,1}]), patch.object(launcher.os,'sched_setaffinity') as cpu, patch.object(launcher,'memory_policy',side_effect=results):
                result=launcher.apply_placement(plan)
                self.assertEqual(result['status'],'unbound');self.assertEqual(result['memory_policy'],'unchanged')
                self.assertEqual(cpu.call_args.args,(0,{0,1}));self.assertIn('placement_failed',result['reason'])

    def test_cpu_only_does_not_change_memory_policy(self):
        from npbench.infrastructure import gpu_numa_launcher as launcher
        with patch.object(launcher.os,'sched_getaffinity',side_effect=[{0,1},{1}]), patch.object(launcher.os,'sched_setaffinity'), patch.object(launcher,'memory_policy',side_effect=AssertionError('CPU only')):
            self.assertTrue(launcher.apply_placement({'allowed_cpus':[1],'memory_policy':'unchanged'})['launcher_verified'])

    def test_cpu_affinity_failure_falls_back_and_rollback_failure_stops(self):
        from npbench.infrastructure import gpu_numa_launcher as launcher
        plan={'allowed_cpus':[1],'memory_policy':'unchanged'}
        with patch.object(launcher.os,'sched_getaffinity',side_effect=[{0,1},{0,1}]), patch.object(launcher.os,'sched_setaffinity',side_effect=[OSError('denied'),None]):
            self.assertEqual(launcher.apply_placement(plan)['status'],'unbound')
        with patch.object(launcher.os,'sched_getaffinity',side_effect=[{0,1},{1,2},{1,2}]), patch.object(launcher.os,'sched_setaffinity',side_effect=[None,OSError('rollback denied')]):
            with self.assertRaises(OSError):launcher.apply_placement(plan)

    def test_page_summary_and_read_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'maps';path.write_text('100 bind:1 anon=3 N1=3\n200 default file=/lib/a N0=2 N1=1\n')
            summary=r.numa_page_summary(path)
            self.assertEqual(summary['pages_by_node'],{'1':4,'0':2})
            self.assertEqual(summary['anonymous_pages_by_node'],{'1':3})
            self.assertIn('error',r.numa_page_summary(Path(directory)/'missing'))

if __name__ == '__main__': unittest.main()
