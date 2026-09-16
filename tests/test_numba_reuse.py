"""Dispatcher counter semantics; no benchmark execution in these unit tests."""
from collections import Counter
from types import SimpleNamespace
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from npbench.infrastructure.numba_framework import NumbaFramework


class NumbaReuseTest(unittest.TestCase):
    def setUp(self):
        self.framework = NumbaFramework('numba')
        self.stats = SimpleNamespace(cache_hits=Counter(), cache_misses=Counter())
        self.impl = SimpleNamespace(stats=self.stats, signatures=[])

    def test_fresh_disk_hit_requires_no_compile_miss(self):
        before = self.framework.cache_snapshot(self.impl)
        self.stats.cache_hits['array'] += 1
        self.impl.signatures.append('array')
        result = self.framework.cache_observation(self.impl, before, 'fresh_process')
        self.assertEqual(result['state'], 'disk_hit')
        self.assertTrue(result['reuse_verified'])
        self.stats.cache_misses['scalar'] += 1
        self.assertFalse(self.framework.cache_observation(self.impl, before, 'fresh_process')['reuse_verified'])
        self.assertEqual(before['hits'], {})  # snapshot must not share live counters

    def test_unchanged_counters_without_signature_are_not_memory_reuse(self):
        before = self.framework.cache_snapshot(self.impl)
        self.assertFalse(self.framework.cache_observation(self.impl, before, 'same_process')['reuse_verified'])
        self.impl.signatures.append('array')
        before = self.framework.cache_snapshot(self.impl)
        self.assertTrue(self.framework.cache_observation(self.impl, before, 'same_process')['reuse_verified'])
        self.assertFalse(self.framework.cache_observation(self.impl, before, 'fresh_process')['reuse_verified'])

    def test_missing_or_reset_counters_never_certify_reuse(self):
        self.assertEqual(self.framework.cache_observation(object(), None, 'fresh_process')['state'], 'unverified')
        self.stats.cache_hits['array'] = 2
        before = self.framework.cache_snapshot(self.impl)
        self.stats.cache_hits.clear()
        self.assertEqual(self.framework.cache_observation(self.impl, before, 'fresh_process')['state'], 'unverified')

    def test_worker_rejects_miss_even_when_artifact_files_are_unchanged(self):
        from npbench.infrastructure.lifecycle import worker
        stats, impl = self.stats, self.impl
        class TestRegion:
            stage = 'execute'
            def __init__(self, *args, **kwargs): self.impl = impl
            def __call__(self, data):
                stats.cache_misses['array'] += 1
                impl.signatures.append('array')
                return []
            def observe_arrays(self, data): return {}
            def release(self): pass
        request = {'benchmark':'gemm', 'framework':'numba', 'implementation':'nopython-mode',
                   'preset':'S', 'process_index':1, 'repeat':0, 'validate':True,
                   'golden_cache':'unused', 'version':'unit', 'implementation_sources':{},
                   'golden_sha256':'payload', 'samples_file':'samples.jsonl'}
        bundle = {'inputs':{}, 'arrays':{}, 'returns':[]}
        event = {'key':'key', 'sha256':'payload', 'producer_executions':0}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root/'request.json').write_text(json.dumps(request))
            with patch('npbench.infrastructure.lifecycle.Path.cwd', return_value=root), \
                 patch('npbench.infrastructure.lifecycle.generate_framework', return_value=self.framework), \
                 patch.object(self.framework, 'version', return_value='unit'), \
                 patch.object(self.framework, 'artifact_policy', return_value='numba_disk_cache'), \
                 patch('npbench.infrastructure.lifecycle.golden.source_hashes', return_value={}), \
                 patch('npbench.infrastructure.lifecycle.golden.load_or_create', return_value=(bundle,event)), \
                 patch('npbench.infrastructure.lifecycle.artifact_state', return_value={'cached.nbc':{'sha256':'same'}}), \
                 patch('npbench.infrastructure.lifecycle.Region', TestRegion), \
                 patch('npbench.infrastructure.resources.observe_resources', return_value={}), \
                 patch.dict(os.environ, {'NPBENCH_RESOURCE_PROBE':'0'}):
                self.assertEqual(worker(root/'request.json'), 1)
            row = json.loads((root/'samples.jsonl').read_text())
            self.assertTrue(row['validated'])
            self.assertEqual(row['artifacts']['state'], 'unchanged')
            self.assertEqual(row['status'], 'artifact_reuse_failed')
            self.assertEqual(row['failure_stage'], 'restore')


if __name__ == '__main__':
    unittest.main()
