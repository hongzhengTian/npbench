# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
import importlib
import pathlib

from npbench.infrastructure import Benchmark, Framework
from typing import Callable, Sequence, Tuple

_impl = {
    'object-mode': 'o',
    'object-mode-parallel': 'op',
    'object-mode-parallel-range': 'opr',
    'nopython-mode': 'n',
    'nopython-mode-parallel': 'np',
    'nopython-mode-parallel-range': 'npr'
}


class NumbaFramework(Framework):
    """ A class for reading and processing framework information. """

    def __init__(self, fname: str):
        """ Reads framework information.
        :param fname: The framework name.
        """

        super().__init__(fname)

    def impl_files(self, bench: Benchmark) -> Sequence[Tuple[str, str]]:
        """ Returns the framework's implementation files for a particular
        benchmark.
        :param bench: A benchmark.
        :returns: A list of the benchmark implementation files.
        """

        parent_folder = pathlib.Path(__file__).parent.absolute()
        implementations = []
        for impl_name, impl_postfix in _impl.items():
            pymod_path = parent_folder.joinpath(
                "..", "..", "npbench", "benchmarks", bench.info["relative_path"],
                bench.info["module_name"] + "_" + self.info["postfix"] + "_" + impl_postfix + ".py")
            implementations.append((pymod_path, impl_name))
        return implementations

    def implementation_names(self, bench):
        return [label for source, label in self.impl_files(bench) if source.is_file()]

    def artifact_policy(self, implementation):
        if not hasattr(implementation, 'enable_caching'):
            return 'python'
        # Enabling the cache is a request, not proof that this specialization
        # can be serialized. Numba explicitly excludes lifted code and dynamic
        # globals in CompileResultCacheImpl.check_cachable.
        for result in implementation.overloads.values():
            if any(not loop.can_cache for loop in result.lifted) or result.library.has_dynamic_globals:
                return 'numba_memory_only'
        return 'numba_disk_cache'

    def load_implementation(self, bench, label, *, restore=False):
        module = 'npbench.benchmarks.' + bench.info['relative_path'].replace('/', '.')
        module += '.' + bench.info['module_name'] + '_' + self.info['postfix'] + '_' + _impl[label]
        implementation = getattr(importlib.import_module(module), bench.info['func_name'])
        # The optional lifecycle mode enables the dispatcher's supported disk
        # cache without changing the upstream algorithm or JIT options.
        if hasattr(implementation, 'enable_caching'):
            implementation.enable_caching()
        return implementation

    def cache_snapshot(self, implementation):
        """Copy dispatcher counters outside the measured region."""
        stats = getattr(implementation, 'stats', None)
        if stats is None or not all(hasattr(stats, name) for name in ('cache_hits', 'cache_misses')):
            return None
        return {'hits': {str(k): int(v) for k, v in stats.cache_hits.items()},
                'misses': {str(k): int(v) for k, v in stats.cache_misses.items()},
                'signatures': sorted(str(s) for s in implementation.signatures)}

    def cache_observation(self, implementation, before, phase):
        """Distinguish entry-dispatcher disk hits, compilation and memory reuse.

        Artifact integrity is checked separately by the lifecycle worker.
        These counters do not certify arbitrary nested runtime activity.
        """
        after = self.cache_snapshot(implementation)
        observation = {'schema': 1, 'scope': 'entry_dispatcher',
                       'before': before, 'after': after, 'state': 'unverified'}
        if before is None or after is None:
            return observation
        hits = sum(after['hits'].values()) - sum(before['hits'].values())
        misses = sum(after['misses'].values()) - sum(before['misses'].values())
        observation.update(cache_hits=hits, cache_misses=misses)
        if hits < 0 or misses < 0:
            return observation
        if misses:
            observation['state'] = 'cache_miss'
        elif hits:
            observation['state'] = 'disk_hit'
        elif before['signatures'] and before['signatures'] == after['signatures']:
            observation['state'] = 'memory_reuse'
        observation['reuse_verified'] = ((phase == 'fresh_process' and observation['state'] == 'disk_hit')
                                         or (phase == 'same_process' and observation['state'] == 'memory_reuse'))
        return observation

    def implementations(self, bench: Benchmark) -> Sequence[Tuple[Callable, str]]:
        """ Returns the framework's implementations for a particular benchmark.
        :param bench: A benchmark.
        :returns: A list of the benchmark implementations.
        """

        module_pypath = "npbench.benchmarks.{r}.{m}".format(r=bench.info["relative_path"].replace('/', '.'),
                                                            m=bench.info["module_name"])
        if "postfix" in self.info.keys():
            postfix = self.info["postfix"]
        else:
            postfix = self.fname
        module_str = "{m}_{p}".format(m=module_pypath, p=postfix)
        func_str = bench.info["func_name"]

        implementations = []
        for impl_name, impl_postfix in _impl.items():
            ldict = dict()
            try:
                module = importlib.import_module("{m}_{p}".format(m=module_str, p=impl_postfix))
                ldict['impl'] = getattr(module, func_str)
                implementations.append((ldict['impl'], impl_name))
            except ImportError:
                continue
            except Exception:
                print("Failed to load the {r} {f} implementation.".format(r=self.info["full_name"], f=impl_name))
                continue

        return implementations
