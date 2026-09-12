"""A host-to-host call boundary using NPBench's existing framework adapters."""
import numpy as np

from .golden import outputs


def to_host(value):
    if type(value).__module__.split('.')[0] == 'cupy':
        return value.get()
    return value


class Region:
    """Lazy preparation, execution, synchronization, and required copyback.

    Fresh mutable host inputs are prepared by the caller outside the timer.
    Source imports and decorator construction happen before measurement.
    Compilation/restoration belongs to the first call, including frameworks
    that normally compile eagerly in implementations().
    """
    def __init__(self, bench, framework, label, writebacks, restore=False, artifacts_available=True):
        self.bench, self.framework, self.label = bench, framework, label
        self.writebacks = writebacks
        self.restore = restore
        self.artifacts_available = artifacts_available
        self.stage = 'prepare'
        self.impl = framework.load_implementation(bench, label, restore=restore)
        self.imports = framework.imports()
        self.statement = compile(framework.exec_str(bench, self.impl), '<npbench region>', 'exec')
        self.context = None

    def __call__(self, data):
        self.stage = 'prepare'
        fw, bench = self.framework, self.bench
        if self.restore and not self.artifacts_available and fw.artifact_policy(self.impl) in ('native', 'numba_disk_cache'):
            raise FileNotFoundError('No artifacts remain for the fresh-process reuse measurement')
        context = dict(data, **self.imports, __npb_impl=self.impl)
        # CPU/CoVA already consume host arrays. Device conversion is part of
        # the GPU region, rather than untimed timeit setup.
        copy = fw.copy_func()
        device_inputs = fw.uses_device_arrays()
        for name, argument in zip(bench.info['input_args'], fw.args(bench, self.impl)):
            context[argument] = copy(data[name]) if device_inputs and name in bench.info['array_args'] else data[name]
        self.stage = 'execute'
        exec(self.statement, context)
        self.stage = 'materialize'
        result = [to_host(value) for value in outputs(context['__npb_result'])]
        for name, argument in zip(bench.info['input_args'], fw.args(bench, self.impl)):
            if name in self.writebacks and device_inputs:
                np.copyto(data[name], to_host(context[argument]), casting='no')
        fw.synchronize()
        self.context = context
        return result

    def observe_arrays(self, data):
        # Audit even undeclared mutations. Read-only device arrays are checked
        # outside the timer; they need no host copyback for the region result.
        return {name: to_host(self.context[argument])
                for name, argument in zip(self.bench.info['input_args'],
                                          self.framework.args(self.bench, self.impl))
                if name in self.bench.info['array_args']}
