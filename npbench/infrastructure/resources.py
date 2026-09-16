"""Untimed process/resource observations; metadata is not utilization proof."""
import os
import time
from pathlib import Path


def observe_resources(framework, region):
    result = dict(process_resources(), schema=2,
                  configured_pools_are_not_active_thread_measurements=True)
    try:
        from threadpoolctl import threadpool_info
        result['threadpools'] = threadpool_info()
    except ImportError:
        result['threadpools_unavailable'] = 'threadpoolctl is not installed'
    if framework.fname == 'numba':
        import numba
        result['numba_configured_max_threads'] = numba.config.NUMBA_NUM_THREADS
        try:
            result['numba_threading_layer'] = numba.threading_layer()
            result['numba_threads'] = numba.get_num_threads()
        except ValueError:
            result['numba_threading_layer'] = 'not_initialized_by_workload'
    if framework.uses_device_arrays() or framework.fname in ('cova_llvm_gpu', 'cova_openmp_gpu'):
        import cupy
        result['gpu'] = gpu_identity()
        result['gpu']['input_array_devices'] = {key: int(value.device.id) for key, value in region.context.items()
                                               if isinstance(value, cupy.ndarray)}
    return result


def thread_cpu_snapshot():
    """Linux CPU accounting around a call, outside its performance timer."""
    result = {'wall': time.perf_counter(), 'ticks_per_second': os.sysconf('SC_CLK_TCK'), 'threads': {}}
    try:
        paths = list(Path('/proc/self/task').glob('*/stat'))
        for path in paths:
            try:
                # comm may contain spaces and parentheses; fields after its
                # final parenthesis start at state (field 3).
                fields = path.read_text().rsplit(')', 1)[1].split()
                result['threads'][path.parent.name] = {
                    'start': int(fields[19]), 'ticks': int(fields[11]) + int(fields[12])}
            except FileNotFoundError:
                continue  # A thread may exit while enumerating /proc.
    except (OSError, ValueError, IndexError) as error:
        result['probe_error'] = type(error).__name__ + ': ' + str(error)
    return result


def thread_cpu_activity(before, after):
    """Observed thread work, with tick resolution and missing exits explicit."""
    if before.get('probe_error') or after.get('probe_error'):
        return {'probe_error': before.get('probe_error') or after['probe_error']}
    deltas = {}
    for tid, value in after['threads'].items():
        old = before['threads'].get(tid)
        ticks = old['ticks'] if old and old['start'] == value['start'] else 0
        deltas[tid] = max(0, value['ticks'] - ticks) / after['ticks_per_second']
    return {'cpu_seconds_by_thread': deltas,
            'threads_with_observed_cpu_work': sum(value > 0 for value in deltas.values()),
            'cpu_tick_seconds': 1 / after['ticks_per_second'],
            'bracketing_wall_seconds': after['wall'] - before['wall'],
            'exited_thread_ids': sorted(set(before['threads']) - set(after['threads'])),
            'exited_threads_and_child_processes_not_counted': True,
            'includes_snapshot_overhead_outside_call_timer': True}


def gpu_identity():
    """CUDA identity remains available when NVML/system nvidia-smi fails."""
    import cupy
    runtime = cupy.cuda.runtime
    device = runtime.getDevice()
    props = runtime.getDeviceProperties(device)
    identity = {'device_index': device, 'driver_version': runtime.driverGetVersion(),
                     'runtime_version': runtime.runtimeGetVersion()}
    for key in ('name', 'uuid', 'pciDomainID', 'pciBusID', 'pciDeviceID', 'totalGlobalMem', 'major', 'minor'):
        value = props.get(key)
        if isinstance(value, bytes):
            value = value.hex() if key == 'uuid' else value.decode(errors='replace')
        identity[key] = value
    if not identity.get('uuid'):
        # Query the current CUDA context through the driver when this
        # CuPy build does not expose UUID in getDeviceProperties.
        import ctypes
        try:
            driver = ctypes.CDLL('libcuda.so.1')
            current = ctypes.c_int()
            # Metadata-only preflight may not yet own a CUDA context.
            runtime.free(0)
            code = driver.cuCtxGetDevice(ctypes.byref(current))
            uuid = (ctypes.c_ubyte * 16)()
            get_uuid = getattr(driver, 'cuDeviceGetUuid_v2', None) or driver.cuDeviceGetUuid
            if code == 0:
                code = get_uuid(ctypes.byref(uuid), current)
            if code == 0:
                identity['uuid'] = bytes(uuid).hex()
                identity['uuid_source'] = 'CUDA driver current context'
            else:
                identity['uuid_error'] = 'CUDA driver error ' + str(code)
        except (OSError, AttributeError) as error:
            identity['uuid_error'] = str(error)
    return identity


def process_resources():
    """Observe each Linux thread separately: affinity(0) is not process-wide."""
    result = {'affinity': sorted(os.sched_getaffinity(0)), 'affinity_scope': 'calling_thread',
              'thread_affinities': {}, 'thread_environment': {k: os.environ.get(k) for k in
                  ('OMP_NUM_THREADS', 'OMP_PROC_BIND', 'OMP_PLACES', 'OPENBLAS_NUM_THREADS',
                   'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS', 'NUMBA_THREADING_LAYER')}}
    for thread in Path('/proc/self/task').iterdir():
        try:
            result['thread_affinities'][thread.name] = sorted(os.sched_getaffinity(int(thread.name)))
        except ProcessLookupError:
            continue
    result['live_os_threads'] = len(result['thread_affinities'])
    result['memory_nodes'] = next((s.partition(':')[2].strip() for s in Path('/proc/self/status').read_text().splitlines()
                                   if s.startswith('Mems_allowed_list:')), None)
    return result


def allocation_snapshot():
    """Stable allocation identity, separate from load and runtime thread pins."""
    import subprocess
    online = Path('/sys/devices/system/cpu/online').read_text().strip()
    groups = Path('/proc/self/cgroup').read_text()
    limits = {}
    roots = [Path('/sys/fs/cgroup')]
    for line in groups.splitlines():
        if line.startswith('0::'):
            folder = Path('/sys/fs/cgroup') / line[3:].lstrip('/')
            if folder.is_relative_to('/sys/fs/cgroup'):
                roots += [folder, *[parent for parent in folder.parents if parent.is_relative_to('/sys/fs/cgroup')]]
    for root in set(roots):
        for name in ('cpu.max', 'cpuset.cpus.effective', 'cpuset.mems.effective', 'memory.max'):
            path = root / name
            if path.is_file():
                limits[str(path)] = path.read_text().strip()
    try:
        numa = subprocess.check_output(['numactl', '--show'], text=True, stderr=subprocess.STDOUT, timeout=10)
    except (OSError, subprocess.SubprocessError) as error:
        numa = 'unavailable: ' + str(error)
    return {'online_cpus': online, 'affinity': sorted(os.sched_getaffinity(0)), 'cgroup': groups,
            'cgroup_limits': limits, 'numa_policy': numa,
            'kernel': os.uname().release,
            'nvidia_kernel': Path('/proc/driver/nvidia/version').read_text() if Path('/proc/driver/nvidia/version').exists() else None}


def idle_observation(affinity, *, gpu=False):
    """Brief host-load sample; unavailable GPU occupancy remains explicit."""
    import subprocess
    def ticks():
        data = {}
        for line in Path('/proc/stat').read_text().splitlines():
            key, *values = line.split()
            if key.startswith('cpu') and key[3:].isdigit() and int(key[3:]) in affinity:
                # Exclude guest fields: already counted in user/nice.
                nums = list(map(int, values[:8])); data[key] = (sum(nums), nums[3]+nums[4])
        return data
    before = ticks(); time.sleep(.3); after = ticks()
    total = idle = 0
    for key in before.keys() & after.keys():
        total += after[key][0]-before[key][0]; idle += after[key][1]-before[key][1]
    result = {'host_busy_fraction': 1-idle/total if total else None,
              'sample_seconds': .3, 'gpu_processes': None, 'gpu_occupancy_verified': not gpu}
    if gpu:
        try:
            output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory',
                                              '--format=csv,noheader,nounits'], stderr=subprocess.STDOUT, text=True, timeout=10, env=monitor_environment())
            rows = [line.strip() for line in output.splitlines() if line.strip()]
            result['gpu_processes'] = rows
            result['gpu_occupancy_verified'] = True
        except (OSError, subprocess.SubprocessError) as error:
            result['gpu_probe_error'] = str(error)
            if hasattr(error, 'output'): result['gpu_probe_output'] = error.output
    frequencies = []
    for cpu in affinity:
        path = Path(f'/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq')
        if path.exists():
            frequencies.append(int(path.read_text()))
    result['cpu_frequency_khz_range'] = [min(frequencies), max(frequencies)] if frequencies else None
    if gpu:
        fields = 'uuid,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,power.limit,utilization.gpu'
        try:
            result['gpu_telemetry_csv'] = subprocess.check_output(
                ['nvidia-smi', '--query-gpu='+fields, '--format=csv,noheader,nounits'],
                stderr=subprocess.STDOUT, text=True, timeout=10, env=monitor_environment()).strip()
            result['gpu_telemetry_fields'] = fields
        except (OSError, subprocess.SubprocessError) as error:
            result['gpu_telemetry_error'] = str(error)
    result['observed_competition'] = bool(result['gpu_processes']) or (result['host_busy_fraction'] or 0) > .1
    return result


def monitor_environment():
    """An optional matching NVML library is scoped to the monitor subprocess."""
    env = dict(os.environ)
    directory = env.get('NPBENCH_NVML_LIBRARY_DIR')
    if directory:
        if not (Path(directory) / 'libnvidia-ml.so.1').is_file():
            raise FileNotFoundError('Configured NVML monitoring library is missing')
        env['LD_LIBRARY_PATH'] = directory + os.pathsep + env.get('LD_LIBRARY_PATH', '')
    return env
