"""Set worker placement before exec; intentionally imports only the standard library."""
import ctypes
import json
import os
from pathlib import Path
import platform
import sys


def memory_policy(mode=None, nodes=()):
    """Read/set Linux task policy without a libnuma or numactl dependency."""
    numbers = {'x86_64': (238, 239), 'aarch64': (237, 236)}
    if platform.system() != 'Linux' or platform.machine() not in numbers:
        raise OSError('unsupported set_mempolicy architecture')
    set_number, get_number = numbers[platform.machine()]
    bits = ctypes.sizeof(ctypes.c_ulong) * 8
    # get_mempolicy requires room for all possible system nodes, not just the target.
    possible = Path('/sys/devices/system/node/possible').read_text().strip()
    highest = max(int(x) for part in possible.split(',') for x in part.split('-'))
    maxnode = ((highest + 1 + bits - 1) // bits) * bits
    mask = (ctypes.c_ulong * (maxnode // bits))()
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    if mode is None:
        actual = ctypes.c_int()
        status = libc.syscall(ctypes.c_long(get_number), ctypes.byref(actual),
                              ctypes.byref(mask), ctypes.c_ulong(maxnode), ctypes.c_void_p(), ctypes.c_ulong(0))
    else:
        for node in nodes:
            if not 0 <= node < maxnode:
                raise ValueError('memory node outside possible topology')
            mask[node // bits] |= 1 << (node % bits)
        status = libc.syscall(ctypes.c_long(set_number), ctypes.c_int(mode),
                              ctypes.byref(mask), ctypes.c_ulong(maxnode))
        actual = ctypes.c_int(mode)
    if status < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return {'mode': actual.value, 'nodes': [n for n in range(maxnode) if mask[n // bits] & (1 << (n % bits))]}


def apply_placement(plan):
    """Apply and verify atomically; undo CPU placement when memory setup fails."""
    result = dict(plan)
    original_cpus = os.sched_getaffinity(0)
    original_memory = None
    memory_changed = False
    try:
        if plan.get('memory_policy') == 'bind':
            original_memory = memory_policy()
        os.sched_setaffinity(0, plan['allowed_cpus'])
        if sorted(os.sched_getaffinity(0)) != plan['allowed_cpus']:
            raise OSError('CPU affinity verification failed')
        if original_memory is not None:
            memory_policy(2, [plan['numa_node']])  # MPOL_BIND
            memory_changed = True
            observed = memory_policy()
            if observed != {'mode': 2, 'nodes': [plan['numa_node']]}:
                raise OSError('memory policy verification failed')
            result['observed_memory_policy'] = observed
        result['launcher_verified'] = True
    except (OSError, ValueError) as error:
        if memory_changed:
            memory_policy(original_memory['mode'], original_memory['nodes'])
            if memory_policy() != original_memory:
                raise OSError('memory policy rollback verification failed')
        if os.sched_getaffinity(0) != original_cpus:
            os.sched_setaffinity(0, original_cpus)
            if os.sched_getaffinity(0) != original_cpus:
                raise OSError('CPU placement rollback verification failed')
        result.update(status='unbound', allowed_cpus=sorted(original_cpus), memory_policy='unchanged',
                      memory_nodes=[], reason='placement_failed: ' + str(error), launcher_verified=False)
    result['launcher_pid'] = os.getpid()
    return result


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    result = apply_placement(request['gpu_numa_binding'])
    Path(request['gpu_numa_result']).write_text(json.dumps(result) + '\n')
    os.execvpe(sys.argv[2], sys.argv[2:], os.environ)


if __name__ == '__main__':
    main()
