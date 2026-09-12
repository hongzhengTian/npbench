"""Adapt CoVA's returned arrays to NPBench's return and mutation protocol."""

from functools import wraps
from importlib.metadata import PackageNotFoundError, version
import os

import numpy as np

from .framework import Framework


class CovaFramework(Framework):
    """Call public decorated kernels with host NumPy arrays.

    Kernels return the reference results followed by the arrays to write back.
    Native GPU calls include transfers and synchronization before returning.
    """

    def version(self):
        if os.environ.get("NPBENCH_COVA_VERSION"):
            return os.environ["NPBENCH_COVA_VERSION"]
        try:
            return version("cova")
        except PackageNotFoundError:
            return "source-unversioned"

    def implementations(self, bench):
        contract = bench.info["cova"]
        return_count = contract["return_count"]
        writeback_indices = [bench.info["input_args"].index(name)
                             for name in contract["writeback_args"]]
        expected_count = return_count + len(writeback_indices)
        (compiled, label), = super().implementations(bench)

        @wraps(compiled)
        def invoke(*args):
            result = compiled(*args)
            values = (list(result) if isinstance(result, (tuple, list))
                      else [] if result is None else [result])
            if len(values) != expected_count:
                raise ValueError(
                    f"CoVA returned {len(values)} outputs; "
                    f"expected {expected_count} for {bench.bname}")
            updates = list(zip(writeback_indices, values[return_count:]))
            for index, value in updates:
                if not isinstance(value, np.ndarray):
                    raise TypeError("CoVA writeback output must be a NumPy array")
                if value.shape != args[index].shape or value.dtype != args[index].dtype:
                    raise ValueError("CoVA writeback output shape or dtype mismatch")
            for index, value in updates:
                np.copyto(args[index], value, casting="no")
            if return_count == 0:
                return None
            if return_count == 1:
                return values[0]
            return tuple(values[:return_count])

        return [(invoke, label)]
