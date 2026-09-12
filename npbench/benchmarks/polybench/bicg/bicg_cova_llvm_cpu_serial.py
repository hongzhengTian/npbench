import numpy as np
from cova import cova


@cova(backend="cpu", tool="llvm-cpu-serial")
def kernel(A, p, r):

    return r @ A, A @ p
