import numpy as np
from cova import cova


@cova(backend="cpu", tool="openmp-cpu")
def kernel(A, p, r):

    return r @ A, A @ p
