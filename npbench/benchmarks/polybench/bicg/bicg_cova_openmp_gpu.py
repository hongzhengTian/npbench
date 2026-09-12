import numpy as np
from cova import cova


@cova(backend="gpu", tool="openmp-gpu")
def kernel(A, p, r):

    return r @ A, A @ p
