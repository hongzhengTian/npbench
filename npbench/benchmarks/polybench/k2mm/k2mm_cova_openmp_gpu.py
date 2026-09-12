import numpy as np
from cova import cova


@cova(backend="gpu", tool="openmp-gpu")
def kernel(alpha, beta, A, B, C, D):
    D[:] = alpha * A @ B @ C + beta * D

    return D
