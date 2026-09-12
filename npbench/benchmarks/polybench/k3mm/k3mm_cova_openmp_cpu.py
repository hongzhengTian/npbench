import numpy as np
from cova import cova


@cova(backend="cpu", tool="openmp-cpu")
def kernel(A, B, C, D):

    return A @ B @ C @ D
