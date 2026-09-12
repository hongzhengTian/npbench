import numpy as np
from cova import cova


@cova(backend="cpu", tool="serial-cpu")
def kernel(alpha, beta, A, B, C, D):
    D[:] = alpha * A @ B @ C + beta * D

    return D
