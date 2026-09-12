import numpy as np
from cova import cova


@cova(backend="gpu", tool="llvm-gpu")
def kernel(alpha, beta, C, A, B):

    C[:] = alpha * A @ B + beta * C

    return C
