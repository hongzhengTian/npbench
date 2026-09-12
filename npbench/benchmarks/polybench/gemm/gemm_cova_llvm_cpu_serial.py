import numpy as np
from cova import cova


@cova(backend="cpu", tool="llvm-cpu-serial")
def kernel(alpha, beta, C, A, B):

    C[:] = alpha * A @ B + beta * C

    return C
