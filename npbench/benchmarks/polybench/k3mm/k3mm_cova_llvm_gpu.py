import numpy as np
from cova import cova


@cova(backend="gpu", tool="llvm-gpu")
def kernel(A, B, C, D):

    return A @ B @ C @ D
