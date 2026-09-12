import numpy as np
from cova import cova


@cova(backend="cpu", tool="llvm-cpu")
def kernel(A):
    A[:] = np.linalg.cholesky(A) + np.triu(A, k=1)

    return A
