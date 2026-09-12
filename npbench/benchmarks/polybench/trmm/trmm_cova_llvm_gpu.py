import numpy as np
from cova import cova


@cova(backend="gpu", tool="llvm-gpu")
def kernel(alpha, A, B):

    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            B[i, j] += np.dot(A[i + 1:, i], B[i + 1:, j])
    B *= alpha

    return B
