import numpy as np
from cova import cova


@cova(backend="cpu", tool="serial-cpu")
def kernel(NR, NQ, NP, A, C4):

    # for r in range(NR):
    #     for q in range(NQ):
    #         sum[:] = A[r, q, :] @ C4
    #         A[r, q, :] = sum
    A[:] = np.reshape(np.reshape(A, (NR, NQ, 1, NP)) @ C4, (NR, NQ, NP))

    return A
