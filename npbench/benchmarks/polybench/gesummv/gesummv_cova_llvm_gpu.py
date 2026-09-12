import numpy as np
from cova import cova


@cova(backend="gpu", tool="llvm-gpu")
def kernel(alpha, beta, A, B, x):

    return alpha * A @ x + beta * B @ x
