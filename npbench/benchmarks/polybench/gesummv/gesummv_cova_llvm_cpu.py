import numpy as np
from cova import cova


@cova(backend="cpu", tool="llvm-cpu")
def kernel(alpha, beta, A, B, x):

    return alpha * A @ x + beta * B @ x
