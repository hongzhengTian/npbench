import numpy as np
from cova import cova


@cova(backend="gpu", tool="llvm-gpu")
def kernel(A, x):

    return (A @ x) @ A
