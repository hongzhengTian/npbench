import numpy as np
from cova import cova


@cova(backend="cpu", tool="llvm-cpu")
def kernel(A, x):

    return (A @ x) @ A
