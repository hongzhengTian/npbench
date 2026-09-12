import numpy as np
from cova import cova

@cova(backend="cpu", tool="llvm-cpu-serial")
def kernel(M, float_n, data):
    return np.cov(np.transpose(data))
