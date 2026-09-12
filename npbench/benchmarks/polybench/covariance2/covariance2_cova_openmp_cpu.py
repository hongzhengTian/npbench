import numpy as np
from cova import cova

@cova(backend="cpu", tool="openmp-cpu")
def kernel(M, float_n, data):
    return np.cov(np.transpose(data))
