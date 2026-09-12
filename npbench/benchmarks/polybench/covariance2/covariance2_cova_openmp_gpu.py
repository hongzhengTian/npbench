import numpy as np
from cova import cova

@cova(backend="gpu", tool="openmp-gpu")
def kernel(M, float_n, data):
    return np.cov(np.transpose(data))
