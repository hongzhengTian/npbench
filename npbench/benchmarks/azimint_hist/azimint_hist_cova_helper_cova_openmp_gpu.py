from cova import cova
from npbench.benchmarks.azimint_hist.azimint_hist_cova_helper_kernels import azimint_hist as _kernel

azimint_hist = cova(backend="gpu", tool="openmp-gpu")(_kernel)
