from cova import cova
from npbench.benchmarks.polybench.adi.adi_corrected_cova_kernels import kernel as _kernel

kernel = cova(backend="gpu", tool="llvm-gpu")(_kernel)
