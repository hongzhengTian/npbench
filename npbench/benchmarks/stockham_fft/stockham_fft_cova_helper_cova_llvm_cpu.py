from cova import cova
from npbench.benchmarks.stockham_fft.stockham_fft_cova_helper_kernels import stockham_fft as _kernel

stockham_fft = cova(backend="cpu", tool="llvm-cpu")(_kernel)
