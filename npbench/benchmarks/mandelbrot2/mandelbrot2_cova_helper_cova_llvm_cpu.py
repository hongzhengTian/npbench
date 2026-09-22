from cova import cova
from npbench.benchmarks.mandelbrot2.mandelbrot2_cova_helper_kernels import mandelbrot as _kernel

mandelbrot = cova(backend="cpu", tool="llvm-cpu")(_kernel)
