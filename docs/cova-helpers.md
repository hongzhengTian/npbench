# Explicit CoVA helper adaptations

These optional registrations retain the original mathematical workloads and reuse their exact canonical input/output bundles through `golden_reference`.
They do not replace original implementations or increase the native benchmark denominator.
Keep helper coverage separate from original-source compilation and scientific workload repairs.

| Registration | Adaptation | Supported routes |
| --- | --- | --- |
| `azimint_hist_cova_helper` | Explicit equal-width edges, boundary-corrected bin assignment, independently owned partial histograms, and ordered partial reduction. | LLVM CPU, LLVM GPU, OpenMP GPU. |
| `mandelbrot2_cova_helper` | Replace `mgrid`, bind linspace results before indexing, and express shape rebinding with reshape; retain shrinking selections, ufunc out, paired scatter, and Mandelbrot iteration. | LLVM CPU and LLVM GPU. |
| `stockham_fft_cova_helper` | Replace integer `mgrid` coordinates with an explicit compiled grid helper; retain Stockham stages, repeat, transpose, reshape, complex twiddles and matrix multiplication. | LLVM CPU and LLVM GPU. |

A registered GPU route is not by itself evidence that its main computation runs on the device.
Inspect placement, generated code, strict results and actual kernel activity for the selected size.

The histogram helper accepts one-dimensional finite float64 coordinates and matching float64 weights with a positive specialized integer bin count and finite representable range.
It returns int64 counts, float64 weighted sums and float64 edges, including the rightmost endpoint in the last bin.
Equal-valued coordinates use the NumPy half-unit range extension; empty coordinates use the interval [0, 1].
Explicit edges, density, automatic bin rules, nonfinite coordinates, and other dtype combinations are outside this helper contract.
At most 16 independently owned partial histograms bound temporary storage independently of input length; storage still scales with the number of bins.
Each chunk accumulates in input order, followed by a chunk-order reduction; floating results are checked against NumPy under the unchanged benchmark policy, without a claim of bitwise identical accumulation order.
There are no shared-bin updates or scheduler/thread-ID assumptions.

The FFT helper keeps the radix/stage algorithm and complex128 output, with N = R**K, R >= 2 and K >= 0.
It returns the updated `y` to satisfy CoVA's returned-input writeback contract; the NPBench adapter retains the original zero-return workload contract.
The reference remains the original Stockham implementation; the unit test additionally checks a library FFT as an independent oracle, never as the tested implementation.

The Mandelbrot helper uses the original float64 grid, complex128 iteration, int64 escape counts, and transposed output order.
Its tests cover no iterations, no escaping points, and arrays that shrink to zero.
The original Mandelbrot2 canonical reference and strict comparison policy remain unchanged.

Run pure numerical and registration checks with `python tests/test_cova_helpers.py`.
Add `llvm-cpu` or `llvm-gpu` to execute the helpers through CoVA, then run the same command with `restore` appended in the same artifact directory to forbid frontend and native recompilation.
For `openmp-gpu`, select `CoVAHelperTests.test_histogram_edges_counts_weights` explicitly after the tool argument.
Use `--histogram-case 0` through `--histogram-case 6` to run each boundary case in its own process and artifact directory when diagnosing specialization-loading failures.
Isolated case success does not establish that all specializations can coexist in one process.
Use the existing lifecycle runner with `--require-golden` for measurements, preserving all numerical and input-state checks.
