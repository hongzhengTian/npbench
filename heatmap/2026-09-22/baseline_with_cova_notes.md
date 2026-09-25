# Baseline plus existing CoVA observations

These files add six CoVA routes to the original fourteen baseline routes for the same 54 L workloads.
The original `baseline_heatmaps_data.json` and three `baseline_*_heatmap.png` files are unchanged.
No benchmark, golden producer, compiler or GPU workload was executed to generate this view.

## Outputs

- `baseline_with_cova_heatmaps_data.json`: all original baseline cells unchanged, plus CoVA observations, scalar samples, source hashes, identities, restrictions and superseded selections.
- `baseline_with_cova_results.csv`: a flat table for follow-up analysis.
- `baseline_with_cova_initialization_heatmap.png`: initialization, including lazy compilation where measured.
- `baseline_with_cova_fresh_process_heatmap.png`: first call in a later process using persistent artifacts.
- `baseline_with_cova_same_process_heatmap.png`: subsequent calls, aggregated as the median of per-process medians.
- `baseline_with_cova_verification.json`: checks and hashes of the unchanged original files.

All three new figures use one shared logarithmic color scale in seconds.
Colors show absolute times, not speedups or a certified ranking.
The added CPU columns are EmitC serial, EmitC OpenMP, LLVM serial and LLVM OpenMP; GPU columns are OpenMP target and LLVM CUDA.
GPU column headings describe requested routes, not a guarantee of actual GPU numerical execution.

## Selection and restrictions

The selector starts from `2026-09-21-npbench-cova-L`, overlays L cells in each closure stage's `audit.json/final` in chronological order, then overlays the final optimization candidate cohorts.
It never selects a batch because its time is smaller, never pools samples across versions, and never substitutes an S or diagnostic middle-sized run for L.
Later failed cohorts supersede earlier success instead of retaining a favorable historical timing.
Missing lifecycle phases remain missing rather than falling back to a different batch.

The latest optimization observations cover azimint_naive LLVM CPU/GPU and the GEMM/Mandelbrot1 LLVM GPU controls.
The latest azimint CPU cohort has one restored process; azimint GPU and both controls have three restored processes.
Their initialization is one separate bootstrap call; fresh-process values use the first restored call, and same-process values use the remaining two calls per process.
The diagnostic driver uses the full Region call boundary and canonical strict validation; it does not carry the full lifecycle harness row schema, so its origin remains explicitly marked.
Latest azimint GPU restored calls were profiled, while its initialization was not.
No throughput-only or device-resident timing replaces a host-to-host call.

`H` marks an older CoVA build, including closure-stage observations, rather than a full revalidation of that route on the current compiler.
`P` marks the profiled azimint GPU restored observations.
`CPU` marks a requested GPU route whose historical placement reported only CPU; `GPU?` means primary GPU execution is not independently verified for that selected cohort.
The newest GEMM/Mandelbrot1 control cohorts have no per-process trace in their driver output and therefore retain `GPU?`; prior closure GPU qualification is not silently promoted to a fresh trace result.
`*` retains baseline restrictions and marks all CoVA values as reference observations.
`†` marks explicit baseline repairs, corrected ADI or a CoVA helper.

The azimint_hist, Mandelbrot2 and Stockham helpers are used only for routes with measured helper evidence and matching original golden identities.
The adi_corrected row uses only corrected ADI results with the identical corrected golden; its four unmeasured CoVA routes remain missing.
All displayed CoVA timings match the corresponding baseline NumPy golden key and payload SHA256.
Numerical failures, errors and timeouts have no displayed time, and nbody gains no passing observation.
Historical errors describe the measured build, not a claim that the current compiler still fails that program.

This is a reference view combining different collection dates and compiler versions, not a new all-route acceptance matrix or a synchronized cross-system comparison.
Original baseline restrictions, resource limitations and source repairs remain applicable.
No raw input arrays, native binaries or profiler files are copied into these outputs; only scalar measurements and their provenance are added.

## Reproduction

From the NPBench root, with Python, matplotlib and the Research evidence checkout available:

```bash
python scripts/plot_baseline_with_cova.py \
  --baseline baseline_heatmaps_data.json \
  --evidence-root /HSC/users/tianhong/myHome/myResearchLife/notebook/01-research/cova/reports/evidence \
  --output .
python -m unittest discover -s tests -p test_cova_heatmap_selection.py -v
```

`plot_baseline_heatmaps.py` remains the original baseline entry point.
Its renderer now accepts optional presentation parameters so the combined view can reuse the drawing implementation without modifying original data or outputs.
