# Explicit mathematical-equivalence adaptation; see docs/cova-helpers.md.
import numpy as np


def histogram_counts_weights(a, bins, weights):
    # Contract: finite float64 coordinates/weights and a positive scalar bin count.
    lower = 0.0
    upper = 1.0
    if a.shape[0] > 0:
        lower = np.min(a)
        upper = np.max(a)
    if lower == upper:
        lower -= 0.5
        upper += 0.5
    edges = np.linspace(lower, upper, bins + 1)
    # Bound partial storage independently of input length. A chunk owns all
    # its buckets; no scheduler or device thread identifier enters indexing.
    chunks = min(16, (a.shape[0] + 65535) // 65536)
    chunk_size = 0
    if chunks > 0:
        chunk_size = (a.shape[0] + chunks - 1) // chunks
    counts = np.zeros((chunks, bins), dtype=np.int64)
    sums = np.zeros((chunks, bins), dtype=weights.dtype)
    for chunk in range(chunks):
        for i in range(chunk * chunk_size, min((chunk + 1) * chunk_size, a.shape[0])):
            index = int((a[i] - lower) / (upper - lower) * bins)
            if index == bins:
                index = bins - 1
            if a[i] < edges[index]:
                index -= 1
            if index < bins - 1 and a[i] >= edges[index + 1]:
                index += 1
            counts[chunk, index] += 1
            sums[chunk, index] += weights[i]
    return np.sum(counts, axis=0), np.sum(sums, axis=0), edges


def azimint_hist(data, radius, npt):
    counts, sums, edges = histogram_counts_weights(radius, npt, data)
    return sums / counts
