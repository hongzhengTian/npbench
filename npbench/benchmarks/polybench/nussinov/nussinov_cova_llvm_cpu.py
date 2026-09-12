import numpy as np
from cova import cova


@cova(backend="cpu", tool="llvm-cpu")
def kernel(N, seq):

    table = np.zeros((N, N), np.int32)

    for i in range(N - 1, -1, -1):
        for j in range(i + 1, N):
            if j - 1 >= 0:
                table[i, j] = max(table[i, j], table[i, j - 1])
            if i + 1 < N:
                table[i, j] = max(table[i, j], table[i + 1, j])
            if j - 1 >= 0 and i + 1 < N:
                if i < j - 1:
                    if seq[i] + seq[j] == 3:
                        pair_score = 1
                    else:
                        pair_score = 0
                    table[i,
                          j] = max(table[i, j],
                                   table[i + 1, j - 1] + pair_score)
                else:
                    table[i, j] = max(table[i, j], table[i + 1, j - 1])
            for k in range(i + 1, j):
                table[i, j] = max(table[i, j], table[i, k] + table[k + 1, j])

    return table
