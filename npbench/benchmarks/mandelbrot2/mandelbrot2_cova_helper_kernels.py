# Explicit equivalent grid/reshape prelude; see docs/cova-helpers.md.
import numpy as np

def grid(rows, cols):
    a = np.empty((rows, cols), dtype=np.int64)
    b = np.empty((rows, cols), dtype=np.int64)
    for row in range(rows):
        a[row, :] = row
    for col in range(cols):
        b[:, col] = col
    return a, b


def mandelbrot(xmin, xmax, ymin, ymax, xn, yn, itermax, horizon=2.0):
    Xi, Yi = grid(xn, yn)
    x_values = np.linspace(xmin, xmax, xn, dtype=np.float64)
    y_values = np.linspace(ymin, ymax, yn, dtype=np.float64)
    X = x_values[Xi]
    Y = y_values[Yi]
    C = X + Y * 1j
    N_ = np.zeros(C.shape, dtype=np.int64)
    Z_ = np.zeros(C.shape, dtype=np.complex128)
    Xi = np.reshape(Xi, (xn * yn,))
    Yi = np.reshape(Yi, (xn * yn,))
    C = np.reshape(C, (xn * yn,))
    Z = np.zeros(C.shape, np.complex128)
    for i in range(itermax):
        if not len(Z):
            break
        np.multiply(Z, Z, Z)
        np.add(Z, C, Z)
        I = abs(Z) > horizon
        N_[Xi[I], Yi[I]] = i + 1
        Z_[Xi[I], Yi[I]] = Z[I]
        np.logical_not(I, I)
        Z = Z[I]
        Xi, Yi = Xi[I], Yi[I]
        C = C[I]
    return Z_.T, N_.T
