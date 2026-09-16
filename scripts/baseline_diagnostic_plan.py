"""Small, fixed diagnostic matrix; no CoVA or full-campaign retry selection."""
GROUPS = {
    'correctness': [('cholesky','dace_gpu','auto_opt'),('cholesky','dace_gpu','parallel'),
                    ('syrk','dace_gpu','auto_opt'),('syrk','dace_gpu','parallel')],
    'restore': [('channel_flow','dace_gpu','auto_opt'),('softmax','dace_gpu','auto_opt'),
                ('channel_flow','dace_cpu','auto_opt'),('softmax','dace_cpu','auto_opt')],
    'contracts': [('adi','dace_cpu','fusion'),('adi','numba','nopython-mode'),('adi','cupy','default'),
                  ('correlation','cupy','default'),('nbody','dace_cpu','fusion'),('nbody','dace_gpu','fusion'),
                  ('mlp','numba','nopython-mode'),('resnet','numba','nopython-mode')],
    'performance': [('floyd_warshall','dace_cpu','parallel'),('floyd_warshall','numpy','default'),
                    ('spmv','dace_gpu','fusion'),('spmv','dace_gpu','parallel'),('spmv','dace_cpu','auto_opt'),
                    ('correlation','numba','nopython-mode'),('correlation','numpy','default'),
                    ('covariance','numba','nopython-mode'),('covariance','numpy','default'),
                    ('nussinov','dace_gpu','auto_opt'),('nussinov','dace_cpu','auto_opt'),
                    ('arc_distance','cupy','default')],
    'p2': [('stockham_fft','dace_cpu','fusion'),('stockham_fft','dace_gpu','fusion'),
           ('mandelbrot2','dace_cpu','fusion'),('correlation','numba','nopython-mode-parallel'),
           ('covariance','numba','nopython-mode-parallel'),('lenet','numba','object-mode'),
           ('lenet','dace_gpu','fusion'),('lenet','dace_gpu','parallel'),
           ('azimint_naive','dace_gpu','parallel'),('resnet','dace_gpu','parallel'),
           ('durbin','numba','nopython-mode-parallel'),('cavity_flow','numba','nopython-mode-parallel'),
           ('mlp','dace_gpu','fusion'),('mandelbrot1','numba','nopython-mode'),('mandelbrot1','dace_gpu','fusion'),
           ('azimint_hist','numba','nopython-mode-parallel-range'),('seidel_2d','numba','nopython-mode-parallel'),
           ('jacobi_1d','dace_gpu','fusion'),('jacobi_2d','dace_gpu','fusion'),
           ('contour_integral','dace_gpu','fusion'),('conv2d_bias','dace_gpu','fusion'),
           ('nbody','numba','nopython-mode'),('nbody','cupy','default')],
    # Opt-in, using existing L golden. Never turns p2 into an L campaign.
    'large_spots': [('cholesky','dace_gpu','auto_opt'),('syrk','dace_gpu','auto_opt'),
                    ('floyd_warshall','dace_cpu','parallel'),('spmv','dace_gpu','fusion'),('spmv','dace_gpu','parallel')],
}

if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('group', choices=GROUPS)
    args=parser.parse_args()
    for row in GROUPS[args.group]:print('\t'.join(row))
