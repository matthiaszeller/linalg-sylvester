import argparse

from benchmark_utils import log_results, vary_matrix_size, benchmark, vary_block_size
from recursive import rtrgsyl
from utils import solve_sylvester_scipy, build_matrices


parser = argparse.ArgumentParser()
parser.add_argument('algorithm', choices=['rtrgsyl', 'scipy'], help='choose algorithm')
parser.add_argument('benchmark', choices=['size', 'blks'], help='choose element to vary')
parser.add_argument('-b', help='block size if size varies', type=int, default=50)
parser.add_argument('-s', help='matrix size if blks varies', type=int, default=1000)
parser.add_argument('-r', '--runs', help='number of runs', type=int, default=5)
args = parser.parse_args()

# --- Select solving function
if args.algorithm == 'scipy':
    solve_fun = solve_sylvester_scipy
elif args.algorithm == 'rtrgsyl':
    solve_fun = rtrgsyl
else:
    raise ValueError('unknown algorithm')


# --- Benchmark running time, varying matrix sizes
if args.benchmark == 'size':
    m = 1500
    n = [100, 200, 500, 1000, 2000]

    solve_args = {'blks': args.b} if args.algorithm == 'rtrgsyl' else dict()
    res = benchmark(benchmark_fun=vary_matrix_size,
                    solve_fun=solve_fun,
                    grid=n,
                    n_runs=args.runs,
                    bargs={
                        'm': m
                    },
                    sargs=solve_args)
    #t = vary_matrix_size(solve, vary=n, m=m, n_runs=5)

    log_results('benchmark.csv', 'scipy', res)

# --- Benchmark running time, varying block size
elif args.benchmark == 'blks':
    if args.algorithm == 'scipy':
        raise ValueError('invalid script arguments, scipy function has no block size')

    # Set big matrix size
    m, n = 1000, 1000
    blks = [10, 20, 50, 100, 200, 300, 500]
    res = benchmark(benchmark_fun=vary_block_size,
                    solve_fun=solve_fun,
                    grid=blks,
                    n_runs=args.runs,
                    m=m,
                    n=n)

