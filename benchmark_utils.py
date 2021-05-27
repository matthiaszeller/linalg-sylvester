"""
Utility functions to perform benchmarks.
"""

import json
from time import time
from typing import List, Dict, Tuple, Callable, Iterable

import numpy as np
import pandas as pd

from recursive import rtrgsyl
from utils import check_sol, build_matrices, solve_sylvester_scipy


def multiple_runs(solve_fun: Callable, n_runs: int, m: int, n: int, check_solution: bool = True, **args):
    """Return computation times for multiple runs.

    :param solve_fun: function taking matrices (np.ndarray) A mxm, B nxn, C mxn and optional keyword arguments
    :param n_runs: number of replications
    :param m: size of matrix A
    :param n: size of matrix B
    :param check_solution: whether to check solution after calling solve_fun
    :param args: passed to solve_fun
    :return: list of solving times
    """
    res = list(np.zeros(n_runs))
    for i in range(n_runs):
        A, B, C = build_matrices(m, n)
        X = C.copy()
        t = time()
        solve_fun(A, B, X, **args)
        t = time() - t
        res[i] = t
        if check_solution:
            assert check_sol(A, B, C, X)

    return res


def vary_block_size(solve_fun: Callable,
                    grid: Iterable[int],
                    m: int,
                    n: int,
                    n_runs: int = 5,
                    check_solution: bool = True):
    res = []
    for e in grid:
        print(f'blks={e}', end=', ')
        times = multiple_runs(solve_fun, n_runs, m, n, check_solution)
        res.append({
            'blk': e,
            'time': times
        })

    return res


def vary_matrix_size(solve_fun: Callable,
                     grid: Iterable[int],
                     m: int = None,
                     n: int = None,
                     n_runs: int = 5,
                     check_solution: bool = True,
                     **args):
    """Return computational time of `solve_fun` for different matrix sizes.
    Define either m or n, the other dimension will vary accoding to `vary`.

    :param solve_fun: passed to multiple_runs
    :param grid: array of the matrix size that varies (m or n depending on which one is None)
    :param m: if defined, dimension that is fixed (i.e. n varies)
    :param n: if defined, dimension that is fixed (i.e. m varies)
    :param n_runs: passed to `multiple_runs`
    :param check_solution: passed to `mutliple_runs`
    :param args: passed to solve_fun
    :return: list of dict with keys m, n, time
    """
    # Determine which matrix size to vary
    if m is None and n is None:
        raise ValueError

    def get_mx_size():
        if m is None:
            for v in grid:
                yield v, n
        else:
            for v in grid:
                yield m, v

    times = []
    for m, n in get_mx_size():
        print(f'({m}, {n})', end=', ')
        times.append({
            'm': m, 'n': n,
            'time': multiple_runs(solve_fun, n_runs, m, n, check_solution, **args)
        })
    return times


def log_results(fname: str, desc: str, data: List[Dict], blks=np.nan):
    """Append data in file in csv format.
    Format has columns:
        - desc
        - blks, if provided
        - m
        - n
        - all following columns are the computational times

    :param fname: file name to write in
    :param data: list of dict
    """
    df = pd.DataFrame(data)
    # Reorder columns
    cols = ['m', 'n', 'time']
    df = df[cols]
    # Additional description
    df['desc'] = desc
    df['blks'] = blks
    # Extend series of list (of times) to new columns
    df_times = pd.DataFrame.from_dict(
        dict(zip(df.index, df.time.values))
    ).T

    df = df.join(df_times).drop(columns='time')

    df.to_csv(fname, mode='a', index=False, header=False)


MAP_SOLVE_FUN_NAME = {
    rtrgsyl: 'rtrgsyl',
    solve_sylvester_scipy: 'scipy'
}

MAP_BECHMARK_FUN_NAME = {
    vary_matrix_size: 'size',
    vary_block_size: 'blks'
}


def benchmark(benchmark_fun: Callable, solve_fun: Callable, grid: Iterable, n_runs: int, bargs=None, sargs=None):
    """
    Benchmark utility function, enabling easy logging. Wraps another benchmark function.

    :param benchmark_fun: the function that performs benchmark
    :param solve_fun: the solving function
    :param grid: grid of values to benchmark
    :param n_runs: number of repetitions
    :param bargs: dic, arguments passed to the benchmark_function, will be stored in the log
    :param sargs: dic, arguments passed to the solve_fun, will be stored in the log
    """
    if bargs is None:
        bargs = dict()
    if sargs is None:
        sargs = dict()

    print('Perform benchmark on {} using algorithm {}'
          .format(MAP_BECHMARK_FUN_NAME[benchmark_fun], MAP_SOLVE_FUN_NAME[solve_fun]))
    print('args:', ', '.join(f'{k}={v}' for k, v in bargs.items()))

    res = benchmark_fun(solve_fun=solve_fun, grid=grid, n_runs=n_runs, **bargs, **sargs)
    # Log additional arguments
    for dic in res:
        dic.update(bargs)
        dic.update(sargs)
        dic['algorithm'] = MAP_SOLVE_FUN_NAME[solve_fun]
        dic['benchmark'] = MAP_BECHMARK_FUN_NAME[benchmark_fun]

    return res
