"""
Utility functions to perform benchmarks.
"""

import json
from time import time
from typing import List, Dict, Tuple, Callable

import numpy as np
import pandas as pd

from utils import check_sol, build_matrices


def multiple_runs(solve_fun: Callable, n_runs: int, m: int, n: int, check_solution: bool = True):
    """Return computation times for multiple runs.

    :param solve_fun: function taking matrices (np.ndarray) A mxm, B nxn, C mxn and optional keyword arguments
    :param n_runs: number of replications
    :param m: size of matrix A
    :param n: size of matrix B
    :param check_solution: whether to check solution after calling solve_fun
    """
    res = list(np.zeros(n_runs))
    for i in range(n_runs):
        A, B, C = build_matrices(m, n)
        X = C.copy()
        t = time()
        solve_fun(A, B, X)
        t = time() - t
        res[i] = t
        if check_solution:
            assert check_sol(A, B, C, X)

    return res


def vary_matrix_size(solve_fun: Callable,
                     vary: List[int],
                     m: int = None,
                     n: int = None,
                     n_runs: int = 5,
                     check_solution: bool = True):
    """Return computational time of `solve_fun` for different matrix sizes.
    Define either m or n, the other dimension will vary accoding to `vary`.

    :param solve_fun: passed to multiple_runs
    :param vary: array of the matrix size that varies (m or n depending on which one is None)
    :param m: if defined, dimension that is fixed (i.e. n varies)
    :param n: if defined, dimension that is fixed (i.e. m varies)
    :param n_runs: passed to `multiple_runs`
    :param check_solution: passed to `mutliple_runs`
    :return: list of dict with keys m, n, time
    """
    # Determine which matrix size to vary
    if m is None and n is None:
        raise ValueError

    def get_mx_size():
        if m is None:
            for v in vary:
                yield v, n
        else:
            for v in vary:
                yield m, v

    times = []
    for m, n in get_mx_size():
        print(f'({m}, {n})', end=', ')
        times.append({
            'm': m, 'n': n,
            'time': multiple_runs(solve_fun, n_runs, m, n, check_solution)
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

