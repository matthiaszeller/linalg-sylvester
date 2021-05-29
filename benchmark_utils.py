"""
Utility functions to perform benchmarks.
"""
import inspect
import json
from time import time
from typing import List, Dict, Tuple, Callable, Iterable, Any

import numpy as np
import pandas as pd

from recursive import rtrgsyl
from utils import check_sol, build_matrices, solve_sylvester_scipy, solve_bartels_stewart, solve_sylvester_linear


def benchmark(solve_fun: Callable,
              vary_param: Tuple[str, Iterable],
              log_context: Dict[str, Any] = None,
              check_solution: bool = True,
              n_runs: int = 5,
              bertel_stewart: bool = False,
              **kwargs):
    if log_context is None:
        log_context = dict()

    variable, values = vary_param
    results = []
    # If the variable that varies is for multiple_runs, it will automatically understand it,
    # otherwise it passes the variable to solve_fun
    for value in values:
        print(f'{variable}={value}')
        run_config = kwargs.copy()
        run_config[variable] = value
        if bertel_stewart:
            times = multiple_runs_bertel_stewart(
                solve_fun=solve_fun,
                n_runs=n_runs,
                check_solution=check_solution,
                **run_config
            )
        else:
            times = multiple_runs(
                solve_fun=solve_fun,
                n_runs=n_runs,
                check_solution=check_solution,
                **run_config
            )
        # Log computational time
        run_config['time'] = times
        # Additional log
        run_config.update(log_context)
        results.append(run_config)

    return results


def multiple_runs(solve_fun: Callable, n_runs: int, dim: Tuple[int, int], check_solution: bool = True, **kwargs):
    """Return computation times for multiple runs.

    :param solve_fun: function taking matrices (np.ndarray) A mxm, B nxn, C mxn and optional keyword arguments
    :param n_runs: number of replications
    :param dim: dimension of matrices (m, n), m size of A, n size of B
    :param check_solution: whether to check solution after calling solve_fun
    :param kwargs: passed to solve_fun
    :return: list of solving times
    """
    m, n = dim
    res = list(np.zeros(n_runs))
    for i in range(n_runs):
        A, B, C = build_matrices(m, n)
        X = C.copy()
        t = time()
        solve_fun(A, B, X, **kwargs)
        t = time() - t
        res[i] = t
        if check_solution:
            assert check_sol(A, B, C, X)

    return res


def multiple_runs_bertel_stewart(solve_fun: Callable,
                                 n_runs: int,
                                 dim: Tuple[int, int],
                                 check_solution: bool = True, **kwargs):
    """Return computation times for multiple runs, calling solve_bartels_stewart.

    :param solve_fun: function taking matrices (np.ndarray) A mxm, B nxn, C mxn and optional keyword arguments
    :param n_runs: number of replications
    :param dim: dimension of matrices (m, n), m size of A, n size of B
    :param check_solution: whether to check solution after calling solve_fun
    :param kwargs: passed to solve_fun
    :return: list tuples (time_schur, time_solve, time_back)
    """
    m, n = dim
    res = []
    for i in range(n_runs):
        A, B, C = build_matrices(m, n)
        r = solve_bartels_stewart(A, B, C, solve_fun, **kwargs)
        res.append(r[1:])
        if check_solution:
            X = r[0]
            assert check_sol(A, B, C, X)

    return res


def multiple_runs_schur(solve_fun: Callable, n_runs: int, m: int, n: int, check_solution: bool = True, **args):
    """
    Return computation times for multiple runs, using utils.solve to wrap solve_fun.
    That is, first map to Schur form, solve with solve_fun, then map back.
    """
    res_schur = list(np.zeros(n_runs))
    res_solve, res_back = res_schur.copy(), res_schur.copy()
    for i in range(n_runs):
        A, B, C = build_matrices(m, n)
        X = C.copy()
        _, res_schur[i], res_solve[i], res_back[i] = solve_bartels_stewart(A, B, C, solve_fun, **args)
        if check_solution:
            assert check_sol(A, B, C, X)

    return res_schur, res_solve, res_back


def vary_block_size(solve_fun: Callable,
                    grid: Iterable[int],
                    m: int,
                    n: int,
                    n_runs: int = 5,
                    check_solution: bool = True):
    res = []
    for e in grid:
        print(f'blks={e}', end=', ')
        res_schur, res_solve, res_back = multiple_runs_schur(solve_fun, n_runs, m, n, check_solution, blks=e)
        res.append({
            'blk': e,
            'time_solve': res_solve,
            'time_schur': res_schur,
            'time_back': res_back
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
