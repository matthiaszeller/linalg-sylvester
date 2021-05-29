"""
Utility functions to perform benchmarks.
"""

from time import time
from typing import Dict, Tuple, Callable, Iterable, Any

import numpy as np

from utils import check_sol, build_matrices, solve_bartels_stewart


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

