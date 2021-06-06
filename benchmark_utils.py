"""
Utility functions to perform benchmarks.
"""

from time import time
from typing import Dict, Tuple, Callable, Iterable, Any, Union

import numpy as np

from utils import check_sol, build_matrices, solve_bartels_stewart


def benchmark(solve_fun: Callable,
              vary_param: Tuple[str, Iterable],
              log_context: Dict[str, Any] = None,
              check_solution: bool = True,
              n_runs: Union[int, Tuple[int, int]] = 5,
              bartel_stewart: bool = False,
              **kwargs):
    """Interface function for benchmarking.
    :param n_runs: fixed if int. Provide boundaries (nruns_min, nruns_max) to vary nruns in function of matrix size
    """
    if log_context is None:
        log_context = dict()
    if isinstance(n_runs, int):
        get_nruns = lambda _: n_runs
    else:
        if vary_param[0] != 'dim':
            raise ValueError
        nmin, nmax = min(e[0] for e in vary_param[1]), max(e[0] for e in vary_param[1])
        get_nruns = lambda n: compute_nruns(nmin, max(n_runs), nmax, min(n_runs), n)

    variable, values = vary_param
    results = []
    # If the variable that varies is for multiple_runs, it will automatically understand it,
    # otherwise it passes the variable to solve_fun
    for value in values:
        r = get_nruns(value[0] if variable == 'dim' else kwargs['dim'][0])
        print(f'{variable}={value} {r} runs')
        run_config = kwargs.copy()
        run_config[variable] = value
        if bartel_stewart:
            times = multiple_runs_bertel_stewart(
                solve_fun=solve_fun,
                n_runs=r,
                check_solution=check_solution,
                **run_config
            )
        else:
            times = multiple_runs(
                solve_fun=solve_fun,
                n_runs=r,
                check_solution=check_solution,
                **run_config
            )
        # Log computational time
        run_config['time'] = times
        # Additional log
        run_config.update(log_context)
        results.append(run_config)

    return results


def compute_nruns(nmin: int, nruns: int, nmax: int, nruns2: int, n: int):
    """Compute the number of runs given the matrix size n."""
    nmin, nmax, n = np.log([nmin, nmax, n])
    slope = (nruns2 - nruns) / (nmax - nmin)
    r = (n - nmin) * slope + nruns
    r = np.clip(r, nruns2, nruns)
    return int(r)


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
    i = 0
    while i < n_runs:
        A, B, C = build_matrices(m, n)
        X = C.copy()
        t = time()
        solve_fun(A, B, X, **kwargs)
        t = time() - t
        res[i] = t
        if check_solution:
            if check_sol(A, B, C, X):
                i += 1
            else:
                print('WARNING: incorrect solution, retrying...')
        else:
            i += 1

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
    i = 0
    while i < n_runs:
        A, B, C = build_matrices(m, n)
        r = solve_bartels_stewart(A, B, C, solve_fun, **kwargs)
        res.append(r[1:])
        if check_solution:
            X = r[0]
            if check_sol(A, B, C, X):
                i += 1
            else:
                print('WARNING: incorrect solution, retrying...')
        else:
            i += 1

    return res

