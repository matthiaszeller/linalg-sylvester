

from time import time
from typing import Callable, Tuple

import numpy as np
from scipy.linalg import schur
from scipy.linalg import solve_sylvester as sp_sylvester


def solve(A: np.ndarray, B: np.ndarray, C: np.ndarray, fun: Callable) -> Tuple[np.ndarray, float, float, float]:
    """Solve the Sylvester equation with given function in three steps and return timing of each step:
        i) Schur decomposition
        ii) Solving upper quasi-triangular matrix equation
        iii) Map back to original coordinate system

    :param fun: function taking 3 input arguments A, B, C and writes the solution in C
    :return: tuple (X, t_schur, t_trisolve, t_back), solution and timings of each step
    """
    # ------ Schur decomposition:
    #       A = U R U^T
    #       B = V S V^T
    t_schur = time()
    R, U = schur(A)
    S, V = schur(B)
    # New system becomes RZ - ZS = D, Z is unknown
    D = np.linalg.multi_dot((U.T, C, V))
    t_schur = time() - t_schur

    # ------ Solve matrix equation RZ - ZS = D
    # Solver modifies D in place
    t_trisolve = time()
    fun(R, S, D)
    t_trisolve = time() - t_trisolve

    # ------ Back transformation
    # Z = U^T X V <=> X = U Z V^T
    t_back = time()
    Z = D
    X = np.linalg.multi_dot((U, Z, V.T))
    t_back = time() - t_back

    return X, t_schur, t_trisolve, t_back


def build_matrices(m, n):
    """Generate random matrices A (mxm), B (nxn), C (mxn)."""
    A = np.random.randn(m, m)
    B = np.random.randn(n, n)
    C = np.random.randn(m, n)
    return A, B, C


def solve_sylvester(A: np.ndarray, B: np.ndarray, C: np.ndarray):
    """Solve the Sylvester equation AX - XB = C with scipy's function by modifying C in place."""
    # Warning, need [:, :] slicing to modify C in place
    # Warning 2, scipy's solve_sylvester function solves AX + XB = C, not AX - XB = C, put minus sign
    C[:, :] = sp_sylvester(A, -B, C)


def gemm(A: np.ndarray, B: np.ndarray, C: np.ndarray):
    """Perform general matrix multiply and add (GEMM) operation in place:
        C <- C + AB
    """
    C += A @ B


def check_sol(A: np.ndarray, B: np.ndarray, C: np.ndarray, x: np.ndarray) -> bool:
    """Check solution of the Sylvester equation"""
    return np.allclose(A @ x - x @ B, C)
