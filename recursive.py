

from math import floor

import numpy as np

from utils import gemm, solve_sylvester_scipy


def split_matrix(M: np.ndarray):
    """Find the splitting point which halves the matrix quasi upper-triangular M,
    without cutting 2x2 diagonal blocks.

    :param M: upper quasi-triangular matrix M of dimension nxn, n > 2
    :return: int, splitting point `m`. Assumes splitting between m and m+1
    """
    n = M.shape[0]
    m = floor(n / 2) - 1 # -1 since 0-indexed
    # If splitting point occurs at 2x2 diagonal block, split under
    # the diagonal block. We detect 2x2 diagonal blocks when item (m+1, m)
    # is non zero. Since n > 2, m+1 will never be out of bounds
    if M[m+1, m] != 0.0:
        m += 1 # cut right under the 2x2 diagonal block

    return m


def rtrgsyl(A: np.ndarray, B: np.ndarray, C: np.ndarray, blks: int):
    """
    Solver for continuous-time Sylvester matrix equation:
        AX - XB = C

    Assumes A, B, C are quasi upper-triangular. C will be overwritten.

    :param A: mxm matrix
    :param B: nxn matrix
    :param C: mxn matrix, this will be overwritten
    :param blks: block size under which we switch to standard algo for solving small-sized equation
    :return: None, the solution overwrites C
    """
    m, n = A.shape[0], B.shape[0]
    # If size if small enough, solve with standard algo
    if m <= blks and n <= blks:
        solve_sylvester_scipy(A, B, C)
    else:
        # Implementation note: slicing a numpy array A[a:b, c:d] returns a VIEW on the matrix, not a copy
        #                      modifying the view thus modifies the original matrix
        if n <= m/2:
            # Split A by rows & cols, C by rows
            k = split_matrix(A)
            A11 = A[:k+1, :k+1]
            A12 = A[:k+1, k+1:]
            A22 = A[k+1:, k+1:]
            C1 = C[:k+1, :]
            C2 = C[k+1:, :]
            # Recursion: solve for X2, store it in C2
            rtrgsyl(A22, B, C2, blks)
            X2 = C2 # for code readability
            # Modify C1 in place
            gemm(-A12, X2, C1)
            # Recursion: solve for X1, store it in C1
            rtrgsyl(A11, B, C1, blks)
        elif m <= n/2:
            # Split B by rows & cols, C by cols
            k = split_matrix(B)
            B11 = B[:k+1, :k+1]
            B12 = B[:k+1, k+1:]
            B22 = B[k+1:, k+1:]
            C1 = C[:, :k+1]
            C2 = C[:, k+1:]
            # Recursion: solve for X1, store it in C1
            rtrgsyl(A, B11, C1, blks)
            X1 = C1
            # Update C2 in place
            gemm(X1, B12, C2)
            # Recursion: solve for X2, store it in C2
            rtrgsyl(A, B22, C2, blks)
        else: # m, n >= blks
            # Split A, B, C by rows and columns
            # Index at which to split A and rows of C
            kA = split_matrix(A)
            # Index at which to split B and columns of C
            kB = split_matrix(B)
            A11 = A[:kA+1, :kA+1]
            A12 = A[:kA+1, kA+1:]
            A22 = A[kA+1:, kA+1:]

            B11 = B[:kB+1, :kB+1]
            B12 = B[:kB+1, kB+1:]
            B22 = B[kB+1:, kB+1:]

            C11 = C[:kA+1, :kB+1]
            C12 = C[:kA+1, kB+1:]
            C21 = C[kA+1:, :kB+1]
            C22 = C[kA+1:, kB+1:]
            # Recursion: must first solve for X21, store it in C21
            rtrgsyl(A22, B11, C21, blks)
            X21 = C21 # this is just for code readability
            # Update C11, C22 in place
            gemm(-A12, X21, C11)
            gemm(C21, B12, C22)
            # Independent recursion steps: solve for X11, X22
            rtrgsyl(A22, B22, C22, blks)
            rtrgsyl(A11, B11, C11, blks)
            X22 = C22 # code readability
            X11 = C11
            # Update C12
            gemm(-A12, X22, C12)
            gemm(X11, B12, C12)
            # Recursion: solve for X12
            rtrgsyl(A11, B22, C12, blks)

