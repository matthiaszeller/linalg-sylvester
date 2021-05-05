from time import time

import numpy as np
from scipy.linalg import schur
from typing import Tuple, Callable
from recursive import rtrgsyl

import argparse

from utils import build_matrices, solve

DEFAULT_M = 100
DEFAULT_N = 100
DEFAULT_BLKS = 30


def print_times(t1, t2, t3):
    print(f'Schur: {t1:.3}\nSolve: {t2:.3}\nBack: {t3:.3}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('m', help='size of matrix A', type=int, default=DEFAULT_M, nargs='?')
    parser.add_argument('n', help='size of matrix B', type=int, default=DEFAULT_N, nargs='?')
    parser.add_argument('blks', help='max size to solve in standard way', type=int, default=DEFAULT_BLKS, nargs='?')
    args = parser.parse_args()

    m = args.m
    n = args.n
    blks = args.blks

    A, B, C = build_matrices(m, n)
    fun = lambda A, B, C: rtrgsyl(A, B, C, blks)
    X, t1, t2, t3 = solve(A, B, C, fun)
    print_times(t1, t2, t3)

