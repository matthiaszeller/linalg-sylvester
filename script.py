from time import time

import numpy as np
from scipy.linalg import schur
from typing import Tuple, Callable
from recursive import rtrgsyl
from utils import build_matrices, solve_bartels_stewart, check_sol, solve_sylvester_linear

m, n = 1000, 1000
blks = 20

mx = build_matrices(m, n)
A, B, C = mx

print(f'Created random matrices with shapes:\n{", ".join(str(M.shape) for M in mx)}')

print('Solving Sylvester equation with Bartels Stewart and rtrgsyl ...')

X, t_schur, t_solve, t_back = solve_bartels_stewart(A, B, C, rtrgsyl, blks=blks)

print('Checking validity of solution by plugging X into equation...')
assert check_sol(A, B, C, X)

print('\nsolving times:')
print(f'i) schur decomp : \t{t_schur:.3}')
print(f'ii) rtrgsyl: \t\t{t_solve:.3}')
print(f'iii) map back: \t\t{t_back:.3}')


print('Solving Sylvester equation with Bartels Stewart and solving small systems as linear systems ...')

X, t_schur, t_solve, t_back = solve_bartels_stewart(A, B, C, rtrgsyl, blks=blks, std_solver=solve_sylvester_linear)
assert check_sol(A, B, C, X)

print('\nsolving times:')
print(f'i) schur decomp : \t{t_schur:.3}')
print(f'ii) rtrgsyl: \t\t{t_solve:.3}')
print(f'iii) map back: \t\t{t_back:.3}')
