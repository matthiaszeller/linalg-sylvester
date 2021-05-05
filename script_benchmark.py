

from benchmark_utils import log_results, vary_matrix_size
from recursive import rtrgsyl
from utils import solve_sylvester


# m = 1500
# n = [100, 200, 500, 1000, 2000]
#
# solve = solve_sylvester
# t = vary_matrix_size(solve, vary=n, m=m, n_runs=5)
#
# log_results('benchmark.csv', 'scipy', t)


m = 200
n = [100, 200, 500, 1000]

solve = lambda A, B, C: rtrgsyl(A, B, C, blks=50)
t = vary_matrix_size(solve, vary=n, m=m, n_runs=5)

log_results('test.csv', 'rtrgsyl-scipy', t)
