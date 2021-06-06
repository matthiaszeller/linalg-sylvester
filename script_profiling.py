

from benchmark_utils import log_results, vary_matrix_size, benchmark, vary_block_size
from recursive import rtrgsyl
from utils import solve_sylvester_scipy, build_matrices, check_sol, solve_bartels_stewart

m, n = 100, 100
blks = 50

# A, B, C = build_matrices(100, 100)
#
# X, _, _, _ = solve_bartels_stewart(A, B, C, rtrgsyl, blks=blks)
#
# assert check_sol(A, B, C, X)
#
# A, B, C = build_matrices(100, 100)
# X, _, _, _ = solve_bartels_stewart(A, B, C, solve_sylvester_scipy)
#
# assert check_sol(A, B, C, X)

benchmark(
    solve_fun=solve_sylvester_scipy,
    vary_param=('dim', [(100, 100)]),
    bartel_stewart=True
)
