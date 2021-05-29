

import unittest

from recursive import rtrgsyl
from utils import solve_bartels_stewart, build_matrices, check_sol


class TestRtrgsyl(unittest.TestCase):

    def test_without_recursion(self):
        A, B, C = build_matrices(8, 2)
        # blks > matrix sizes => no recursion
        fun = lambda A, B, C: rtrgsyl(A, B, C, 10)
        X = solve_bartels_stewart(A, B, C, fun)[0]
        self.assertTrue(check_sol(A, B, C, X))

        A, B, C = build_matrices(2, 8)
        X = solve_bartels_stewart(A, B, C, fun)[0]
        self.assertTrue(check_sol(A, B, C, X))

    def test_case_1(self):
        # n <= n/2
        fun = lambda A, B, C: rtrgsyl(A, B, C, 10)

        A, B, C = build_matrices(100, 20)
        X = solve_bartels_stewart(A, B, C, fun)[0]
        self.assertTrue(check_sol(A, B, C, X))

    def test_case_2(self):
        # m <= n/2
        fun = lambda A, B, C: rtrgsyl(A, B, C, 10)

        A, B, C = build_matrices(20, 100)
        X = solve_bartels_stewart(A, B, C, fun)[0]
        self.assertTrue(check_sol(A, B, C, X))

    def test_case_3(self):
        fun = lambda A, B, C: rtrgsyl(A, B, C, 10)

        A, B, C = build_matrices(100, 100)
        X = solve_bartels_stewart(A, B, C, fun)[0]
        self.assertTrue(check_sol(A, B, C, X))


if __name__ == '__main__':
    unittest.main()
