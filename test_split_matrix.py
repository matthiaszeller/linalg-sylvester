

import unittest

import numpy as np

from recursive import split_matrix


class TestSplitMatrix(unittest.TestCase):

    def test_triangular(self):
        a = np.array([
            [1, 2, 3],
            [0, 4, 5],
            [0, 0, 6]
        ])
        self.assertEqual(split_matrix(a), 0)
        a = np.array([
            [0, 1, 2, 3],
            [0, 4, 5, 6],
            [0, 0, 7, 8],
            [0, 0, 0, 9]
        ])
        self.assertEqual(split_matrix(a), 1)

        for k in range(3, 10):
            n = 2*k + 1
            a = np.arange(n*n).reshape(n, n)
            a = np.triu(a)
            m = k - 1
            self.assertEqual(split_matrix(a), m, msg=f'k={k}, n={n}, m={m}')

    def test_quasi_triangular(self):
        a = np.array([
            [1, 2, 3],
            [0, 4, 5],
            [0, 6, 7]
        ])
        self.assertEqual(split_matrix(a), 0)
        a = np.array([
            [1, 2, 3, 4],
            [0, 5, 6, 7],
            [0, 8, 9, 10],
            [0, 0, 0, 11]
        ])
        self.assertEqual(split_matrix(a), 2)


if __name__ == '__main__':
    unittest.main()
