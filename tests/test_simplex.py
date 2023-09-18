import src.simplex as simplex
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)


def test_tableau():
    C = [1, 2]
    A = [[1, 1], [1, -1]]
    b = [2, 1]
    tableau = simplex.to_tableau(C, A, b)
    expected = np.array([[1, 1, 1, 0, 2], [1, -1, 0, 1, 1], [-1, -2, 0, 0, 0]])

    assert np.array_equal(tableau, expected) is True

    C = [1, 1, 0, 0]
    A = [[1, 1, 1, 0], [1, -1, 0, 1]]
    b = [2, 1]

    tableau = simplex.to_tableau(C, A, b)
    expected = np.array(
        [[1, 1, 1, 0, 1, 0, 2], [1, -1, 0, 1, 0, 1, 1], [-1, -1, 0, 0, 0, 0, 0]]
    )

    assert np.array_equal(tableau, expected) is True


def test_simplex():
    C = [1, 1, 0]
    A = [[-1, 1, 1], [1, 0, 0], [0, 1, 0]]
    b = [2, 4, 4]

    solution = simplex.solve_using_simplex_method(C, A, b)
    print(solution)
    assert solution.x == [2, 2, 0]
