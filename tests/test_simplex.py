from typing import Collection

import pytest

import src.simplex as simplex
import numpy as np
import logging

from tests.conftest import dataset

logging.basicConfig(level=logging.DEBUG)


def test_simplex():
    C = [1, 1, 0]
    A = [[-1, 1, 1], [1, 0, 0], [0, 1, 0]]
    b = [2, 4, 4]

    solution = simplex.solve_using_simplex_method(C, A, b)
    assert solution.x == {"s0": 2.0, "x0": 4.0, "x1": 4.0, "z": -8.0}


eps = 1e-6


@pytest.mark.parametrize(
    "C, A, b, expected_f",
    [(testcase.C, testcase.A, testcase.b, testcase.expected_f) for testcase in dataset],
)
def test_simplex_dataset(
    C: Collection[float],
    A: Collection[Collection[float]],
    b: Collection[float],
    expected_f: float,
    # expected_x: Collection[float],
):
    solution = simplex.solve_using_simplex_method(C, A, b)
    print(f"{solution.f=} {expected_f=}")
    assert abs(solution.f - expected_f) < eps
