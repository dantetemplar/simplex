from collections.abc import Collection
from typing import Optional

import numpy as np
import logging
from dataclasses import dataclass

logging.getLogger(__name__).setLevel(logging.INFO)


def _logger():
    return logging.getLogger(__name__)


ObjectiveCoefficients = Collection[float]
"""A vector of coefficients of the objective function"""
ConstraintCoefficients = Collection[Collection[float]]
"""A matrix of coefficients of the constraints"""
RightHandSides = Collection[float]
"""A vector of right-hand sides of the constraints"""


@dataclass
class Solution:
    """
    A solution of the linear programming problem.
    """

    x: np.ndarray
    """A vector of values of the variables"""
    f: float
    """The value of the objective function"""

    C: Optional[ObjectiveCoefficients] = None

    def __str__(self):
        if self.C is None:
            return f"x = {self.x}\nf = {self.f}"
        else:
            objective = [f"{c}*x{i}" for i, c in enumerate(self.C)]
            objective = " + ".join(objective)
            return (
                f"Objective function: {objective}\n" f"x = {self.x}\n" f"f = {self.f}"
            )


def solve_using_simplex_method(
    C: ObjectiveCoefficients,
    A: ConstraintCoefficients,
    b: RightHandSides,
    max_iterations: int = 1000,
) -> Solution:
    """
    Solves the linear programming problem using the simplex method.

    :param C: A vector of coefficients of the objective function
    :param A: A matrix of coefficients of the constraints
    :param b: A vector of right-hand sides of the constraints
    :param max_iterations: Maximum number of iterations
    :return: solution of the linear programming problem and the value of the objective function

    Example:
    >>> C = [1, 1, 0]
    >>> A = [[-1, 1, 1], [1, 0, 0], [0, 1, 0]]
    >>> b = [2, 4, 4]
    >>> solution = solve_using_simplex_method(C, A, b)
    >>> solution.x
    array([4., 4., 2.])
    >>> solution.f
    8.0

    Example:
    >>> C = [1.2, 1.7]
    >>> A = [[1, 0], [0, 1], [1, 1]]
    >>> b = [3000, 4000, 5000]
    >>> solution = solve_using_simplex_method(C, A, b)
    >>> solution.x
    array([1000., 4000.])
    >>> solution.f
    8000.0
    """

    tableau = to_tableau(C, A, b)
    _logger().info(f"Initial tableau:\n{tableau=}")

    iteration = 0
    prev_f = f = 0.0

    while not is_optimal(tableau):
        iteration += 1
        _logger().info(f"Iteration {iteration}")
        if iteration > max_iterations:
            raise RuntimeError("Maximum number of iterations exceeded")

        pivot_column = find_pivot_column(tableau)
        pivot_row = find_pivot_row(tableau, pivot_column)

        _logger().info(f"{pivot_column=}, {pivot_row=}")
        _logger().info(f"pivot_value = {tableau[pivot_row, pivot_column]}")
        tableau[pivot_row, :] /= tableau[pivot_row, pivot_column]
        for eq_i in range(tableau.shape[0]):
            if eq_i != pivot_row:
                _logger().info(f"{eq_i=}")
                delta_row = tableau[pivot_row, :] * tableau[eq_i, pivot_column]
                _logger().info(f"{delta_row=}")
                tableau[eq_i, :] -= delta_row
                _logger().info(f"new_row={tableau[eq_i, :]}")

        _logger().info(f"{tableau=}")
        f = tableau[-1, -1]
        _logger().info(f"{f=}")
        _logger().info(f"delta_f = {f - prev_f}")
        prev_f = f
    x = get_solution(tableau)
    _logger().info(f"{x=}, {f=}")

    return Solution(x, -f, C)


def get_solution(tableau: np.ndarray) -> np.ndarray:
    """
    Extracts the solution from the tableau.

    :param tableau: The tableau
    :return: The solution
    """
    cnt_of_equations, cnt_of_variables = tableau.shape
    cnt_of_slack = cnt_of_equations - 1
    cnt_of_target = cnt_of_variables - cnt_of_slack - 1
    x = np.zeros(cnt_of_target)

    for i in range(cnt_of_target):
        indices = np.where(tableau[:, i] == 1)[0]
        solutions_for_variable = tableau[indices, -1]
        if len(solutions_for_variable) == 1:
            x[i] = solutions_for_variable[0]
        elif len(solutions_for_variable) == 0:
            x[i] = 0
        else:
            raise RuntimeError("The tableau is not optimal")
    return x


def to_tableau(
    C: ObjectiveCoefficients, A: ConstraintCoefficients, b: RightHandSides
) -> np.ndarray:
    """
    Converts the linear programming problem to the tableau form.

    As an example, the following problem:

    >>> C = [1, 2]
    >>> A = [[1, 1], [1, -1]]
    >>> b = [2, 1]

    will be converted to the following tableau (with an added slack variable for each constraint):

    >>> to_tableau(C, A, b)
    array([[ 1.,  1.,  1.,  0.,  2.],
           [ 1., -1.,  0.,  1.,  1.],
           [ 1.,  2.,  0.,  0.,  0.]])
    """

    C = np.array(C)
    A = np.array(A)
    b = np.array(b)

    cnt_of_equations, cnt_of_variables = A.shape
    cnt_of_slack = cnt_of_equations

    if len(C) != cnt_of_variables:
        raise ValueError(
            f"Number of coefficients of the objective function ({len(C)}) "
            f"does not match the number of variables ({cnt_of_variables})"
        )

    if len(b) != cnt_of_equations:
        raise ValueError(
            f"Number of right-hand sides ({len(b)}) "
            f"does not match the number of equations ({cnt_of_equations})"
        )

    _logger().info(f"{cnt_of_equations=}")
    _logger().info(f"{cnt_of_variables=}")

    tableau: np.ndarray = np.zeros(
        (cnt_of_equations + 1, cnt_of_variables + cnt_of_slack + 1)
    )
    tableau[-1, :cnt_of_variables] = C
    tableau[:-1, :cnt_of_variables] = A
    tableau[:-1, cnt_of_variables:-1] = np.eye(cnt_of_equations)
    tableau[:-1, -1] = np.array(b)
    return tableau


def is_optimal(tableau: np.ndarray) -> bool:
    """
    Checks if the tableau is optimal.

    :param tableau: The tableau
    :return: True if the tableau is optimal, False otherwise
    """
    return np.all(tableau[-1, :-1] <= 0)


def find_pivot_column(tableau: np.ndarray) -> int:
    """
    Finds the pivot column in the tableau.

    :param tableau: The tableau
    :return: The index of the pivot column
    """
    # z = tableau[-1]
    # print(z)
    # column = next(i for i, x in enumerate(z[:-1]) if x > 0)

    _temp = tableau[-1, :-1].copy()
    _temp[_temp <= 0] = np.inf
    return np.argmin(_temp)


def find_pivot_row(tableau: np.ndarray, pivot_column: int) -> int:
    """
    Finds the pivot row in the tableau.

    :param tableau: The tableau
    :param pivot_column: The index of the pivot column
    :return: The index of the pivot row
    """
    divisors = tableau[:-1, pivot_column]
    restrictions = np.divide(
        tableau[:-1, -1],
        tableau[:-1, pivot_column],
        out=np.full(len(divisors), np.inf),
        where=divisors > 0,
    )
    return np.argmin(restrictions)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    C = [1, 1, 0]
    A = [[-1, 1, 1], [1, 0, 0], [0, 1, 0]]
    b = [2, 4, 4]

    solution = solve_using_simplex_method(C, A, b)
    print(solution)
    # [[-1  1  1  0  0  2]
    #  [ 1  0  0  1  0  4]
    #  [ 0  1  0  0  1  4]
    #  [ 1  1  0  0  0  0]]
    # [[ 0.  1.  1.  1.  0.  6.]
    #  [ 1.  0.  0.  1.  0.  4.]
    #  [ 0.  1.  0.  0.  1.  4.]
    #  [ 0.  1.  0. -1.  0. -4.]]
    # [[ 0.  0.  1.  1. -1.  2.]
    #  [ 1.  0.  0.  1.  0.  4.]
    #  [ 0.  1.  0.  0.  1.  4.]
    #  [ 0.  0.  0. -1. -1. -8.]]
