from collections.abc import Collection
from typing import Optional

import numpy as np
import logging
from dataclasses import dataclass


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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

    f: float
    """The value of the objective function"""
    x: Optional[np.ndarray] = None
    """A vector of values of the variables"""
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


class Tableau:
    _tableau: np.ndarray
    """The tableau with the coefficients of the problem"""
    _variable_cols: list[int]  # without the solution column
    _variable_rows: list[int]  # without the z row

    @property
    def m(self):
        return self._tableau

    @property
    def z(self) -> np.ndarray:
        """
        The row of the objective function. Without the solution column.
        """
        return self._tableau[-1, :-1]

    @property
    def solution(self) -> np.ndarray:
        """
        The column of the solution.
        """
        return self._tableau[:, -1]

    @property
    def f(self) -> float:
        """
        The value of the objective function.
        """
        return self.solution[-1].item()

    def is_optimal(self) -> bool:
        """
        Checks if the tableau is optimal.

        :return: True if the tableau is optimal, False otherwise
        """
        return np.all(self.z <= 0)

    def __init__(self, tableau: np.ndarray):
        self._tableau = tableau
        self._variable_cols = list(range(tableau.shape[1] - 1))
        self._variable_rows = list(range(tableau.shape[0] - 1))

    @classmethod
    def base_case_to_tableau(
        cls, C: ObjectiveCoefficients, A: ConstraintCoefficients, b: RightHandSides
    ) -> "Tableau":
        """
        Converts the [#input_data]_ base case of linear programming problem to the tableau form.

        Tableau form::

            +-------------------------+
            |   Coeff-s  |  Solution  |
            +=========================+
            | A  A  1  0 | b          |
            | A  A  0  1 | b          |
            | C  C  0  0 | 0          |
            +-------------------------+

        As an example, the following problem:

        >>> C = [1, 2]
        >>> A = [[1, 1], [1, -1]]
        >>> b = [2, 1]

        will be converted to the following tableau (with an added slack variable for each constraint):

        >>> Tableau.base_case_to_tableau(C, A, b).m
        array([[ 1.,  1.,  1.,  0.,  2.],
               [ 1., -1.,  0.,  1.,  1.],
               [ 1.,  2.,  0.,  0.,  0.]])

        .. [#input_data] Base case of linear programming problem is a problem in the following form:

            * All constraints are inequalities of the form :math:`a_1 x_1 + a_2 x_2 + ... + a_n x_n <= b_i`.
            * Maximization problem of objective function :math:`c_1 x_1 + c_2 x_2 + ... + c_n x_n`.
            * All variables are non-negative.

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

        logger.info(f"{cnt_of_equations=}")
        logger.info(f"{cnt_of_variables=}")

        tableau: np.ndarray = np.zeros(
            (cnt_of_equations + 1, cnt_of_variables + cnt_of_slack + 1)
        )
        tableau[-1, :cnt_of_variables] = C
        tableau[:-1, :cnt_of_variables] = A
        tableau[:-1, cnt_of_variables:-1] = np.eye(cnt_of_equations)
        tableau[:-1, -1] = np.array(b)
        return Tableau(tableau)

    def find_pivot_column(self) -> int:
        """
        Finds the pivot column in the tableau.

        :return: The index of the pivot column
        """

        _temp = self.z.copy()
        _temp[_temp <= 0] = np.inf
        return np.argmin(_temp)

    def find_pivot_row(self, pivot_column: int) -> int:
        """
        Finds the pivot row in the tableau.

        :param pivot_column: The index of the pivot column
        :return: The index of the pivot row
        """
        divisors = self._tableau[:-1, pivot_column]
        restrictions = np.divide(
            self._tableau[:-1, -1],
            self._tableau[:-1, pivot_column],
            out=np.full(len(divisors), np.inf),
            where=divisors > 0,
        )
        return np.argmin(restrictions)

    def swap_variable(self, pivot_row, pivot_column):
        """
        Swaps the variable in the pivot row and pivot column.

        :param pivot_row: Pivot row with loose variable
        :param pivot_column: Pivot column with tight variable
        """

        pivot_value = self._tableau[pivot_row, pivot_column]
        self._tableau[pivot_row, :] /= pivot_value

    def __repr__(self):
        _ = np.array2string(
            self._tableau, precision=2, floatmode="fixed", prefix="Tableau(", suffix=")"
        )

        return "Tableau(" + _ + ")"


def solve_using_simplex_method(
    C: ObjectiveCoefficients,
    A: ConstraintCoefficients,
    b: RightHandSides,
    max_iterations: int = 1000,
    ftol: float = 1e-8,
) -> Solution:
    """
    Solves the linear programming problem using the simplex method.

    :param ftol: Tolerance of the objective function
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

    tableau = Tableau.base_case_to_tableau(C, A, b)
    logger.info(f"Initial tableau:\n{tableau}")

    solved_tableau, delta_f, iteration = _simplex(
        tableau, max_iterations=max_iterations, ftol=ftol
    )
    logger.info(f"Solved in {iteration} iterations and error {delta_f}")

    f = -solved_tableau.f

    if abs(delta_f) > 0:
        logger.info("{f=}")
        return Solution(f=f, C=C)
    else:
        x = get_solution(solved_tableau.m)
        logger.info(f"{x=}, {f=}")
        return Solution(x=x, f=f, C=C)


def _simplex(
    tableau: Tableau, max_iterations: int, ftol: float
) -> tuple[Tableau, float, int]:
    """
    Solves the linear programming problem using the simplex method.

    :param tableau: problem in the tableau form (already with artificial and slack variables)
    :param max_iterations: maximum number of iterations, after which the algorithm will raise a runtime exception
    :return: solved tableau
    """
    iteration = 0
    f: float
    prev_f: float
    delta_f: float
    f = prev_f = tableau.f

    while not tableau.is_optimal():
        iteration += 1
        logger.info(f"Iteration {iteration}")

        if iteration > max_iterations:
            raise RuntimeError("Maximum number of iterations exceeded")

        pivot_column = tableau.find_pivot_column()
        pivot_row = tableau.find_pivot_row(pivot_column)
        # swap around pivot
        tableau.swap_variable(pivot_row=pivot_row, pivot_column=pivot_column)
        pivot_row_values = tableau.m[pivot_row, :]
        # perform row operations to make pivot column 0 except for pivot row (pivot row is already 1)
        for eq_i in range(tableau.m.shape[0]):
            if eq_i != pivot_row:
                delta_row = pivot_row_values * tableau.m[eq_i, pivot_column]
                tableau.m[eq_i, :] -= delta_row

        f = tableau.f
        delta_f = f - prev_f
        if abs(delta_f) < ftol:
            logger.info("Optimal solution found by tolerance")
            break
        prev_f = f
    delta_f = f - prev_f

    return tableau, delta_f, iteration


def get_cnts_of_variables(tableau: np.ndarray) -> tuple[int, int]:
    """
    Returns the number of variables, slack variables and artificial variables in the tableau.

    :param tableau: The tableau
    :return: The number of variables, slack variables and artificial variables in the tableau
    """
    cnt_of_equations, cnt_of_variables = tableau.shape
    cnt_of_slack = cnt_of_equations - 1
    cnt_of_target = cnt_of_variables - cnt_of_slack - 1
    return cnt_of_slack, cnt_of_target


def get_solution(tableau: np.ndarray) -> np.ndarray:
    """
    Extracts the solution from the tableau.

    :param tableau: The solved tableau
    :return: The solution of the linear programming problem (values of the variables)
    """
    cnt_of_slack, cnt_of_target = get_cnts_of_variables(tableau)

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


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    C = [3, 9]  # z = 3x1 + 9x2
    A = [[1, 4], [1, 2]]  # x1 + 4x2 <= , x1 + 2x2 <= 4
    b = [8, 4]

    solution = solve_using_simplex_method(C, A, b, ftol=0.1)
    print(solution)
