from dataclasses import dataclass
from typing import Optional, Collection

import numpy as np

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
    x: list[float]
    """The values of the variables"""
    C: Optional[ObjectiveCoefficients] = None
    """The coefficients of the objective function"""
    A: Optional[ConstraintCoefficients] = None
    """The coefficients of the constraints"""
    b: Optional[RightHandSides] = None
    """The right-hand sides of the constraints"""

    def __str__(self):
        result_string = []

        if self.C is not None:
            result_string.append("Maximize Objective function:")
            objective = [f"{c}*x{i}" for i, c in enumerate(self.C)]
            objective = " + ".join(objective)
            result_string.append(objective)

        if self.A is not None:
            result_string.append("Constraints:")
            for i, (row, right_hand_side) in enumerate(zip(self.A, self.b)):
                constraint = [f"{c}*x{i}" for i, c in enumerate(row)]
                constraint = " + ".join(constraint)
                result_string.append(f"{constraint} <= {right_hand_side}")

        result_string.append("Solution:")
        result_string.append(f"f = {self.f}")
        result_string.append(
            ", ".join([f"x{i} = {x_term}" for i, x_term in enumerate(self.x)])
        )

        return "\n".join(result_string)


def check_if_problem_is_solvable(
    _C: ObjectiveCoefficients, _A: ConstraintCoefficients, b: RightHandSides
):
    """
    Checks if the linear programming problem is solvable.

    :param _C: Vector of coefficients of the objective function
    :param _A: Matrix of coefficients of the constraints
    :param b: Vector of right-hand sides of the constraints
    """

    # check if right-hand sides are non-negative
    if np.any(np.array(b) < 0):
        raise RuntimeError(
            "The problem is not solvable by Simplex because of the negativity of right-hand sides in "
            "constraints."
        )
