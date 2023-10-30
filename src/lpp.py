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


class Problem:
    __slots__ = (
        "C",
        "A",
        "b",
        "is_augmented",
        "number_of_targets",
        "number_of_constraints",
    )

    C: np.ndarray
    A: np.ndarray
    b: np.ndarray
    is_augmented: bool
    number_of_targets: Optional[int]
    number_of_constraints: Optional[int]

    def __init__(
        self,
        C: ObjectiveCoefficients,
        A: ConstraintCoefficients,
        b: RightHandSides,
        number_of_targets: Optional[int] = None,
        number_of_constraints: Optional[int] = None,
    ):
        self.C = np.array(C)
        self.A = np.array(A)
        self.b = np.array(b)
        self.is_augmented = False

        if number_of_targets is None:
            number_of_targets = self.C.shape[0]

        if number_of_constraints is None:
            number_of_constraints = self.A.shape[0]

        self.number_of_targets = number_of_targets
        self.number_of_constraints = number_of_constraints

    @staticmethod
    def augment_problem(
        problem: "Problem",
    ):
        """
        Augments the problem with slack variables.
        """

        if problem.is_augmented:
            return

        problem.C = np.concatenate((problem.C, np.zeros(problem.number_of_constraints)))
        problem.A = np.concatenate(
            (problem.A, np.eye(problem.number_of_constraints)), axis=1
        )
        problem.is_augmented = True

    def augment(self):
        """
        Augments the problem with slack variables.
        """
        Problem.augment_problem(self)

    @property
    def solvable(self):
        return np.all(self.b >= 0)
