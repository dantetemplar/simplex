import logging
from dataclasses import dataclass
from typing import Literal

from src.interface import (
    read_equation_type,
    read_number_of_variables,
    read_number_of_constraints,
    read_coefficients_of_objective_function,
    read_coefficients_of_constraints,
    read_coefficients_of_right_hand_side,
)
from src.lpp import check_if_problem_is_solvable, Problem
from src.simplex import solve_using_simplex_method


@dataclass
class UserInput:
    equation_type: Literal["max", "min"]
    number_of_variables: int
    number_of_constraints: int
    C: list[float]
    A: list[list[float]]
    b: list[float]


def read_input() -> UserInput:
    equation_type = read_equation_type()
    number_of_variables = read_number_of_variables()
    number_of_constraints = read_number_of_constraints()

    C = read_coefficients_of_objective_function(number_of_variables)
    A = read_coefficients_of_constraints(number_of_variables, number_of_constraints)
    b = read_coefficients_of_right_hand_side(number_of_constraints)

    return UserInput(
        equation_type=equation_type,
        number_of_variables=number_of_variables,
        number_of_constraints=number_of_constraints,
        C=C,
        A=A,
        b=b,
    )


if __name__ == "__main__":
    from colorama import init, deinit

    init()

    logging.basicConfig(level=logging.INFO)

    user_input = read_input()

    if user_input.equation_type == "min":
        user_input.C = [-c for c in user_input.C]

    problem = Problem(
        C=user_input.C,
        A=user_input.A,
        b=user_input.b,
    )

    problem.augment()

    if not problem.solvable:
        raise RuntimeError(
            "The problem is not solvable by Simplex because of the negativity of right-hand sides in "
            "constraints."
        )

    solution = solve_using_simplex_method(problem.C, problem.A, problem.b)

    deinit()
