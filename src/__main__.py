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
from src.lpp import check_if_problem_is_solvable
from src.simplex import solve_using_simplex_method


@dataclass
class Problem:
    equation_type: Literal["max", "min"]
    number_of_variables: int
    number_of_constraints: int
    C: list[float]
    A: list[list[float]]
    b: list[float]


def read_input() -> Problem:
    equation_type = read_equation_type()
    number_of_variables = read_number_of_variables()
    number_of_constraints = read_number_of_constraints()

    C = read_coefficients_of_objective_function(number_of_variables)
    A = read_coefficients_of_constraints(number_of_variables, number_of_constraints)
    b = read_coefficients_of_right_hand_side(number_of_constraints)

    return Problem(
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

    problem = read_input()

    if problem.equation_type == "min":
        problem.C = [-c for c in problem.C]

    check_if_problem_is_solvable(problem.C, problem.A, problem.b)
    solution = solve_using_simplex_method(problem.C, problem.A, problem.b)
    print(solution)

    deinit()
