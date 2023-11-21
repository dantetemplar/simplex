import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.interface import (
    read_equation_type,
    read_number_of_variables,
    read_number_of_constraints,
    read_coefficients_of_objective_function,
    read_constraints,
    read_interior_point_trial_solution,
    read_transportation_problem,
    read_transportation_method,
    # read_coefficients_of_constraints,
    # read_coefficients_of_right_hand_side,
)


from transportation import TransportationProblem


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
    A, b = read_constraints(number_of_variables, number_of_constraints)

    return UserInput(
        equation_type=equation_type,
        number_of_variables=number_of_variables,
        number_of_constraints=number_of_constraints,
        C=C,
        A=A,
        b=b,
    )


def lpp_pipeline():
    from src.lpp import Problem
    from interior_point import solve_using_interior_point_method, augment_trial_solution
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
        number_of_targets=user_input.number_of_variables,
        number_of_constraints=user_input.number_of_constraints,
    )
    problem.augment()

    while True:
        trial_solution = np.array(
            read_interior_point_trial_solution(user_input.number_of_variables)
        )
        if problem.is_feasible(trial_solution):
            break
    trial_solution = augment_trial_solution(problem, trial_solution)

    """
    if not problem.solvable:
        raise RuntimeError(
            "The problem is not solvable by Simplex because of the negativity of right-hand sides in "
            "constraints."
        )

    solution = solve_using_simplex_method(problem)
    """
    solution = solve_using_interior_point_method(
        problem, first_trial_solution=trial_solution
    )
    solution.type = "Maximize" if user_input.equation_type == "max" else "Minimize"
    print(solution)
    deinit()


def transportation_pipeline():
    from colorama import init, deinit
    from src.transportation import (
        TransportationProblem,
        solve_using_north_west_corner,
        solve_using_vogel,
        solve_using_russells,
    )

    init()
    supply_amounts, demand_amounts, coefficients = read_transportation_problem()

    problem = TransportationProblem(
        supplies=supply_amounts,
        demands=demand_amounts,
        costs=coefficients,
    )

    method = read_transportation_method()

    solution = None
    if method == "northwest":
        solution = solve_using_north_west_corner(problem)
    elif method == "vogel":
        solution = solve_using_vogel(problem)
    elif method == "russel":
        solution = solve_using_russells(problem)

    print(solution)


if __name__ == "__main__":
    # lpp_pipeline()
    transportation_pipeline()
