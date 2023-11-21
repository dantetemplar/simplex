from typing import Literal, Optional
from colorama import Fore, Style

ERROR_COMMENT = Fore.RED + "Invalid input. Please try again. " + Style.RESET_ALL


def read_int(
    comment: str,
    /,
    gt: Optional[int] = None,
    error_comment: str = ERROR_COMMENT,
    default: Optional[int] = None,
) -> int:
    while True:
        try:
            r = input(comment)
            if default is not None and not r:
                return default
            r = int(r)
            if gt is not None and r <= gt:
                raise ValueError
            return r
        except ValueError:
            print(error_comment)


def read_floats_list(
    comment: str,
    /,
    length: Optional[int] = None,
    gt: Optional[int] = None,
    error_comment: str = ERROR_COMMENT,
) -> list[float]:
    while True:
        try:
            floats = list(map(float, input(comment).split()))
            if length is not None and len(floats) != length:
                raise ValueError

            if gt is not None and any(x <= gt for x in floats):
                raise ValueError

            return floats
        except ValueError:
            print(error_comment)


def read_equation_type() -> Literal["max", "min"]:
    while True:
        equation_type = input(
            "Enter the type of equation (`max`, `min`, or just Enter for `max`):\n"
        )
        equation_type = equation_type.lower() if equation_type else "max"

        if equation_type == "max":
            return "max"
        elif equation_type == "min":
            return "min"
        else:
            print(ERROR_COMMENT)


def read_number_of_variables() -> int:
    return read_int("Enter the number of variables:\n", gt=0)


def read_number_of_constraints() -> int:
    return read_int("Enter the number of constraints:\n", gt=0)


def read_coefficients_of_objective_function(number_of_variables: int) -> list[float]:
    return read_floats_list(
        f"Enter the coefficients of the objective function (separate by whitespace, {number_of_variables} "
        f"coefficients):\n",
        length=number_of_variables,
    )


def read_constraints(
    number_of_variables: int, number_of_constraints: int
) -> tuple[list[list[float]], list[float]]:
    constraints: list[list[float]] = []
    rhs_coefficients: list[float] = []

    def read_constraint(i: int) -> tuple[list[float], float]:
        while True:
            try:
                raw = input(
                    f'Enter {i} constraint: {number_of_variables} coefficients, sign ("<" or ">"), '
                    f"then right-hand side coefficient, everything separated by whitespace:\n"
                ).split()
                if len(raw) < 3:
                    print("1", len(raw), raw)
                    raise ValueError
                if raw[-2] not in "<>":
                    print("2")
                    raise ValueError
                constraint = [c if "<" in raw[-2] else -c for c in map(float, raw[:-2])]
                if len(constraint) != number_of_variables:
                    print("3")
                    raise ValueError
                rhs = float(raw[-1]) if "<" in raw[-2] else -float(raw[-1])
                return constraint, rhs
            except ValueError:
                print(Fore.RED + "Invalid input." + Style.RESET_ALL)

    for i in range(number_of_constraints):
        constraint_i, rhs_i = read_constraint(i)
        constraints.append(constraint_i)
        rhs_coefficients.append(rhs_i)

    return constraints, rhs_coefficients


def read_interior_point_trial_solution(number_of_variables) -> list[float]:
    return read_floats_list(
        f"Enter feasible trial solution: values of {number_of_variables} variables, separated by whitespace:\n",
        length=number_of_variables,
    )


def read_units(number_of_variables: int, units_input_prompt: str) -> list[float]:
    return read_floats_list(
        f"{units_input_prompt} (separate by whitespace, {number_of_variables} float coefficients):\n",
        length=number_of_variables,
        gt=0,
    )


def read_transportation_problem() -> tuple[list[float], list[float], list[list[float]]]:
    supply_number = read_int("Enter the number of supplies (default: 3):\n", default=3)
    demand_number = read_int("Enter the number of demands (default: 4):\n", default=4)
    supply_amounts = read_units(supply_number, "Enter Supplies")
    demand_amounts = read_units(demand_number, "Enter Demands")

    coefficients = [
        read_units(demand_number, f"Enter distribution unit cost for supply {supply_i}")
        for supply_i in range(supply_number)
    ]

    return supply_amounts, demand_amounts, coefficients


def read_transportation_method() -> Literal["northwest", "vogel", "russel"]:
    while True:
        method = input(
            "Enter the method of transportation problem (`northwest`, `vogel`, `russel`, or just Enter for "
            "`northwest`):\n"
        )
        method = method.lower() if method else "northwest"

        if method == "northwest":
            return "northwest"
        elif method == "vogel":
            return "vogel"
        elif method == "russel":
            return "russel"
        else:
            print(ERROR_COMMENT)


"""
def read_coefficients_of_constraints(
    number_of_variables: int, number_of_constraints: int
) -> list[list[float]]:
    constraints: list[list[float]] = []

    def read_constraint(i: int) -> list[float]:
        while True:
            try:
                coefficients = list(
                    map(
                        float,
                        input(
                            f"Enter the coefficients of the {i} constraint (separate by whitespace, "
                            f"{number_of_variables} coefficients):\n"
                        ).split(),
                    )
                )
                if len(coefficients) != number_of_variables:
                    raise ValueError
                return coefficients
            except ValueError:
                print(
                    Fore.RED
                    + f"Invalid number of coefficients. Please enter exactly {number_of_variables} coefficients. "
                    + Style.RESET_ALL
                )

    for i in range(number_of_constraints):
        constraints.append(read_constraint(i))

    return constraints


def read_coefficients_of_right_hand_side(number_of_constraints: int) -> list[float]:
    while True:
        try:
            coefficients = list(
                map(
                    float,
                    input(
                        "Enter the coefficients of the right-hand side of the constraints(separate by whitespace, "
                        "{number_of_constraints} coefficients):\n"
                    ).split(),
                )
            )
            if len(coefficients) != number_of_constraints:
                raise ValueError
            return coefficients
        except ValueError:
            print(
                Fore.RED
                + f"Invalid number of coefficients. Please enter exactly {number_of_constraints} coefficients. "
                + Style.RESET_ALL
            )
"""
