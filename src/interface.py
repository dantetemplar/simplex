from typing import Literal
from colorama import Fore, Style


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
            print(
                Fore.RED + "Invalid equation type. Please try again. " + Style.RESET_ALL
            )


def read_number_of_variables() -> int:
    while True:
        try:
            number_of_variables = int(input("Enter the number of variables:\n"))
            return number_of_variables
        except ValueError:
            print(
                Fore.RED
                + "Invalid number of variables. Please try again. "
                + Style.RESET_ALL
            )


def read_number_of_constraints() -> int:
    while True:
        try:
            number_of_constraints = int(input("Enter the number of constraints:\n"))
            return number_of_constraints
        except ValueError:
            print(
                Fore.RED
                + "Invalid number of constraints. Please try again. "
                + Style.RESET_ALL
            )


def read_coefficients_of_objective_function(number_of_variables: int) -> list[float]:
    while True:
        try:
            coefficients = list(
                map(
                    float,
                    input(
                        f"Enter the coefficients of the objective function(separate by whitespace, "
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


def read_constraints(
    number_of_variables: int, number_of_constraints: int
) -> tuple[list[list[float]], list[float]]:
    constraints: list[list[float]] = []
    rhs_coefficients: list[float] = []

    def read_constraint(i: int) -> tuple[list[float], float]:
        while True:
            try:
                raw = input(
                    f"Enter {i} constraint: {number_of_variables} coefficients, sign (\"<\" or \">\"), "
                    f"then right-hand side coefficient:\n"
                ).split(),
                if len(raw) < 3:
                    raise ValueError
                if raw[-2] not in '<>':
                    raise ValueError
                constraint = [c if '<' in raw[-2] else -c for c in map(float, raw[:-2])]
                if len(constraint) != number_of_variables:
                    raise ValueError
                rhs = float(raw[-1]) if '<' in raw[-2] else -float(raw[-1])
                return constraint, rhs
            except ValueError:
                print(
                    Fore.RED
                    + f"Invalid input."
                    + Style.RESET_ALL
                )

    for i in range(number_of_constraints):
        constraint_i, rhs_i = read_constraint(i)
        constraints.append(constraint_i)
        rhs_coefficients.append(rhs_i)

    return constraints, rhs_coefficients

'''
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
'''
