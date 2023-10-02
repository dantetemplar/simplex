import logging

from simplex import solve_using_simplex_method

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    C = list(map(float, input(
        "Enter the coefficients of the objective function(separate by whitespace):\n"
    ).split()))
    cnt = int(input("Enter the number of constraints:\n"))

    A = [list(map(float, input(
        f"Enter the coefficients of the {i} constraint:\n"
    ).split())) for i in range(cnt)]
    b = list(map(float, input(
        "Enter the coefficients of the right-hand side of the constraints:\n"
    ).split()))
    solution = solve_using_simplex_method(C, A, b)
    print(solution)
