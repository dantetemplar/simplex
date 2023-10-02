# Maximization LPP solver using Simplex method

## Installation

Install python version 3.11+ (from [official website](www.python.org) or using `sudo apt-get install python3.11`).

Make sure to add python to `PATH`.

Run following commands:

```
pip3.11 install poetry
poetry install
```

The package `poetry` will automatically install dependencies for project. If it's not you can follow other instructions from [official website](https://python-poetry.org/docs/).

## Usage

To run main program, use the following:

```
poetry run python -m src
```

### Input format
For the LPP in the following form:

Maximize $f = c_1x_1 + c_2x_2 + ... + c_mx_m$ subject to following $n$ constraints:

$$
\begin{cases}
    a_{11}x_1 + a_{12}x_1 + ... + a_{1m}x_m \le b_1 \\
    a_{21}x_1 + a_{22}x_1 + ... + a_{2m}x_m \le b_2 \\
    ... \\
    a_{n1}x_1 + a_{n2}x_1 + ... + a_{nm}x_m \le b_n
\end{cases}
$$

Input:
- $c_1 \ \ c_2 \ \ ... \ \ c_m$ - as coefficients of the objective function,
- $n$ - as number of constraints,
- $a_{i0} \ \ a_{i1} \ \ ... \ \ a_{im}$ - as coefficients of the $i$'th constraint,
- $b_1 \ \ b_2 \ \ ... \ \ b_n$ - as coefficients of the right-hand side of the constraints


### Output Format

The output format is provided in the Solution class, which includes:

Output:

- Solved in $x$ iterations and error $xx.xx$
- Objective function
- Constraints
- Solution:
  - $f$: The value of the objective function,
  - $x$: A dictionary containing the values of the variables,
  - $s$: Represents the value of the slack variable
  - $z$: Represents the value of the artificial variable 

## Example

Given the following `Linear Programming Problem`:

$C = [1, 2]$

$A = [[4, -2], [-1, 3], [-2, -4]]$

$b = [12, 6, 16]$

The solver produces the following `result`:

Solved in 2 iterations and error 0.0

Objective function:

$1.0 * x_0 + 2.0 * x_1$

Constraints:

$4.0 * x_0 + -2.0 * x_1 <= 12.0$

$-1.0 * x_0 + 3.0 * x_1 <= 6.0$

$-2.0 * x_0 + -4.0 * x_1 <= 16.0$

Solution:

$f = 12.0$

$x_0 = 4.8, x_1 = 3.6, s_2 = 40.0, z = -12.0$

This demonstrates how to use the solver and showcases the output format.


## License

Solver is available under the MIT license. See the LICENSE file for more info.
