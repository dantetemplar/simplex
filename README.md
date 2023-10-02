# Maximization LPP solver using Simplex method

## Installation

Install python version 3.11+ (from [official website](www.python.org) or using `sudo apt-get install python3.11`).

Run following commands:

```
pip3.11 install poetry
poetry install
```

The package `poetry` will automatically install dependencies for project.

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

## License

Solver is available under the MIT license. See the LICENSE file for more info.
