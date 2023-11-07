# Implementation of Interior-Point algorithm

### Content

- [Installation](#installation)
- [Usage](#usage)
- [Example](#example)
- [License](#license)

## Installation

Install python version 3.11+. Make sure to add python to `PATH`.

Using command line, go to the repo folder (`.../simplex/`).

Run following commands:

```bash
pip3.11 install poetry
poetry install
```

The package `poetry` will automatically install dependencies for project. If it's not you can follow other instructions from [official website](https://python-poetry.org/docs/).

## Usage

To run main program, use the following:

```bash
poetry run python -m src
```

### Input format
Solves the LPP in the following form:

#### Objective function:
Maximize or minimize $f = c_1x_1 + c_2x_2 + ... + c_nx_n$

Where:

- $c_1 \ \ c_2 \ \ ... \ \ c_n$ are coefficients associated with decision variables $x_1 \ \ x_2 \ \ ... \ \ x_n$
- $x_1 \ \ x_2 \ \ ... \ \ x_n$ are the decision variables to be determined.

#### Subjected to the following constraints:


$$
\begin{cases}
    a_{11}x_1 + a_{12}x_1 + ... + a_{1n}x_n \lesseqgtr b_1 \\
    a_{21}x_1 + a_{22}x_1 + ... + a_{2n}x_n \lesseqgtr b_2 \\
    ... \\
    a_{m1}x_1 + a_{m2}x_1 + ... + a_{mn}x_n \lesseqgtr b_m
\end{cases}
$$

Where:

- $a_{ij}$ represents coefficients associated with the $j$th decision variable ($x_j$) is a constraint.
- $b_i$ - are constants associated with each constraint.
- $m$ - represents the number of constraints.
- $n$ is the number of decision variables.


### Output Format

The output format is provided in the Solution class, which includes:

Output:
- Optimum of objective function $f(x)$
- $x$: optimal values of the variables

## Example

### Given
Linear Programming Problem as follows:

Maximize: $F (x_1, x_2) = x_1 + x_2$

subjectec to: $x_1 + 2x_2 + 3x_3 = 6$

$$
\begin{cases}
  2x_1 + 4x_2 \le 16 \\
  x_1 + 3x_2 \ge 9 \\
  x_1, x_2 \ge 0
\end{cases}
$$

initial trial solution $(x_1, x_2) = (0.5, 3.5)$



### Result:

Solved in 2 iterations and error 0.0

Objective function optimum: $F(x) = 6.999942864694068$, when $(x_1, x_2) = (0.0000571353, 0.0000571353)$


### Full input and output:
```
Enter the type of equation (`max`, `min`, or just Enter for `max`):
max
Enter the number of variables:
2
Enter the number of constraints:
2
Enter the coefficients of the objective function(separate by whitespace, 2 coefficients):
1 1
Enter 0 constraint: 2 coefficients, sign ("<" or ">"), then right-hand side coefficient, everything separated by whitespace:
2 4 < 16
Enter 1 constraint: 2 coefficients, sign ("<" or ">"), then right-hand side coefficient, everything separated by whitespace:
1 3 > 9
Enter feasible trial solution: values of 2 variables, separated by whitespace:
0.5 3.5
Maximize objective function:               
1.0*x0 + 1.0*x1 + 0.0*x2 + 0.0*x3          
Constraints:                               
2.0*x0 + 4.0*x1 + 1.0*x2 + 0.0*x3 <= 16.0  
-1.0*x0 + -3.0*x1 + 0.0*x2 + 1.0*x3 <= -9.0
Solution:
f = 6.999942864694068
x0 = 0.0000571353, x1 = 0.0000571353
```


## License

Solver is available under the MIT license. See the LICENSE file for more info.
