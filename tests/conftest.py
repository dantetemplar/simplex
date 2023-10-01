from dataclasses import dataclass
from typing import Collection


@dataclass
class TestCase:
    C: Collection[float]
    A: Collection[Collection[float]]
    b: Collection[float]
    expected_f: float


dataset: list[TestCase] = []


def solve_using_scipy(C, A, b) -> float:
    from scipy.optimize import linprog

    from scipy.optimize import OptimizeResult

    res: OptimizeResult = linprog(
        c=[-c for c in C],
        A_ub=A,
        b_ub=b,
        bounds=[(0, None) for _ in range(len(C))],
        method="highs-ds",
        options={"tol": 1e-6, "maxiter": 1000},
    )

    if res.success:
        return -res.fun


def validate_with_scipy(C, A, b, F):
    # check existence of module
    try:
        from scipy.optimize import linprog, OptimizeResult
    except ImportError:
        return

    res: OptimizeResult = linprog(
        c=[-c for c in C],
        A_ub=A,
        b_ub=b,
        bounds=[(0, None) for _ in range(len(C))],
        method="highs-ds",
        options={"tol": 1e-6, "maxiter": 1000},
    )

    if res.success:
        result = -res.fun
        assert abs(result - F) < 1e-6


# 0 Answer: F=12 X=(12/5, 14/5, 8) max
C = [1, 2]
A = [[4, -2], [-1, 3], [-2, -4]]
b = [12, 6, 16]
F = 12
validate_with_scipy(C, A, b, F)

dataset.append(TestCase(C, A, b, expected_f=F))
# 1 Answer: F=27/2 X=(0, 9/4, 0) max
C = [3, 6, 2]
A = [[8, 4, 10], [3, 4, 7], [2, 1, 3]]
b = [12, 9, 5]
F = 27 / 2
validate_with_scipy(C, A, b, F)

dataset.append(TestCase(C, A, b, expected_f=F))
# 2 Answer: F=391/15 X=(23/15, 0, 0) max
C = [17, 2, 10]
A = [[13, 8, 11], [15, 3, 14]]
b = [20, 23]
F = 391 / 15
validate_with_scipy(C, A, b, F)

dataset.append(TestCase(C, A, b, expected_f=F))

# 3 Answer: F=3 X=(3/2, 0) max
C = [2, 1]
A = [[4, 2], [3, 3]]
b = [6, 9]
F = 3
validate_with_scipy(C, A, b, F)

dataset.append(TestCase(C, A, b, expected_f=F))

# 4 Answer: F=46/3 X=(5/3, 17/3, 0) max
C = [-1, 3, -3]
A = [[3, -1, -2], [-2, -4, 4], [1, 0, -1], [-2, 2, 8], [3, 0, 0]]
b = [7, 3, 4, 8, 5]
F = 46 / 3
validate_with_scipy(C, A, b, F)

dataset.append(TestCase(C, A, b, expected_f=F))

# 5 Answer: F=33 X=(3, 12) max
C = [3, 2]
A = [[2, 1], [2, 3], [3, 1]]
b = [18, 42, 24]
F = 33
validate_with_scipy(C, A, b, F)

dataset.append(TestCase(C, A, b, expected_f=F))

# 6 Answer: F=8326 X=(691/5, 118/5) max
C = [50, 60]
A = [[2, 1], [3, 4], [4, 7]]
b = [300, 509, 812]
F = 8326
validate_with_scipy(C, A, b, F)

dataset.append(TestCase(C, A, b, expected_f=F))

# 7 Answer: F=110 X=(10, 10) max
C = [6, 5]
A = [[-3, 5], [-2, 5], [1, 0], [3, -8]]
b = [25, 30, 10, 6]
F = 110
validate_with_scipy(C, A, b, F)

dataset.append(TestCase(C, A, b, expected_f=F))

# 8 Answer: F=11 X=(3, 1) max
C = [3, 2]
A = [[1, 1], [1, -1]]
b = [4, 2]
F = 11
validate_with_scipy(C, A, b, F)

dataset.append(TestCase(C, A, b, expected_f=F))

# 9 Answer: F=27 X=(0, 9, 3) max
C = [1, 2, 3]
A = [[1, 1, 1], [2, 1, 3]]
b = [12, 18]
F = 27
validate_with_scipy(C, A, b, F)

dataset.append(TestCase(C, A, b, expected_f=F))

# 10 MegaHardRandom Answer: F=161.16728624535315 X=(0, 130/269, 2512/269, 0, 0) max
C = [11, 5, 17, -7, -5]
A = [[18, -7, 10, 8, 1], [5, 22, 7, 18, -4], [20, -3, 1, 13, 18]]
b = [90, 76, 92]
F = 161.16728624535315
validate_with_scipy(C, A, b, F)

dataset.append(TestCase(C, A, b, expected_f=F))

# 11 Answer: F=4 X=(0, 5/3) max
C = [1, 4]
A = [[2, 1], [3, 5], [1, 3]]
b = [3, 5, 9]
F = 4
validate_with_scipy(C, A, b, F)

dataset.append(TestCase(C, A, b, expected_f=F))

# 12 Answer: F=13 X=(12/41, 69/41, 0)
C = [13, 5, 4]
A = [[18, 4, 9], [11, 7, -6]]
b = [12, 15]
F = 13
validate_with_scipy(C, A, b, F)

dataset.append(TestCase(C, A, b, expected_f=F))

# 13 MegaHardRandom Answer: F= 5647027/4639  X=(55865/4639, 80615/4639, 33812/4639)
C = [34, 23, 56]
A = [[-2, 2, 15], [13, 18, -22], [32, -17, 21]]
b = [120, 309, 243]
F = 5647027 / 4639
validate_with_scipy(C, A, b, F)
dataset.append(TestCase(C, A, b, expected_f=F))
