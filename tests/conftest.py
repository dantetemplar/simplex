from dataclasses import dataclass
from typing import Collection


@dataclass
class TestCase:
    C: Collection[float]
    A: Collection[Collection[float]]
    b: Collection[float]
    expected_f: float | None = None


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
        if abs(result - F) > 1e-6:
            print(f"scipy: {result=} {F=}")
            print(f"{C=} {A=} {b=}")
            raise AssertionError


def test_pipeline(C, A, b, F):
    validate_with_scipy(C, A, b, F)
    dataset.append(TestCase(C, A, b, expected_f=F))


# 1 Answer: F=12 X=(24/5, 18/5) max
С = [1, 2]
A = [[4, -2], [-1, 3], [-2, -4]]
b = [12, 6, 16]
test_pipeline(С, A, b, 12)

# 2 Answer: F=27/2 X=(0, 9/4, 0) max
C = [3, 6, 2]
A = [[8, 4, 10], [3, 4, 7], [2, 1, 3]]
b = [12, 9, 5]
test_pipeline(C, A, b, 27 / 2)

# 3 Answer: F=391/15 X=(23/15, 0, 0) max
C = [17, 2, 10]
A = [[13, 8, 11], [15, 3, 14]]
b = [20, 23]
test_pipeline(C, A, b, 391 / 15)

# 4 Answer: F=3 X=(3/2, 0) max
C = [2, 1]
A = [[4, 2], [3, 3]]
b = [6, 9]
test_pipeline(C, A, b, 3)

# 5 Answer: F=46/3 X=(5/3, 17/3, 0) max
C = [-1, 3, -3]
A = [[3, -1, -2], [-2, -4, 4], [1, 0, -1], [-2, 2, 8], [3, 0, 0]]
b = [7, 3, 4, 8, 5]
test_pipeline(C, A, b, 46 / 3)

# 6 Answer: F=33 X=(3, 12) max
C = [3, 2]
A = [[2, 1], [2, 3], [3, 1]]
b = [18, 42, 24]
test_pipeline(C, A, b, 33)

# 7 Answer: F=8326 X=(691/5, 118/5) max
C = [50, 60]
A = [[2, 1], [3, 4], [4, 7]]
b = [300, 509, 812]
test_pipeline(C, A, b, 8326)

# 8 Answer: F=110 X=(10, 10) max
C = [6, 5]
A = [[-3, 5], [-2, 5], [1, 0], [3, -8]]
b = [25, 30, 10, 6]
test_pipeline(C, A, b, 110)

# 9 Answer: F=11 X=(3, 1) max
C = [3, 2]
A = [[1, 1], [1, -1]]
b = [4, 2]
test_pipeline(C, A, b, 11)

# 10 Answer: F=27 X=(0, 9, 3) max
C = [1, 2, 3]
A = [[1, 1, 1], [2, 1, 3]]
b = [12, 18]
test_pipeline(C, A, b, 27)

# 11 Answer: F=4 X=(0, 1) max
C = [1, 4]
A = [[2, 1], [3, 5], [1, 3]]
b = [3, 5, 9]
test_pipeline(C, A, b, 4)

# 12 Answer: F=400 x=(40, 0) max
C = [12, 16]
A = [[1, 2], [1, 1]]
b = [40, 30]
test_pipeline(C, A, b, 400)

# 13 Answer: F=17 X=(4, 3) max
C = [2, 3]
A = [[5, 4], [1, 2]]
b = [32, 10]
test_pipeline(C, A, b, 17)

# 14 Answer: F=3.428571 X=(0, 6/7) max
C = [-5, 4]
A = [[2, -14], [1, 7]]
b = [4, 6]
test_pipeline(C, A, b, 3.428571)

# 15 Answer: F=71/5 X=(47/45, 61/45) max
C = [11, 2]
A = [[5, 5], [8, -1], [3, 2]]
b = [12, 7, 6]
test_pipeline(C, A, b, 71 / 5)

# 16 Answer: F=14 X=(2, 0) max
C = [7, 10]
A = [[5, -5], [4, 16]]
b = [10, 8]
test_pipeline(C, A, b, 14)

# 17 Answer: F=13 X=(0, 69/29, 8/29) max
C = [13, 5, 4]
A = [[18, 4, 9], [11, 7, -6]]
b = [12, 15]
test_pipeline(C, A, b, 13)

# 18 Answer: F=20 X=(0, 5, 0) max
C = [3, 4, 6]
A = [[-2, 2, 15], [11, 6, 14], [3, -8, 1]]
b = [12, 30, 24]
test_pipeline(C, A, b, 20)

# 19 Answer: F=13/2 X=(1/6, 0, 5/12) max
C = [14, 7, 10]
A = [[12, 10, 4], [-3, 9, 18], [8, 5, 4], [20, 4, 7]]
b = [12, 7, 3, 7]
test_pipeline(C, A, b, 13 / 2)

wrong_testcase = TestCase(
    C=[14, -4, 10], A=[[7, -5, 8], [-3, 7, -13]], b=[9, 11], expected_f=None
)
