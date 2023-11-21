import logging

import numpy as np
import pandas as pd


class TransportationProblem:
    """
    A transportation problem.
    """

    _costs: pd.DataFrame
    _supplies: pd.Series
    _demands: pd.Series

    @property
    def costs(self) -> pd.DataFrame:
        """
        The costs of the transportation problem.
        """
        return self._costs

    @property
    def supplies(self) -> pd.Series:
        """
        The supplies of the transportation problem.
        """
        return self._supplies

    @property
    def demands(self) -> pd.Series:
        """
        The demands of the transportation problem.
        """
        return self._demands

    def __init__(
        self,
        costs: pd.DataFrame | list[list[float]],
        supplies: pd.Series | list[float],
        demands: pd.Series | list[float],
    ):
        sources = [f"S{i}" for i in range(1, len(supplies) + 1)]
        destinations = [f"D{i}" for i in range(1, len(demands) + 1)]

        if isinstance(costs, list):
            costs = pd.DataFrame(costs, index=sources, columns=destinations)

        if isinstance(supplies, list):
            supplies = pd.Series(supplies, index=sources)

        if isinstance(demands, list):
            demands = pd.Series(demands, index=destinations)

        self._costs = costs
        self._supplies = supplies
        self._demands = demands


class TransportationSolution:
    """
    A solution of a transportation problem.
    """

    problem: TransportationProblem
    """The problem"""

    choices: list[tuple[int, int, float]]
    """The solution"""

    @property
    def cost(self) -> float:
        """
        The cost of the solution.
        """
        return sum(
            self.problem.costs.iloc[i, j] * amount for i, j, amount in self.choices
        )

    def __init__(
        self,
        problem: TransportationProblem,
        choices: list[tuple[int, int, float]],
    ):
        self.problem = problem
        self.choices = choices

    def __str__(self):
        result_string = ["Solution:", f"Cost = {self.cost}", "Choices:"]
        for i, j, amount in self.choices:
            result_string.append(
                f"{amount} from {self.problem.costs.index[i]} to {self.problem.costs.columns[j]}"
            )
        return "\n".join(result_string)


def solve_using_vogel(
    problem: TransportationProblem,
) -> TransportationSolution:
    """
    Solves a transportation problem using the Vogel's approximation method.

    :param problem: The transportation problem
    :return: A solution of the transportation problem
    """

    def get_vogel_penalty(costs: pd.DataFrame) -> tuple[int, int]:
        """
        Computes the Vogel's penalty for the transportation problem considering NaN values.

        :param costs: The costs of the transportation problem
        :return: The Vogel's penalty
        """

        # Fill NaN values with a large number, assuming these are costs and not feasible routes
        filled_costs = costs.fillna(np.inf)

        from_row = filled_costs.values.copy()
        from_col = filled_costs.values.copy()

        num_rows, num_cols = filled_costs.shape

        min_from_row = np.zeros(num_rows)
        min_from_col = np.zeros(num_cols)
        prev_min_from_row = np.zeros(num_rows)
        prev_min_from_col = np.zeros(num_cols)

        for i in range(num_rows):
            from_row[i, :] = np.sort(from_row[i, :])
            iterator = (x for x in from_row[i, :] if x != np.inf)
            min_from_row[i] = next(iterator, np.inf)
            prev_min_from_row[i] = next(iterator, np.inf)

        for j in range(num_cols):
            from_col[:, j] = np.sort(from_col[:, j])
            iterator = (x for x in from_col[:, j] if x != np.inf)
            min_from_col[j] = next(iterator, np.inf)
            prev_min_from_col[j] = next(iterator, np.inf)

        with np.errstate(invalid="ignore"):
            diff_from_row = np.subtract(
                prev_min_from_row, min_from_row, where=np.isfinite
            )
            diff_from_col = np.subtract(
                prev_min_from_col, min_from_col, where=np.isfinite
            )

        max_diff_from_row = np.nanmax(diff_from_row)
        max_diff_from_col = np.nanmax(diff_from_col)

        logging.info(f"Diff from row: {diff_from_row}")
        logging.info(f"Diff from col: {diff_from_col}")

        if max_diff_from_row > max_diff_from_col:
            max_penalty_row = np.nanargmax(diff_from_row)
            max_penalty_col = np.nanargmin(filled_costs.iloc[max_penalty_row])
        else:
            max_penalty_col = np.nanargmax(diff_from_col)
            max_penalty_row = np.nanargmin(filled_costs.iloc[:, max_penalty_col])

        logging.info(
            f"Max penalty row: {max_penalty_row}, Max penalty col: {max_penalty_col}"
        )
        return max_penalty_row, max_penalty_col

    # Initialize variables and containers
    supply = problem.supplies.copy()
    solution_df = problem.costs.copy()
    demand = problem.demands.copy()
    choices = []

    while True:
        if supply.sum() == 0 and demand.sum() == 0:
            logging.info("Supply and demand are depleted.")
            break

        util_log_current_state(supply, demand, solution_df)

        max_penalty_row, max_penalty_col = get_vogel_penalty(solution_df)

        util_step(
            supply=supply,
            demand=demand,
            solution_df=solution_df,
            col_idx=max_penalty_col,
            row_idx=max_penalty_row,
            choices=choices,
        )

    return TransportationSolution(problem, choices)


def util_log_current_state(supply, demand, solution_df):
    logging.info(f"Supply:\n{supply}")
    logging.info(f"Demand:\n{demand}")
    logging.info(f"Costs:\n{solution_df}")


def solve_using_north_west_corner(
    problem: TransportationProblem,
) -> TransportationSolution:
    """
    Solves a transportation problem using the North-West Corner Rule.

    :param problem: The transportation problem
    :return: A solution of the transportation problem
    """

    # Initialize variables and containers
    supply = problem.supplies.copy()
    solution_df = problem.costs.copy()
    demand = problem.demands.copy()
    choices = []
    pivot_row = 0
    pivot_col = 0

    while True:
        if supply.sum() == 0 and demand.sum() == 0:
            logging.info("Supply and demand are depleted.")
            break

        util_log_current_state(supply, demand, solution_df)

        # Find the minimum between supply and demand
        min_shipment = min(supply.iloc[pivot_row], demand.iloc[pivot_col])

        logging.info(f"Min shipment: {min_shipment}")
        # Allocate shipments
        choices.append((pivot_row, pivot_col, min_shipment))
        # Update supply and demand
        supply.iloc[pivot_row] -= min_shipment
        demand.iloc[pivot_col] -= min_shipment

        # Update costs matrix (reduce affected row/column to NaN)
        if supply.iloc[pivot_row] == 0:
            logging.info(f"Supply {solution_df.index[pivot_row]} is depleted.")
            solution_df.iloc[pivot_row, :] = np.nan
            pivot_row += 1

        if demand.iloc[pivot_col] == 0:
            logging.info(f"Demand {solution_df.columns[pivot_col]} is depleted.")
            solution_df.iloc[:, pivot_col] = np.nan
            pivot_col += 1

    return TransportationSolution(problem, choices)


def util_step(
    supply: pd.Series,
    demand: pd.Series,
    solution_df: pd.DataFrame,
    col_idx: int,
    row_idx: int,
    choices: list[tuple[int, int, float]],
):
    # Find the minimum between supply and demand
    min_shipment = min(supply.iloc[row_idx], demand.iloc[col_idx])
    logging.info(f"Min shipment: {min_shipment}")
    # Allocate shipments
    choices.append((row_idx, col_idx, min_shipment))
    # Update supply and demand
    supply.iloc[row_idx] -= min_shipment
    demand.iloc[col_idx] -= min_shipment
    # Update costs matrix (reduce affected row/column to NaN)
    if supply.iloc[row_idx] == 0:
        logging.info(f"Supply {solution_df.index[row_idx]} is depleted.")
        solution_df.iloc[row_idx, :] = np.nan
    if demand.iloc[col_idx] == 0:
        logging.info(f"Demand {solution_df.columns[col_idx]} is depleted.")
        solution_df.iloc[:, col_idx] = np.nan


def solve_using_russells(problem: TransportationProblem) -> TransportationSolution:
    """
    Solves a transportation problem using Russell's Approximation method.

    :param problem: The transportation problem
    :return: A solution of the transportation problem
    """

    def get_russell_penalty(costs: pd.DataFrame) -> tuple[int, int]:
        """
        Computes the Russell's penalty for the transportation problem considering NaN values.

        :param costs: The costs of the transportation problem
        :return: The Russell's penalty
        """

        # Fill NaN values with a large number, assuming these are costs and not feasible routes
        filled_costs = costs.values.copy()

        num_rows, num_cols = filled_costs.shape

        row_penalties = np.nanmax(
            filled_costs, where=np.isfinite(filled_costs), axis=1, initial=0
        )

        col_penalties = np.nanmax(
            filled_costs, where=np.isfinite(filled_costs), axis=0, initial=0
        )

        logging.info(f"Row penalties: {row_penalties}")
        logging.info(f"Col penalties: {col_penalties}")
        # fill costs by rule: c_ij = c_ij - max(c_i) - max(c_j)
        filled_costs -= col_penalties
        for i in range(num_rows):
            filled_costs[i] -= row_penalties[i]

        # find the maximum penalty (highest negative value)
        max_penalty_row, max_penalty_col = np.unravel_index(
            np.nanargmin(filled_costs, axis=None), filled_costs.shape
        )

        return max_penalty_row, max_penalty_col

    # Initialize variables and containers
    supply = problem.supplies.copy()
    solution_df = problem.costs.copy()
    demand = problem.demands.copy()
    choices = []

    while True:
        if supply.sum() == 0 and demand.sum() == 0:
            logging.info("Supply and demand are depleted.")
            break

        util_log_current_state(supply, demand, solution_df)

        penalty_row, penalty_col = get_russell_penalty(solution_df)

        util_step(
            supply=supply,
            demand=demand,
            solution_df=solution_df,
            col_idx=penalty_col,
            row_idx=penalty_row,
            choices=choices,
        )

    return TransportationSolution(problem, choices)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    #
    # problem = TransportationProblem(
    #     costs=pd.DataFrame(
    #         [
    #             [3, 2, 7, 6],
    #             [7, 5, 2, 3],
    #             [2, 5, 4, 5],
    #         ],
    #         index=["F1", "F2", "F3"],
    #         columns=["D1", "D2", "D3", "D4"],
    #     ),
    #     supplies=pd.Series([50, 60, 25], index=["F1", "F2", "F3"]),
    #     demands=pd.Series([60, 40, 20, 15], index=["D1", "D2", "D3", "D4"]),
    # )
    M = 1000
    sources = [f"S{i}" for i in range(1, 5)]
    destinations = [f"D{i}" for i in range(1, 6)]

    problem = TransportationProblem(
        costs=pd.DataFrame(
            [
                [16, 16, 13, 22, 17],
                [14, 14, 13, 19, 15],
                [19, 19, 20, 23, M],
                [M, 0, M, 0, 0],
            ],
            index=sources,
            columns=destinations,
        ),
        supplies=pd.Series([50, 60, 50, 50], index=sources),
        demands=pd.Series([30, 20, 70, 30, 60], index=destinations),
    )

    solution = solve_using_vogel(problem)

    print(solution)
