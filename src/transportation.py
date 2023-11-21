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
        costs: pd.DataFrame,
        supplies: pd.Series,
        demands: pd.Series,
    ):
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
            result_string.append(f"{amount} from {self.problem.costs.index[i]} to {self.problem.costs.columns[j]}")
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
            min_from_row[i] = next((x for x in from_row[i, :] if x != np.inf), np.inf)
            prev_min_from_row[i] = next(
                (x for x in from_row[i, :] if x != min_from_row[i]), np.inf
            )

        for j in range(num_cols):
            from_col[:, j] = np.sort(from_col[:, j])
            min_from_col[j] = next((x for x in from_col[:, j] if x != np.inf), np.inf)
            prev_min_from_col[j] = next(
                (x for x in from_col[:, j] if x != min_from_col[j]), np.inf
            )

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

        return max_penalty_row, max_penalty_col

    # Initialize variables and containers
    supply = problem.supplies.copy()
    solution_df = problem.costs.copy()
    demand = problem.demands.copy()
    choices = []
    while True:
        logging.info(f"Supply:\n{supply}")
        logging.info(f"Demand:\n{demand}")
        logging.info(f"Costs:\n{solution_df}")

        max_penalty_row, max_penalty_col = get_vogel_penalty(solution_df)
        logging.info(
            f"Max penalty row: {max_penalty_row}, Max penalty col: {max_penalty_col}"
        )
        # Find the minimum between supply and demand
        min_shipment = min(supply.iloc[max_penalty_row], demand.iloc[max_penalty_col])
        logging.info(f"Min shipment: {min_shipment}")

        # Allocate shipments
        choices.append((max_penalty_row, max_penalty_col, min_shipment))

        # Update supply and demand
        supply.iloc[max_penalty_row] -= min_shipment
        demand.iloc[max_penalty_col] -= min_shipment

        # Update costs matrix (reduce affected row/column to infinity)
        if supply.iloc[max_penalty_row] == 0:
            logging.info(f"Supply {solution_df.index[max_penalty_row]} is depleted.")
            solution_df.iloc[max_penalty_row, :] = np.nan

        if demand.iloc[max_penalty_col] == 0:
            logging.info(
                f"Demand {solution_df.columns[max_penalty_col]} is depleted."
            )
            solution_df.iloc[:, max_penalty_col] = np.nan

        if supply.sum() == 0 and demand.sum() == 0:
            break

    return TransportationSolution(problem, choices)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    problem = TransportationProblem(
        costs=pd.DataFrame(
            [
                [3, 2, 7, 6],
                [7, 5, 2, 3],
                [2, 5, 4, 5],
            ],
            index=["F1", "F2", "F3"],
            columns=["D1", "D2", "D3", "D4"],
        ),
        supplies=pd.Series([50, 60, 25], index=["F1", "F2", "F3"]),
        demands=pd.Series([60, 40, 20, 15], index=["D1", "D2", "D3", "D4"]),
    )

    solution = solve_using_vogel(problem)

    print(solution)
