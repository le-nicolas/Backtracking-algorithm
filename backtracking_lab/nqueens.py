from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal, Sequence

from .metrics import SearchStats

ColumnOrder = Literal["left-to-right", "center-first"]


@dataclass
class NQueensResult:
    size: int
    placements: list[tuple[int, ...]]
    stats: SearchStats

    def to_dict(self, include_boards: bool = False) -> dict[str, object]:
        payload: dict[str, object] = {
            "size": self.size,
            "solutions_found": self.stats.solutions_found,
            "stats": self.stats.to_dict(),
        }
        if self.placements:
            payload["placements"] = [list(placement) for placement in self.placements]
            if include_boards:
                payload["boards"] = [
                    render_nqueens_board(placement).splitlines()
                    for placement in self.placements
                ]
        return payload


def _ordered_columns(size: int, column_order: ColumnOrder) -> list[int]:
    columns = list(range(size))
    if column_order == "left-to-right":
        return columns
    center = (size - 1) / 2
    return sorted(columns, key=lambda col: (abs(col - center), col))


def solve_nqueens(
    size: int,
    *,
    max_solutions: int | None = None,
    count_only: bool = False,
    column_order: ColumnOrder = "center-first",
) -> NQueensResult:
    if size < 1:
        raise ValueError("Board size must be at least 1.")
    if max_solutions is not None and max_solutions < 1:
        raise ValueError("max_solutions must be >= 1.")

    placements: list[tuple[int, ...]] = []
    current = [-1] * size
    stats = SearchStats()

    used_columns = [False] * size
    used_diag_down = [False] * (2 * size - 1)
    used_diag_up = [False] * (2 * size - 1)
    columns_in_order = _ordered_columns(size, column_order)

    start = perf_counter()

    def backtrack(row: int) -> bool:
        if max_solutions is not None and stats.solutions_found >= max_solutions:
            return True
        if row == size:
            stats.solutions_found += 1
            if not count_only:
                placements.append(tuple(current))
            return max_solutions is not None and stats.solutions_found >= max_solutions

        for col in columns_in_order:
            stats.nodes_visited += 1
            d_down = row - col + size - 1
            d_up = row + col
            if used_columns[col] or used_diag_down[d_down] or used_diag_up[d_up]:
                continue

            current[row] = col
            used_columns[col] = True
            used_diag_down[d_down] = True
            used_diag_up[d_up] = True

            should_stop = backtrack(row + 1)

            used_columns[col] = False
            used_diag_down[d_down] = False
            used_diag_up[d_up] = False
            current[row] = -1
            stats.backtracks += 1

            if should_stop:
                return True
        return False

    backtrack(0)
    stats.elapsed_ms = (perf_counter() - start) * 1000
    return NQueensResult(size=size, placements=placements, stats=stats)


def render_nqueens_board(placement: Sequence[int]) -> str:
    size = len(placement)
    rows = []
    for queen_col in placement:
        row = " ".join("Q" if col == queen_col else "." for col in range(size))
        rows.append(row)
    return "\n".join(rows)
