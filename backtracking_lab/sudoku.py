from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from .metrics import SearchStats

FULL_MASK = (1 << 9) - 1


@dataclass
class SudokuResult:
    puzzle: str
    solutions: list[str]
    stats: SearchStats

    def to_dict(self) -> dict[str, object]:
        return {
            "puzzle": self.puzzle,
            "solutions": self.solutions,
            "solutions_found": self.stats.solutions_found,
            "stats": self.stats.to_dict(),
        }


def normalize_puzzle(puzzle: str) -> str:
    cleaned = [char for char in puzzle if char in "0123456789."]
    if len(cleaned) != 81:
        raise ValueError("Sudoku puzzle must contain exactly 81 cells.")
    return "".join("0" if char == "." else char for char in cleaned)


def _box_index(row: int, col: int) -> int:
    return (row // 3) * 3 + (col // 3)


def solve_sudoku(puzzle: str, *, max_solutions: int = 1) -> SudokuResult:
    if max_solutions < 1:
        raise ValueError("max_solutions must be >= 1.")

    normalized = normalize_puzzle(puzzle)
    grid = [int(char) for char in normalized]

    row_masks = [0] * 9
    col_masks = [0] * 9
    box_masks = [0] * 9

    for index, digit in enumerate(grid):
        if digit == 0:
            continue
        row = index // 9
        col = index % 9
        box = _box_index(row, col)
        bit = 1 << (digit - 1)
        if row_masks[row] & bit or col_masks[col] & bit or box_masks[box] & bit:
            raise ValueError("Puzzle contains conflicting digits.")
        row_masks[row] |= bit
        col_masks[col] |= bit
        box_masks[box] |= bit

    stats = SearchStats()
    solutions: list[str] = []
    start = perf_counter()

    def backtrack() -> bool:
        if stats.solutions_found >= max_solutions:
            return True

        best_index = -1
        best_mask = 0
        best_count = 10

        for index, value in enumerate(grid):
            if value != 0:
                continue
            row = index // 9
            col = index % 9
            box = _box_index(row, col)
            used = row_masks[row] | col_masks[col] | box_masks[box]
            candidate_mask = FULL_MASK & ~used
            candidate_count = candidate_mask.bit_count()

            if candidate_count == 0:
                return False
            if candidate_count < best_count:
                best_count = candidate_count
                best_index = index
                best_mask = candidate_mask
                if candidate_count == 1:
                    break

        if best_index == -1:
            stats.solutions_found += 1
            solutions.append("".join(str(value) for value in grid))
            return stats.solutions_found >= max_solutions

        row = best_index // 9
        col = best_index % 9
        box = _box_index(row, col)
        candidate_mask = best_mask

        while candidate_mask:
            bit = candidate_mask & -candidate_mask
            digit = bit.bit_length()
            stats.nodes_visited += 1

            grid[best_index] = digit
            row_masks[row] |= bit
            col_masks[col] |= bit
            box_masks[box] |= bit

            should_stop = backtrack()

            row_masks[row] &= ~bit
            col_masks[col] &= ~bit
            box_masks[box] &= ~bit
            grid[best_index] = 0
            stats.backtracks += 1

            if should_stop:
                return True
            candidate_mask ^= bit
        return False

    backtrack()
    stats.elapsed_ms = (perf_counter() - start) * 1000
    return SudokuResult(puzzle=normalized, solutions=solutions, stats=stats)


def render_sudoku_grid(grid: str) -> str:
    normalized = normalize_puzzle(grid)
    lines: list[str] = []
    for row in range(9):
        row_values = normalized[row * 9 : (row + 1) * 9]
        display = [value if value != "0" else "." for value in row_values]
        lines.append(
            " ".join(display[0:3])
            + " | "
            + " ".join(display[3:6])
            + " | "
            + " ".join(display[6:9])
        )
        if row in (2, 5):
            lines.append("------+-------+------")
    return "\n".join(lines)
