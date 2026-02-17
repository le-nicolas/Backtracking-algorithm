import unittest

from backtracking_lab.sudoku import render_sudoku_grid, solve_sudoku


PUZZLE = (
    "530070000"
    "600195000"
    "098000060"
    "800060003"
    "400803001"
    "700020006"
    "060000280"
    "000419005"
    "000080079"
)

SOLUTION = (
    "534678912"
    "672195348"
    "198342567"
    "859761423"
    "426853791"
    "713924856"
    "961537284"
    "287419635"
    "345286179"
)


class SudokuTests(unittest.TestCase):
    def test_solves_reference_puzzle(self) -> None:
        result = solve_sudoku(PUZZLE)
        self.assertEqual(result.stats.solutions_found, 1)
        self.assertEqual(result.solutions[0], SOLUTION)

    def test_rejects_conflicting_puzzle(self) -> None:
        bad = "550000000" + "0" * 72
        with self.assertRaises(ValueError):
            solve_sudoku(bad)

    def test_render_grid_output(self) -> None:
        rendered = render_sudoku_grid(PUZZLE)
        self.assertIn("|", rendered)
        self.assertIn(".", rendered)


if __name__ == "__main__":
    unittest.main()
