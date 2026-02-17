import unittest

from backtracking_lab.nqueens import render_nqueens_board, solve_nqueens


class NQueensTests(unittest.TestCase):
    def test_known_solution_counts(self) -> None:
        four = solve_nqueens(4, count_only=True)
        five = solve_nqueens(5, count_only=True)
        self.assertEqual(four.stats.solutions_found, 2)
        self.assertEqual(five.stats.solutions_found, 10)

    def test_max_solutions_cap(self) -> None:
        result = solve_nqueens(8, max_solutions=3, count_only=True)
        self.assertEqual(result.stats.solutions_found, 3)

    def test_board_rendering(self) -> None:
        board = render_nqueens_board((1, 3, 0, 2))
        self.assertEqual(len(board.splitlines()), 4)
        self.assertIn("Q", board)


if __name__ == "__main__":
    unittest.main()
