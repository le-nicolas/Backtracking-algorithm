"""Backtracking Lab package exports."""

from .linkage import (
    FourBarDesign,
    FourBarSynthesisResult,
    generate_motion_targets,
    simulate_fourbar_output,
    synthesize_fourbar,
)
from .nqueens import NQueensResult, render_nqueens_board, solve_nqueens
from .sudoku import SudokuResult, render_sudoku_grid, solve_sudoku

__all__ = [
    "FourBarDesign",
    "FourBarSynthesisResult",
    "NQueensResult",
    "SudokuResult",
    "generate_motion_targets",
    "render_nqueens_board",
    "render_sudoku_grid",
    "simulate_fourbar_output",
    "solve_nqueens",
    "solve_sudoku",
    "synthesize_fourbar",
]
