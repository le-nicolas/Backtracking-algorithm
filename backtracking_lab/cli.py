from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from .linkage import synthesize_fourbar
from .nqueens import NQueensResult, render_nqueens_board, solve_nqueens
from .sudoku import render_sudoku_grid, solve_sudoku


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("Value must be >= 1.")
    return parsed


def _float_list(raw: str) -> list[float]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise argparse.ArgumentTypeError("Expected a comma-separated list of numbers.")
    return values


def _range_spec(raw: str) -> list[float]:
    parts = [item.strip() for item in raw.split(":")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "Range must be formatted as start:stop:step (inclusive stop)."
        )
    start, stop, step = (float(part) for part in parts)
    if step <= 0:
        raise argparse.ArgumentTypeError("Range step must be > 0.")
    if stop < start:
        raise argparse.ArgumentTypeError("Range stop must be >= start.")

    values: list[float] = []
    current = start
    guard = 0
    while current <= stop + 1e-12:
        values.append(round(current, 10))
        current += step
        guard += 1
        if guard > 1_000_000:
            raise argparse.ArgumentTypeError("Range produced too many points.")
    return values


def _print_nqueens(result: NQueensResult, max_boards: int) -> None:
    print(f"N-Queens size: {result.size}")
    print(f"Solutions found: {result.stats.solutions_found}")
    print(f"Nodes visited: {result.stats.nodes_visited}")
    print(f"Backtracks: {result.stats.backtracks}")
    print(f"Elapsed: {result.stats.elapsed_ms:.3f} ms")

    if result.placements:
        for index, placement in enumerate(result.placements[:max_boards], start=1):
            print()
            print(f"Solution {index}")
            print(render_nqueens_board(placement))


def _print_sudoku(result_grid: str, solutions: list[str]) -> None:
    print("Input")
    print(render_sudoku_grid(result_grid))
    if not solutions:
        print("\nNo valid solution found.")
        return
    print("\nSolved")
    print(render_sudoku_grid(solutions[0]))


def _load_sudoku_puzzle(args: argparse.Namespace) -> str:
    if (args.puzzle is None) == (args.puzzle_file is None):
        raise ValueError("Provide exactly one of --puzzle or --puzzle-file.")
    if args.puzzle_file is not None:
        return Path(args.puzzle_file).read_text(encoding="utf-8")
    return args.puzzle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="backtracking-lab",
        description="Backtracking Lab: practical solvers + linkage synthesis.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    nqueens = subparsers.add_parser("nqueens", help="Solve the N-Queens problem.")
    nqueens.add_argument("--size", "-n", type=_positive_int, required=True)
    nqueens.add_argument("--max-solutions", type=_positive_int)
    nqueens.add_argument("--count-only", action="store_true")
    nqueens.add_argument(
        "--column-order",
        choices=("left-to-right", "center-first"),
        default="center-first",
    )
    nqueens.add_argument("--show-boards", type=_positive_int, default=3)
    nqueens.add_argument("--json", action="store_true")

    sudoku = subparsers.add_parser("sudoku", help="Solve Sudoku with MRV backtracking.")
    sudoku.add_argument("--puzzle")
    sudoku.add_argument("--puzzle-file")
    sudoku.add_argument("--max-solutions", type=_positive_int, default=1)
    sudoku.add_argument("--json", action="store_true")

    benchmark = subparsers.add_parser(
        "benchmark", help="Quick N-Queens benchmark across board sizes."
    )
    benchmark.add_argument("--sizes", nargs="+", type=_positive_int, default=[8, 10, 12])
    benchmark.add_argument("--max-solutions", type=_positive_int, default=2)
    benchmark.add_argument(
        "--column-order",
        choices=("left-to-right", "center-first"),
        default="center-first",
    )
    benchmark.add_argument("--json", action="store_true")

    linkage = subparsers.add_parser(
        "linkage",
        help="Hybrid four-bar synthesis: backtracking + gradient refinement.",
    )
    linkage.add_argument(
        "--input-angles",
        type=_float_list,
        required=True,
        help="Comma-separated input crank angles in degrees.",
    )
    linkage.add_argument(
        "--target-angles",
        type=_float_list,
        required=True,
        help="Comma-separated target rocker angles in degrees.",
    )
    linkage.add_argument(
        "--ground-range",
        type=_range_spec,
        default=_range_spec("4:8:1"),
        help="Grid range as start:stop:step.",
    )
    linkage.add_argument(
        "--crank-range",
        type=_range_spec,
        default=_range_spec("2:6:1"),
        help="Grid range as start:stop:step.",
    )
    linkage.add_argument(
        "--coupler-range",
        type=_range_spec,
        default=_range_spec("3:9:1"),
        help="Grid range as start:stop:step.",
    )
    linkage.add_argument(
        "--rocker-range",
        type=_range_spec,
        default=_range_spec("2:6:1"),
        help="Grid range as start:stop:step.",
    )
    linkage.add_argument("--top-k", type=_positive_int, default=5)
    linkage.add_argument("--min-transmission", type=float, default=8.0)
    linkage.add_argument("--require-grashof", action="store_true")
    linkage.add_argument("--refine-steps", type=_positive_int, default=120)
    linkage.add_argument("--no-refine", action="store_true")
    linkage.add_argument("--json", action="store_true")

    return parser


def _run_nqueens(args: argparse.Namespace) -> int:
    result = solve_nqueens(
        args.size,
        max_solutions=args.max_solutions,
        count_only=args.count_only,
        column_order=args.column_order,
    )
    if args.json:
        print(json.dumps(result.to_dict(include_boards=True), indent=2))
    else:
        _print_nqueens(result, max_boards=args.show_boards)
    return 0


def _run_sudoku(args: argparse.Namespace) -> int:
    puzzle = _load_sudoku_puzzle(args)
    result = solve_sudoku(puzzle, max_solutions=args.max_solutions)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        _print_sudoku(result.puzzle, result.solutions)
        print()
        print(f"Solutions found: {result.stats.solutions_found}")
        print(f"Nodes visited: {result.stats.nodes_visited}")
        print(f"Backtracks: {result.stats.backtracks}")
        print(f"Elapsed: {result.stats.elapsed_ms:.3f} ms")
    return 0


def _run_benchmark(args: argparse.Namespace) -> int:
    rows: list[dict[str, float | int]] = []
    for size in args.sizes:
        result = solve_nqueens(
            size=size,
            max_solutions=args.max_solutions,
            count_only=True,
            column_order=args.column_order,
        )
        rows.append(
            {
                "size": size,
                "solutions_found": result.stats.solutions_found,
                "nodes_visited": result.stats.nodes_visited,
                "backtracks": result.stats.backtracks,
                "elapsed_ms": result.stats.elapsed_ms,
            }
        )

    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print("size  solutions  nodes     backtracks  elapsed_ms")
        for row in rows:
            print(
                f"{row['size']:>4}  "
                f"{row['solutions_found']:>9}  "
                f"{row['nodes_visited']:>8}  "
                f"{row['backtracks']:>10}  "
                f"{row['elapsed_ms']:>10.3f}"
            )
    return 0


def _run_linkage(args: argparse.Namespace) -> int:
    result = synthesize_fourbar(
        args.input_angles,
        args.target_angles,
        ground_values=args.ground_range,
        crank_values=args.crank_range,
        coupler_values=args.coupler_range,
        rocker_values=args.rocker_range,
        top_k=args.top_k,
        min_transmission_deg=args.min_transmission,
        require_grashof=args.require_grashof,
        refine=not args.no_refine,
        refine_steps=args.refine_steps,
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    print("Four-Bar Synthesis")
    print(f"Input angles (deg):  {', '.join(f'{v:.3f}' for v in result.input_angles_deg)}")
    print(f"Target angles (deg): {', '.join(f'{v:.3f}' for v in result.target_output_deg)}")
    print(
        "Grid shape: "
        f"g={result.grid_shape['ground']}, "
        f"a={result.grid_shape['crank']}, "
        f"b={result.grid_shape['coupler']}, "
        f"c={result.grid_shape['rocker']}"
    )
    print(f"Nodes visited: {result.stats.nodes_visited}")
    print(f"Backtracks: {result.stats.backtracks}")
    print(f"Feasible designs: {result.stats.solutions_found}")
    print(f"Elapsed: {result.stats.elapsed_ms:.3f} ms")

    if not result.candidates:
        print("\nNo feasible mechanism found. Expand ranges or relax constraints.")
        return 0

    print("\nTop candidates")
    for index, candidate in enumerate(result.candidates, start=1):
        print(
            f"{index}. g={candidate.ground:.5f}, a={candidate.crank:.5f}, "
            f"b={candidate.coupler:.5f}, c={candidate.rocker:.5f}, "
            f"branch={candidate.branch:+d}, rmse={candidate.rmse_deg:.4f} deg, "
            f"min_trans={candidate.min_transmission_deg:.3f} deg, "
            f"phase={candidate.phase_offset_deg:.3f} deg"
        )
        if candidate.predicted_output_deg:
            rendered = ", ".join(f"{value:.3f}" for value in candidate.predicted_output_deg)
            print(f"   predicted: {rendered}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "nqueens":
            return _run_nqueens(args)
        if args.command == "sudoku":
            return _run_sudoku(args)
        if args.command == "benchmark":
            return _run_benchmark(args)
        if args.command == "linkage":
            return _run_linkage(args)
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(f"Unknown command: {args.command}", file=sys.stderr)
    return 2
