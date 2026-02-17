from __future__ import annotations

from dataclasses import dataclass
import math
from time import perf_counter
from typing import Sequence

from .metrics import SearchStats

_EPS = 1e-9


@dataclass
class FourBarDesign:
    ground: float
    crank: float
    coupler: float
    rocker: float
    branch: int
    rmse_deg: float
    min_transmission_deg: float
    phase_offset_deg: float = 0.0
    predicted_output_deg: list[float] | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "ground": self.ground,
            "crank": self.crank,
            "coupler": self.coupler,
            "rocker": self.rocker,
            "branch": self.branch,
            "rmse_deg": self.rmse_deg,
            "min_transmission_deg": self.min_transmission_deg,
            "phase_offset_deg": self.phase_offset_deg,
        }
        if self.predicted_output_deg is not None:
            payload["predicted_output_deg"] = self.predicted_output_deg
        return payload


@dataclass
class FourBarSynthesisResult:
    input_angles_deg: list[float]
    target_output_deg: list[float]
    candidates: list[FourBarDesign]
    stats: SearchStats
    grid_shape: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "input_angles_deg": self.input_angles_deg,
            "target_output_deg": self.target_output_deg,
            "grid_shape": self.grid_shape,
            "stats": self.stats.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def _wrap_angle_rad(angle: float) -> float:
    wrapped = (angle + math.pi) % (2 * math.pi) - math.pi
    if wrapped == -math.pi:
        return math.pi
    return wrapped


def _unwrap_to_reference(angle: float, reference: float) -> float:
    return reference + _wrap_angle_rad(angle - reference)


def _angle_error_rad(predicted: float, target: float) -> float:
    return _wrap_angle_rad(predicted - target)


def _rmse_rad(
    predicted: Sequence[float], target: Sequence[float], phase_offset_rad: float
) -> float:
    squared_sum = 0.0
    for predicted_angle, target_angle in zip(predicted, target):
        error = _angle_error_rad(predicted_angle + phase_offset_rad, target_angle)
        squared_sum += error * error
    return math.sqrt(squared_sum / len(predicted))


def _simulate_output_radians(
    input_angles_rad: Sequence[float],
    *,
    ground: float,
    crank: float,
    coupler: float,
    rocker: float,
    branch: int,
) -> tuple[list[float], float] | None:
    if branch not in (-1, 1):
        raise ValueError("branch must be either -1 or 1.")
    if min(ground, crank, coupler, rocker) <= 0:
        return None

    output_angles: list[float] = []
    previous_output: float | None = None
    min_transmission_deg = 180.0

    for theta in input_angles_rad:
        joint_ax = crank * math.cos(theta)
        joint_ay = crank * math.sin(theta)

        vector_x = joint_ax - ground
        vector_y = joint_ay
        distance = math.hypot(vector_x, vector_y)

        if distance <= _EPS:
            return None
        if distance < abs(coupler - rocker) - _EPS:
            return None
        if distance > coupler + rocker + _EPS:
            return None

        gamma = math.atan2(vector_y, vector_x)
        cos_alpha = (rocker * rocker + distance * distance - coupler * coupler) / (
            2.0 * rocker * distance
        )
        cos_alpha = _clamp(cos_alpha, -1.0, 1.0)
        alpha = math.acos(cos_alpha)

        candidate = gamma + alpha if branch > 0 else gamma - alpha
        if previous_output is not None:
            candidate = _unwrap_to_reference(candidate, previous_output)
        output_angles.append(candidate)
        previous_output = candidate

        cos_mu = (coupler * coupler + rocker * rocker - distance * distance) / (
            2.0 * coupler * rocker
        )
        cos_mu = _clamp(cos_mu, -1.0, 1.0)
        transmission = math.degrees(math.acos(cos_mu))
        transmission_margin = min(transmission, 180.0 - transmission)
        min_transmission_deg = min(min_transmission_deg, transmission_margin)

    return output_angles, min_transmission_deg


def simulate_fourbar_output(
    input_angles_deg: Sequence[float],
    *,
    ground: float,
    crank: float,
    coupler: float,
    rocker: float,
    branch: int = 1,
) -> tuple[list[float], float] | None:
    input_angles_rad = [math.radians(angle) for angle in input_angles_deg]
    simulation = _simulate_output_radians(
        input_angles_rad,
        ground=ground,
        crank=crank,
        coupler=coupler,
        rocker=rocker,
        branch=branch,
    )
    if simulation is None:
        return None

    output_rad, min_transmission_deg = simulation
    output_deg = [math.degrees(angle) for angle in output_rad]
    return output_deg, min_transmission_deg


def generate_motion_targets(
    input_angles_deg: Sequence[float],
    *,
    ground: float,
    crank: float,
    coupler: float,
    rocker: float,
    branch: int = 1,
) -> list[float]:
    simulation = simulate_fourbar_output(
        input_angles_deg,
        ground=ground,
        crank=crank,
        coupler=coupler,
        rocker=rocker,
        branch=branch,
    )
    if simulation is None:
        raise ValueError("The provided linkage cannot realize the requested input angles.")
    output_deg, _ = simulation
    return output_deg


def _is_grashof(ground: float, crank: float, coupler: float, rocker: float) -> bool:
    lengths = sorted([ground, crank, coupler, rocker])
    shortest, mid1, mid2, longest = lengths
    return shortest + longest <= mid1 + mid2 + _EPS


def _refine_design(
    design: FourBarDesign,
    *,
    input_angles_rad: Sequence[float],
    target_output_rad: Sequence[float],
    min_transmission_deg: float,
    bounds: dict[str, tuple[float, float]],
    iterations: int,
) -> FourBarDesign:
    params = [
        design.ground,
        design.crank,
        design.coupler,
        design.rocker,
        math.radians(design.phase_offset_deg),
    ]

    def project(values: list[float]) -> list[float]:
        values[0] = _clamp(values[0], bounds["ground"][0], bounds["ground"][1])
        values[1] = _clamp(values[1], bounds["crank"][0], bounds["crank"][1])
        values[2] = _clamp(values[2], bounds["coupler"][0], bounds["coupler"][1])
        values[3] = _clamp(values[3], bounds["rocker"][0], bounds["rocker"][1])
        values[4] = _wrap_angle_rad(values[4])
        return values

    def evaluate(values: list[float]) -> tuple[float, float, float, list[float]]:
        simulation = _simulate_output_radians(
            input_angles_rad,
            ground=values[0],
            crank=values[1],
            coupler=values[2],
            rocker=values[3],
            branch=design.branch,
        )
        if simulation is None:
            return float("inf"), float("inf"), 0.0, []

        predicted_rad, transmission_margin = simulation
        rmse_rad = _rmse_rad(predicted_rad, target_output_rad, values[4])

        penalty = 0.0
        if transmission_margin < min_transmission_deg:
            ratio = (min_transmission_deg - transmission_margin) / min_transmission_deg
            penalty = ratio * ratio

        score = rmse_rad + penalty
        predicted_deg = [math.degrees(angle + values[4]) for angle in predicted_rad]
        return score, math.degrees(rmse_rad), transmission_margin, predicted_deg

    params = project(params)
    best_score, best_rmse, best_transmission, best_prediction = evaluate(params)
    if not math.isfinite(best_score):
        return design

    best_params = params[:]
    learning_rate = 0.15

    for _ in range(iterations):
        gradients: list[float] = []
        for index in range(len(params)):
            step = 1e-4 if index < 4 else 1e-3
            plus = params[:]
            minus = params[:]
            plus[index] += step
            minus[index] -= step
            plus = project(plus)
            minus = project(minus)

            score_plus, _, _, _ = evaluate(plus)
            score_minus, _, _, _ = evaluate(minus)

            if not math.isfinite(score_plus) and not math.isfinite(score_minus):
                gradient = 0.0
            elif not math.isfinite(score_plus):
                gradient = 1.0
            elif not math.isfinite(score_minus):
                gradient = -1.0
            else:
                gradient = (score_plus - score_minus) / (2.0 * step)
            gradients.append(gradient)

        proposal = [value - learning_rate * grad for value, grad in zip(params, gradients)]
        proposal = project(proposal)
        proposal_score, proposal_rmse, proposal_transmission, proposal_prediction = evaluate(
            proposal
        )

        if proposal_score < best_score:
            params = proposal
            best_params = proposal[:]
            best_score = proposal_score
            best_rmse = proposal_rmse
            best_transmission = proposal_transmission
            best_prediction = proposal_prediction
            learning_rate = min(learning_rate * 1.05, 0.4)
        else:
            learning_rate *= 0.5
            if learning_rate < 1e-5:
                break

    return FourBarDesign(
        ground=best_params[0],
        crank=best_params[1],
        coupler=best_params[2],
        rocker=best_params[3],
        branch=design.branch,
        rmse_deg=best_rmse,
        min_transmission_deg=best_transmission,
        phase_offset_deg=math.degrees(best_params[4]),
        predicted_output_deg=best_prediction,
    )


def synthesize_fourbar(
    input_angles_deg: Sequence[float],
    target_output_deg: Sequence[float],
    *,
    ground_values: Sequence[float],
    crank_values: Sequence[float],
    coupler_values: Sequence[float],
    rocker_values: Sequence[float],
    top_k: int = 5,
    min_transmission_deg: float = 8.0,
    require_grashof: bool = False,
    refine: bool = True,
    refine_steps: int = 120,
    max_refine_candidates: int = 8,
) -> FourBarSynthesisResult:
    if len(input_angles_deg) != len(target_output_deg):
        raise ValueError("input_angles_deg and target_output_deg must have the same length.")
    if len(input_angles_deg) < 2:
        raise ValueError("At least two target positions are required.")
    if top_k < 1:
        raise ValueError("top_k must be >= 1.")
    if refine_steps < 1:
        raise ValueError("refine_steps must be >= 1.")
    if min_transmission_deg <= 0:
        raise ValueError("min_transmission_deg must be > 0.")

    if not ground_values or not crank_values or not coupler_values or not rocker_values:
        raise ValueError("Each link must have at least one candidate value.")

    ground_domain = sorted({float(value) for value in ground_values})
    crank_domain = sorted({float(value) for value in crank_values})
    coupler_domain = sorted({float(value) for value in coupler_values})
    rocker_domain = sorted({float(value) for value in rocker_values})

    input_rad = [math.radians(angle) for angle in input_angles_deg]
    target_rad = [math.radians(angle) for angle in target_output_deg]

    stats = SearchStats()
    start = perf_counter()

    candidates: list[FourBarDesign] = []
    cache_distances: dict[tuple[float, float], list[float]] = {}

    def register_candidate(candidate: FourBarDesign) -> None:
        candidates.append(candidate)
        candidates.sort(key=lambda item: item.rmse_deg)
        keep = max(top_k, max_refine_candidates)
        if len(candidates) > keep:
            candidates[:] = candidates[:keep]

    def distances_for(ground: float, crank: float) -> list[float]:
        key = (ground, crank)
        cached = cache_distances.get(key)
        if cached is not None:
            return cached
        values = []
        for theta in input_rad:
            ax = crank * math.cos(theta)
            ay = crank * math.sin(theta)
            values.append(math.hypot(ax - ground, ay))
        cache_distances[key] = values
        return values

    domain = {
        "ground": ground_domain,
        "crank": crank_domain,
        "rocker": rocker_domain,
        "coupler": coupler_domain,
    }
    order = ("ground", "crank", "rocker", "coupler")
    partial: dict[str, float] = {}

    def partial_infeasible() -> bool:
        if "ground" in partial and "crank" in partial:
            all_distances = distances_for(partial["ground"], partial["crank"])

            if "rocker" in partial:
                rocker = partial["rocker"]
                lower = max(abs(distance - rocker) for distance in all_distances)
                upper = min(distance + rocker for distance in all_distances)
                if lower > upper + _EPS:
                    return True

                if "coupler" in partial:
                    coupler = partial["coupler"]
                    if coupler < lower - _EPS or coupler > upper + _EPS:
                        return True
                else:
                    feasible = any(
                        lower - _EPS <= value <= upper + _EPS for value in coupler_domain
                    )
                    if not feasible:
                        return True
        return False

    def evaluate_complete_design() -> None:
        ground = partial["ground"]
        crank = partial["crank"]
        coupler = partial["coupler"]
        rocker = partial["rocker"]

        if require_grashof and not _is_grashof(ground, crank, coupler, rocker):
            stats.backtracks += 1
            return

        local_feasible = False
        for branch in (-1, 1):
            simulation = _simulate_output_radians(
                input_rad,
                ground=ground,
                crank=crank,
                coupler=coupler,
                rocker=rocker,
                branch=branch,
            )
            if simulation is None:
                continue
            predicted_rad, transmission_margin = simulation
            if transmission_margin < min_transmission_deg:
                continue

            local_feasible = True
            stats.solutions_found += 1
            rmse = math.degrees(_rmse_rad(predicted_rad, target_rad, 0.0))
            register_candidate(
                FourBarDesign(
                    ground=ground,
                    crank=crank,
                    coupler=coupler,
                    rocker=rocker,
                    branch=branch,
                    rmse_deg=rmse,
                    min_transmission_deg=transmission_margin,
                    phase_offset_deg=0.0,
                    predicted_output_deg=[math.degrees(value) for value in predicted_rad],
                )
            )
        if not local_feasible:
            stats.backtracks += 1

    def recurse(level: int) -> None:
        if level == len(order):
            evaluate_complete_design()
            return

        name = order[level]
        for value in domain[name]:
            stats.nodes_visited += 1
            partial[name] = value
            if partial_infeasible():
                stats.backtracks += 1
                partial.pop(name, None)
                continue
            recurse(level + 1)
            partial.pop(name, None)

    recurse(0)
    stats.elapsed_ms = (perf_counter() - start) * 1000.0

    if refine and candidates:
        bounds = {
            "ground": (min(ground_domain), max(ground_domain)),
            "crank": (min(crank_domain), max(crank_domain)),
            "coupler": (min(coupler_domain), max(coupler_domain)),
            "rocker": (min(rocker_domain), max(rocker_domain)),
        }
        refined: list[FourBarDesign] = []
        for seed in candidates[:max_refine_candidates]:
            refined.append(
                _refine_design(
                    seed,
                    input_angles_rad=input_rad,
                    target_output_rad=target_rad,
                    min_transmission_deg=min_transmission_deg,
                    bounds=bounds,
                    iterations=refine_steps,
                )
            )

        merged = candidates + refined
        merged.sort(key=lambda item: item.rmse_deg)
        deduplicated: list[FourBarDesign] = []
        seen: set[tuple[float, ...]] = set()
        for item in merged:
            key = (
                round(item.ground, 6),
                round(item.crank, 6),
                round(item.coupler, 6),
                round(item.rocker, 6),
                float(item.branch),
                round(item.phase_offset_deg, 4),
            )
            if key in seen:
                continue
            deduplicated.append(item)
            seen.add(key)
            if len(deduplicated) >= top_k:
                break
        candidates = deduplicated
    else:
        candidates = candidates[:top_k]

    return FourBarSynthesisResult(
        input_angles_deg=[float(value) for value in input_angles_deg],
        target_output_deg=[float(value) for value in target_output_deg],
        candidates=candidates,
        stats=stats,
        grid_shape={
            "ground": len(ground_domain),
            "crank": len(crank_domain),
            "coupler": len(coupler_domain),
            "rocker": len(rocker_domain),
        },
    )
