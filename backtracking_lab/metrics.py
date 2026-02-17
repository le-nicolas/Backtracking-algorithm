from dataclasses import asdict, dataclass


@dataclass
class SearchStats:
    nodes_visited: int = 0
    backtracks: int = 0
    solutions_found: int = 0
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)
