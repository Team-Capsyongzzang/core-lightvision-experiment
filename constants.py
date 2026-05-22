from dataclasses import dataclass


DAMAGE_TO_SCORE = {
    "no-damage": 0.0,
    "minor-damage": 1.0,
    "major-damage": 2.0,
    "destroyed": 3.0,
}

RISK_CLASS_NAMES = ("low", "medium", "high")
DEFAULT_SPLITS = ("tier1", "tier3", "hold", "test")


@dataclass(frozen=True)
class RiskThresholds:
    low_max: float = 0.35
    medium_max: float = 0.65
    coverage_cap: float = 0.30
    building_count_cap: int = 40
    severity_weight: float = 0.8
    impact_weight: float = 0.2
