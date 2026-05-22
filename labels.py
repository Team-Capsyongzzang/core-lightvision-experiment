from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from .constants import DAMAGE_TO_SCORE, RISK_CLASS_NAMES, RiskThresholds


def parse_polygon_wkt(wkt: str) -> list[tuple[float, float]]:
    """Parse a simple POLYGON WKT string into a list of xy tuples."""
    prefix = "POLYGON (("
    suffix = "))"
    if not (wkt.startswith(prefix) and wkt.endswith(suffix)):
        return []
    body = wkt[len(prefix) : -len(suffix)]
    points = []
    for chunk in body.split(","):
        parts = chunk.strip().split()
        if len(parts) < 2:
            continue
        try:
            x = float(parts[0])
            y = float(parts[1])
        except ValueError:
            continue
        points.append((x, y))
    return points


def polygon_area(points: Iterable[tuple[float, float]]) -> float:
    pts = list(points)
    if len(pts) < 3:
        return 0.0
    area = 0.0
    for i, (x1, y1) in enumerate(pts):
        x2, y2 = pts[(i + 1) % len(pts)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _building_features(label_json: dict, key: str = "xy") -> list[dict]:
    features = label_json.get("features", {}).get(key, [])
    return [
        feat
        for feat in features
        if feat.get("properties", {}).get("feature_type") == "building"
    ]


def summarize_tile(
    pre_label_path: Path,
    post_label_path: Path,
    thresholds: RiskThresholds,
) -> dict:
    pre_json = _load_json(pre_label_path)
    post_json = _load_json(post_label_path)

    pre_buildings = _building_features(pre_json)
    post_buildings = _building_features(post_json)

    damage_scores = []
    class_counts = {name: 0 for name in DAMAGE_TO_SCORE}
    max_damage = 0.0

    for feat in post_buildings:
        subtype = feat.get("properties", {}).get("subtype")
        if subtype not in DAMAGE_TO_SCORE:
            continue
        score = DAMAGE_TO_SCORE[subtype]
        class_counts[subtype] += 1
        damage_scores.append(score)
        max_damage = max(max_damage, score)

    building_count = len(pre_buildings) if pre_buildings else len(post_buildings)
    total_damage_buildings = len(damage_scores)

    pre_total_area = 0.0
    for feat in pre_buildings:
        wkt = feat.get("wkt", "")
        pre_total_area += polygon_area(parse_polygon_wkt(wkt))

    image_meta = post_json.get("metadata", {})
    width = int(image_meta.get("width", 1024))
    height = int(image_meta.get("height", 1024))
    image_area = max(1, width * height)

    mean_damage = sum(damage_scores) / len(damage_scores) if damage_scores else 0.0
    mean_damage_norm = mean_damage / 3.0
    max_damage_norm = max_damage / 3.0
    severe_ratio = (
        (class_counts["major-damage"] + class_counts["destroyed"]) / total_damage_buildings
        if total_damage_buildings
        else 0.0
    )
    destroyed_ratio = (
        class_counts["destroyed"] / total_damage_buildings if total_damage_buildings else 0.0
    )
    building_coverage = min(1.0, pre_total_area / image_area)
    building_coverage_norm = min(
        1.0, building_coverage / max(1e-6, thresholds.coverage_cap)
    )
    building_count_norm = min(
        1.0, building_count / max(1, thresholds.building_count_cap)
    )

    severity_score = 0.7 * mean_damage_norm + 0.3 * max_damage_norm
    impact_score = 0.5 * building_coverage_norm + 0.5 * building_count_norm
    risk_score = (
        thresholds.severity_weight * severity_score
        + thresholds.impact_weight * impact_score
    )

    if risk_score < thresholds.low_max:
        risk_class = 0
    elif risk_score < thresholds.medium_max:
        risk_class = 1
    else:
        risk_class = 2

    return {
        "building_count": building_count,
        "post_building_count": total_damage_buildings,
        "building_coverage": building_coverage,
        "mean_damage": mean_damage,
        "max_damage": max_damage,
        "mean_damage_norm": mean_damage_norm,
        "max_damage_norm": max_damage_norm,
        "severe_ratio": severe_ratio,
        "destroyed_ratio": destroyed_ratio,
        "severity_score": severity_score,
        "impact_score": impact_score,
        "risk_score": risk_score,
        "risk_class": risk_class,
        "risk_class_name": RISK_CLASS_NAMES[risk_class],
        "class_count_no_damage": class_counts["no-damage"],
        "class_count_minor": class_counts["minor-damage"],
        "class_count_major": class_counts["major-damage"],
        "class_count_destroyed": class_counts["destroyed"],
        "threshold_low_max": thresholds.low_max,
        "threshold_medium_max": thresholds.medium_max,
        "coverage_cap": thresholds.coverage_cap,
        "building_count_cap": thresholds.building_count_cap,
    }


def summarize_thresholds(thresholds: RiskThresholds) -> dict:
    return asdict(thresholds)
