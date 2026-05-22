from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from .constants import DEFAULT_SPLITS, RiskThresholds
from .labels import summarize_thresholds, summarize_tile


def paired_samples(images_dir: Path, labels_dir: Path):
    for post_image in sorted(images_dir.glob("*_post_disaster.png")):
        stem = post_image.name.replace("_post_disaster.png", "")
        pre_image = images_dir / f"{stem}_pre_disaster.png"
        post_label = labels_dir / f"{stem}_post_disaster.json"
        pre_label = labels_dir / f"{stem}_pre_disaster.json"
        if not (pre_image.exists() and post_label.exists() and pre_label.exists()):
            continue
        yield stem, pre_image, post_image, pre_label, post_label


def infer_disaster_type(stem: str) -> str:
    name = stem.split("_")[0]
    if "-" in name:
        return name.split("-")[0]
    return name


def assign_threshold_labels(rows: list[dict], thresholds: RiskThresholds) -> None:
    for row in rows:
        risk_score = float(row["risk_score"])
        if risk_score < thresholds.low_max:
            risk_class = 0
            risk_class_name = "low"
        elif risk_score < thresholds.medium_max:
            risk_class = 1
            risk_class_name = "medium"
        else:
            risk_class = 2
            risk_class_name = "high"

        row["risk_class"] = risk_class
        row["risk_class_name"] = risk_class_name
        row["label_mode"] = "threshold"
        row["quantile_low_cutoff"] = ""
        row["quantile_high_cutoff"] = ""


def assign_building_aware_threshold_labels(rows: list[dict], thresholds: RiskThresholds) -> None:
    for row in rows:
        building_count = int(float(row["building_count"]))
        if building_count <= 0:
            risk_class = 0
            risk_class_name = "no_building"
        else:
            risk_score = float(row["risk_score"])
            if risk_score < thresholds.low_max:
                risk_class = 1
                risk_class_name = "low"
            elif risk_score < thresholds.medium_max:
                risk_class = 2
                risk_class_name = "medium"
            else:
                risk_class = 3
                risk_class_name = "high"

        row["risk_class"] = risk_class
        row["risk_class_name"] = risk_class_name
        row["label_mode"] = "building_threshold"
        row["quantile_low_cutoff"] = ""
        row["quantile_high_cutoff"] = ""


def assign_quantile_labels(rows: list[dict], low_ratio: float, high_ratio: float) -> None:
    assign_quantile_labels_with_reference(
        rows=rows,
        reference_rows=rows,
        low_ratio=low_ratio,
        high_ratio=high_ratio,
    )


def assign_quantile_labels_with_reference(
    rows: list[dict],
    reference_rows: list[dict],
    low_ratio: float,
    high_ratio: float,
) -> None:
    if not rows:
        return

    sorted_reference_rows = sorted(reference_rows, key=lambda row: float(row["risk_score"]))
    total = len(sorted_reference_rows)
    low_count = int(math.floor(total * low_ratio))
    high_count = int(math.floor(total * high_ratio))

    # Keep at least one sample in the medium bucket when possible.
    if low_count + high_count >= total and total >= 3:
        high_count = max(1, high_count - 1)

    low_end = low_count
    high_start = total - high_count

    low_cutoff = (
        float(sorted_reference_rows[low_end - 1]["risk_score"]) if low_end > 0 else float("-inf")
    )
    high_cutoff = (
        float(sorted_reference_rows[high_start]["risk_score"])
        if high_start < total
        else float("inf")
    )

    for row in rows:
        score = float(row["risk_score"])
        if score <= low_cutoff:
            risk_class = 0
            risk_class_name = "low"
        elif score >= high_cutoff:
            risk_class = 2
            risk_class_name = "high"
        else:
            risk_class = 1
            risk_class_name = "medium"

        row["risk_class"] = risk_class
        row["risk_class_name"] = risk_class_name
        row["label_mode"] = "quantile"
        row["quantile_low_cutoff"] = low_cutoff
        row["quantile_high_cutoff"] = high_cutoff


def build_index(
    dataset_root: Path,
    output_csv: Path,
    thresholds: RiskThresholds,
    label_mode: str,
    quantile_low_ratio: float,
    quantile_high_ratio: float,
    quantile_reference_splits: tuple[str, ...],
) -> int:
    rows = []
    for split in DEFAULT_SPLITS:
        split_dir = dataset_root / split
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        if not (images_dir.exists() and labels_dir.exists()):
            continue

        for stem, pre_image, post_image, pre_label, post_label in paired_samples(
            images_dir, labels_dir
        ):
            stats = summarize_tile(pre_label, post_label, thresholds)
            rows.append(
                {
                    "sample_id": stem,
                    "split": split,
                    "disaster": stem.rsplit("_", 1)[0],
                    "disaster_type_hint": infer_disaster_type(stem),
                    "pre_image": str(pre_image),
                    "post_image": str(post_image),
                    "pre_label": str(pre_label),
                    "post_label": str(post_label),
                    **stats,
                }
            )

    if label_mode == "threshold":
        assign_threshold_labels(rows, thresholds)
    elif label_mode == "building_threshold":
        assign_building_aware_threshold_labels(rows, thresholds)
    elif label_mode == "quantile":
        reference_rows = [
            row for row in rows if row["split"] in set(quantile_reference_splits)
        ] or rows
        assign_quantile_labels_with_reference(
            rows,
            reference_rows=reference_rows,
            low_ratio=quantile_low_ratio,
            high_ratio=quantile_high_ratio,
        )
    else:
        raise ValueError(f"unsupported label mode: {label_mode}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "split",
        "disaster",
        "disaster_type_hint",
        "pre_image",
        "post_image",
        "pre_label",
        "post_label",
        "building_count",
        "post_building_count",
        "building_coverage",
        "mean_damage",
        "max_damage",
        "mean_damage_norm",
        "max_damage_norm",
        "severe_ratio",
        "destroyed_ratio",
        "severity_score",
        "impact_score",
        "risk_score",
        "risk_class",
        "risk_class_name",
        "label_mode",
        "quantile_low_cutoff",
        "quantile_high_cutoff",
        "class_count_no_damage",
        "class_count_minor",
        "class_count_major",
        "class_count_destroyed",
        "threshold_low_max",
        "threshold_medium_max",
        "coverage_cap",
        "building_count_cap",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build xBD risk index CSV.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--low-max", type=float, default=0.35)
    parser.add_argument("--medium-max", type=float, default=0.65)
    parser.add_argument("--coverage-cap", type=float, default=0.30)
    parser.add_argument("--building-count-cap", type=int, default=40)
    parser.add_argument("--severity-weight", type=float, default=0.8)
    parser.add_argument("--impact-weight", type=float, default=0.2)
    parser.add_argument(
        "--label-mode",
        choices=["threshold", "building_threshold", "quantile"],
        default="threshold",
    )
    parser.add_argument("--quantile-low-ratio", type=float, default=0.30)
    parser.add_argument("--quantile-high-ratio", type=float, default=0.30)
    parser.add_argument(
        "--quantile-reference-splits",
        nargs="+",
        default=["tier1", "tier3"],
        help="Splits used to fit quantile cutoffs before applying labels to all rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = RiskThresholds(
        low_max=args.low_max,
        medium_max=args.medium_max,
        coverage_cap=args.coverage_cap,
        building_count_cap=args.building_count_cap,
        severity_weight=args.severity_weight,
        impact_weight=args.impact_weight,
    )
    count = build_index(
        dataset_root=args.dataset_root,
        output_csv=args.output,
        thresholds=thresholds,
        label_mode=args.label_mode,
        quantile_low_ratio=args.quantile_low_ratio,
        quantile_high_ratio=args.quantile_high_ratio,
        quantile_reference_splits=tuple(args.quantile_reference_splits),
    )
    config = summarize_thresholds(thresholds)
    print(f"wrote {count} rows to {args.output}")
    print(f"thresholds={config}")
    print(
        "labeling="
        f"{{'mode': '{args.label_mode}', 'quantile_low_ratio': {args.quantile_low_ratio}, "
        f"'quantile_high_ratio': {args.quantile_high_ratio}, "
        f"'quantile_reference_splits': {args.quantile_reference_splits}}}"
    )


if __name__ == "__main__":
    main()
