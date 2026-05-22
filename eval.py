from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import XBDRiskDataset
from .models import build_model
from .train import format_confusion_matrix, run_epoch


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved xBD risk checkpoint.")
    parser.add_argument("--index-csv", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", choices=["resnet18", "mobilenet_v3_small"], required=True)
    parser.add_argument(
        "--task",
        choices=["three_class", "four_class_building_aware", "high_vs_non_high", "medium_vs_low"],
        default="three_class",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = XBDRiskDataset(
        index_csv=args.index_csv,
        mode=args.split,
        image_size=args.image_size,
        augment=False,
        task=args.task,
    )
    class_names = dataset.class_names
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_model(
        args.model, num_classes=len(class_names), in_channels=6, pretrained=False
    )
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    metrics = run_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        optimizer=None,
        device=device,
        epoch=0,
        phase=args.split,
        num_classes=len(class_names),
    )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"checkpoint={args.checkpoint}")
    print(f"split={args.split}")
    print("confusion_matrix:")
    print(format_confusion_matrix(metrics["confusion_matrix"], class_names))
    print(f"macro_f1={metrics['macro_f1']:.4f}")
    print(f"{class_names[-1]}_recall={metrics['priority_recall']:.4f}")


if __name__ == "__main__":
    main()
