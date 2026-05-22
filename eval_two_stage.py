from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .dataset import TASK_CLASS_NAMES, load_index, split_rows
from .models import build_model
from .train import format_confusion_matrix, macro_f1


FINAL_CLASS_NAMES = ["low", "medium", "high"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a two-stage xBD risk pipeline and print the final 3-class confusion matrix."
    )
    parser.add_argument("--index-csv", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--stage1-model", choices=["resnet18", "mobilenet_v3_small"], required=True)
    parser.add_argument(
        "--stage1-task",
        choices=["three_class", "high_vs_non_high"],
        required=True,
    )
    parser.add_argument("--stage1-image-size", type=int, default=224)
    parser.add_argument("--stage2-checkpoint", type=Path, required=True)
    parser.add_argument("--stage2-model", choices=["resnet18", "mobilenet_v3_small"], required=True)
    parser.add_argument("--stage2-image-size", type=int, default=320)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def _load_six_channel_tensor(pre_path: str, post_path: str, image_size: int, device: torch.device):
    pre = Image.open(pre_path).convert("RGB").resize((image_size, image_size))
    post = Image.open(post_path).convert("RGB").resize((image_size, image_size))
    pre_arr = np.asarray(pre, dtype=np.float32) / 255.0
    post_arr = np.asarray(post, dtype=np.float32) / 255.0
    stacked = np.concatenate([pre_arr, post_arr], axis=2)
    stacked = np.transpose(stacked, (2, 0, 1))
    return torch.from_numpy(stacked).float().unsqueeze(0).to(device)


def _predict_stage1_high(logits: torch.Tensor, task: str) -> bool:
    pred = int(logits.argmax(dim=1).item())
    if task == "high_vs_non_high":
        return pred == 1
    if task == "three_class":
        return pred == 2
    raise ValueError(f"unsupported stage1 task: {task}")


def _priority_recall_from_cm(cm: np.ndarray) -> float:
    idx = 2
    tp = int(cm[idx, idx])
    fn = int(cm[idx, :].sum() - tp)
    return 0.0 if tp + fn == 0 else tp / (tp + fn)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = split_rows(load_index(args.index_csv), args.split)

    stage1_num_classes = len(TASK_CLASS_NAMES[args.stage1_task])
    stage1_model = build_model(
        args.stage1_model,
        num_classes=stage1_num_classes,
        in_channels=6,
        pretrained=False,
    )
    stage1_model.load_state_dict(torch.load(args.stage1_checkpoint, map_location=device))
    stage1_model.to(device)
    stage1_model.eval()

    stage2_model = build_model(
        args.stage2_model,
        num_classes=len(TASK_CLASS_NAMES["medium_vs_low"]),
        in_channels=6,
        pretrained=False,
    )
    stage2_model.load_state_dict(torch.load(args.stage2_checkpoint, map_location=device))
    stage2_model.to(device)
    stage2_model.eval()

    cm = np.zeros((3, 3), dtype=np.int64)
    routed_to_stage2 = 0
    stage1_high_predictions = 0

    with torch.no_grad():
        for row in tqdm(rows, desc=f"two-stage {args.split}", dynamic_ncols=True):
            true_class = int(row["risk_class"])

            stage1_x = _load_six_channel_tensor(
                row["pre_image"], row["post_image"], args.stage1_image_size, device
            )
            stage1_logits = stage1_model(stage1_x)
            is_high = _predict_stage1_high(stage1_logits, args.stage1_task)

            if is_high:
                pred_class = 2
                stage1_high_predictions += 1
            else:
                routed_to_stage2 += 1
                stage2_x = _load_six_channel_tensor(
                    row["pre_image"], row["post_image"], args.stage2_image_size, device
                )
                stage2_logits = stage2_model(stage2_x)
                stage2_pred = int(stage2_logits.argmax(dim=1).item())
                pred_class = stage2_pred  # 0=low, 1=medium

            cm[true_class, pred_class] += 1

    cm_tensor = torch.tensor(cm.tolist(), dtype=torch.int64)
    metrics = {
        "split": args.split,
        "confusion_matrix": cm.tolist(),
        "macro_f1": macro_f1(cm_tensor),
        "high_risk_recall": _priority_recall_from_cm(cm),
        "stage1_task": args.stage1_task,
        "stage1_high_predictions": stage1_high_predictions,
        "routed_to_stage2": routed_to_stage2,
        "total_samples": len(rows),
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"split={args.split}")
    print("final_confusion_matrix:")
    print(format_confusion_matrix(metrics["confusion_matrix"], FINAL_CLASS_NAMES))
    print(f"macro_f1={metrics['macro_f1']:.4f}")
    print(f"high_risk_recall={metrics['high_risk_recall']:.4f}")
    print(
        f"stage1_high_predictions={metrics['stage1_high_predictions']} "
        f"routed_to_stage2={metrics['routed_to_stage2']} "
        f"total_samples={metrics['total_samples']}"
    )


if __name__ == "__main__":
    main()
