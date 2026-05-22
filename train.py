from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

from .dataset import TASK_CLASS_NAMES, XBDRiskDataset
from .models import build_model


def compute_class_weights(dataset: XBDRiskDataset) -> torch.Tensor:
    counts = [0 for _ in dataset.class_names]
    for row in dataset.rows:
        counts[int(row["target_class"])] += 1
    total = sum(counts)
    weights = [total / max(1, c) for c in counts]
    return torch.tensor(weights, dtype=torch.float32)


def build_weighted_sampler(dataset: XBDRiskDataset) -> WeightedRandomSampler:
    counts = [0 for _ in dataset.class_names]
    for row in dataset.rows:
        counts[int(row["target_class"])] += 1

    sample_weights = []
    for row in dataset.rows:
        cls = int(row["target_class"])
        sample_weights.append(1.0 / max(1, counts[cls]))

    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def confusion_matrix(num_classes: int, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for p, t in zip(preds.view(-1), targets.view(-1)):
        cm[t.long(), p.long()] += 1
    return cm


def macro_f1(cm: torch.Tensor) -> float:
    f1_scores = []
    for i in range(cm.size(0)):
        tp = cm[i, i].item()
        fp = cm[:, i].sum().item() - tp
        fn = cm[i, :].sum().item() - tp
        denom = 2 * tp + fp + fn
        f1_scores.append(0.0 if denom == 0 else (2 * tp) / denom)
    return sum(f1_scores) / len(f1_scores)


def accuracy(cm: torch.Tensor) -> float:
    total = cm.sum().item()
    correct = torch.diag(cm).sum().item()
    return 0.0 if total == 0 else correct / total


def priority_recall(cm: torch.Tensor) -> float:
    idx = cm.size(0) - 1
    tp = cm[idx, idx].item()
    fn = cm[idx, :].sum().item() - tp
    return 0.0 if tp + fn == 0 else tp / (tp + fn)


def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    epoch: int,
    phase: str,
    num_classes: int,
):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_samples = 0
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    progress = tqdm(
        loader,
        desc=f"{phase} epoch {epoch}",
        leave=False,
        dynamic_ncols=True,
    )

    for batch in progress:
        images = batch["image"].to(device)
        targets = batch["target"].to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, targets)

        if training:
            loss.backward()
            optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        preds = logits.argmax(dim=1).cpu()
        cm += confusion_matrix(num_classes, preds, targets.cpu())
        progress.set_postfix(
            loss=f"{(total_loss / max(1, total_samples)):.4f}",
            acc=f"{accuracy(cm):.4f}",
        )

    avg_loss = total_loss / max(1, total_samples)
    return {
        "loss": avg_loss,
        "accuracy": accuracy(cm),
        "macro_f1": macro_f1(cm),
        "priority_recall": priority_recall(cm),
        "confusion_matrix": cm.tolist(),
    }


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def format_confusion_matrix(cm: list[list[int]], class_names: list[str] | None = None) -> str:
    if class_names is None:
        class_names = TASK_CLASS_NAMES["three_class"]
    header = ["true\\pred", *class_names]
    rows = [header]
    for name, values in zip(class_names, cm):
        rows.append([name, *[str(v) for v in values]])

    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return "\n".join(
        "  ".join(cell.rjust(widths[i]) for i, cell in enumerate(row)) for row in rows
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train xBD 3-class risk classifier.")
    parser.add_argument("--index-csv", type=Path, required=True)
    parser.add_argument("--model", choices=["resnet18", "mobilenet_v3_small"], default="mobilenet_v3_small")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--use-weighted-sampler", action="store_true")
    parser.add_argument(
        "--task",
        choices=["three_class", "four_class_building_aware", "high_vs_non_high", "medium_vs_low"],
        default="three_class",
    )
    parser.add_argument(
        "--selection-metric",
        choices=["macro_f1", "high_risk_recall", "balanced_priority"],
        default="high_risk_recall",
    )
    return parser.parse_args()


def selection_score(metrics: dict, metric_name: str) -> float:
    if metric_name == "macro_f1":
        return float(metrics["macro_f1"])
    if metric_name == "high_risk_recall":
        return float(metrics["priority_recall"])
    if metric_name == "balanced_priority":
        return 0.7 * float(metrics["priority_recall"]) + 0.3 * float(metrics["macro_f1"])
    raise ValueError(f"unsupported selection metric: {metric_name}")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = XBDRiskDataset(
        index_csv=args.index_csv,
        mode="train",
        image_size=args.image_size,
        augment=True,
        task=args.task,
    )
    val_dataset = XBDRiskDataset(
        index_csv=args.index_csv,
        mode="val",
        image_size=args.image_size,
        augment=False,
        task=args.task,
    )
    test_dataset = XBDRiskDataset(
        index_csv=args.index_csv,
        mode="test",
        image_size=args.image_size,
        augment=False,
        task=args.task,
    )
    class_names = train_dataset.class_names
    num_classes = len(class_names)
    priority_label = class_names[-1]

    train_sampler = build_weighted_sampler(train_dataset) if args.use_weighted_sampler else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    eval_loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(val_dataset, **eval_loader_kwargs)
    test_loader = DataLoader(test_dataset, **eval_loader_kwargs)

    model = build_model(
        args.model, num_classes=num_classes, in_channels=6, pretrained=args.pretrained
    )
    model.to(device)

    class_weights = compute_class_weights(train_dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_selection_score = -1.0
    best_epoch = -1
    history = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch=epoch,
            phase="train",
            num_classes=num_classes,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            None,
            device,
            epoch=epoch,
            phase="val",
            num_classes=num_classes,
        )
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)
        current_selection_score = selection_score(val_metrics, args.selection_metric)
        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"val_{priority_label}_recall={val_metrics['priority_recall']:.4f} "
            f"selection_metric={args.selection_metric} "
            f"selection_score={current_selection_score:.4f}"
        )

        if current_selection_score > best_selection_score:
            best_selection_score = current_selection_score
            best_epoch = epoch
            torch.save(model.state_dict(), args.output_dir / "best_model.pt")
            save_json(args.output_dir / "best_val_metrics.json", val_metrics)

    torch.save(model.state_dict(), args.output_dir / "last_model.pt")

    best_model_path = args.output_dir / "best_model.pt"
    if best_model_path.exists():
        state_dict = torch.load(best_model_path, map_location=device)
        model.load_state_dict(state_dict)

    test_metrics = run_epoch(
        model,
        test_loader,
        criterion,
        None,
        device,
        epoch=args.epochs,
        phase="test",
        num_classes=num_classes,
    )
    save_json(args.output_dir / "history.json", {"history": history})
    save_json(args.output_dir / "test_metrics.json", test_metrics)
    save_json(
        args.output_dir / "train_config.json",
        {
            "index_csv": str(args.index_csv),
            "model": args.model,
            "task": args.task,
            "class_names": class_names,
            "image_size": args.image_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "num_workers": args.num_workers,
            "pretrained": args.pretrained,
            "use_weighted_sampler": args.use_weighted_sampler,
            "device": str(device),
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "test_size": len(test_dataset),
            "best_epoch": best_epoch,
            "selection_metric": args.selection_metric,
            "best_selection_score": best_selection_score,
        },
    )
    print(
        f"best_epoch={best_epoch} "
        f"selection_metric={args.selection_metric} "
        f"best_selection_score={best_selection_score:.4f}"
    )
    print("test_confusion_matrix:")
    print(format_confusion_matrix(test_metrics["confusion_matrix"], class_names))
    print(f"test_macro_f1={test_metrics['macro_f1']:.4f}")
    print(f"test_{priority_label}_recall={test_metrics['priority_recall']:.4f}")


if __name__ == "__main__":
    main()
