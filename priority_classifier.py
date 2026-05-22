from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .models import build_model


DEFAULT_STAGE0_CHECKPOINT = Path("runs/resnet18_stage0_building_vs_no_building_320/best_model.pt")
DEFAULT_STAGE1_CHECKPOINT = Path("runs/resnet18_stage1_high_vs_non_high_320/best_model.pt")
DEFAULT_STAGE2_CHECKPOINT = Path("runs/resnet18_stage2_medium_vs_low_320_buildingaware/best_model.pt")

PRIORITY_LABELS = {
    0: "no_building",
    1: "low",
    2: "medium",
    3: "high",
}


@dataclass(frozen=True)
class PriorityPrediction:
    priority: int
    label: str
    stage0_label: str
    stage1_label: str | None
    stage2_label: str | None
    probabilities: dict[str, dict[str, float]]


def load_six_channel_tensor(
    pre_image: str | Path,
    post_image: str | Path,
    image_size: int,
    device: torch.device,
) -> torch.Tensor:
    pre = Image.open(pre_image).convert("RGB").resize((image_size, image_size))
    post = Image.open(post_image).convert("RGB").resize((image_size, image_size))
    pre_arr = np.asarray(pre, dtype=np.float32) / 255.0
    post_arr = np.asarray(post, dtype=np.float32) / 255.0
    stacked = np.concatenate([pre_arr, post_arr], axis=2)
    stacked = np.transpose(stacked, (2, 0, 1))
    return torch.from_numpy(stacked).float().unsqueeze(0).to(device)


def _load_model(
    checkpoint: Path,
    model_name: str,
    num_classes: int,
    device: torch.device,
) -> torch.nn.Module:
    model = build_model(model_name, num_classes=num_classes, in_channels=6, pretrained=False)
    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _softmax_probs(logits: torch.Tensor, class_names: tuple[str, ...]) -> dict[str, float]:
    values = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().tolist()
    return {name: float(value) for name, value in zip(class_names, values)}


class PriorityClassifier:
    """Three-stage tile priority classifier for scheduler input."""

    def __init__(
        self,
        stage0_checkpoint: Path = DEFAULT_STAGE0_CHECKPOINT,
        stage1_checkpoint: Path = DEFAULT_STAGE1_CHECKPOINT,
        stage2_checkpoint: Path = DEFAULT_STAGE2_CHECKPOINT,
        model_name: str = "resnet18",
        image_size: int = 320,
        device: str | None = None,
        has_building_threshold: float = 0.5,
        high_threshold: float = 0.5,
        medium_threshold: float = 0.5,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_name = model_name
        self.image_size = image_size
        self.has_building_threshold = has_building_threshold
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold

        self.stage0 = _load_model(stage0_checkpoint, model_name, num_classes=2, device=self.device)
        self.stage1 = _load_model(stage1_checkpoint, model_name, num_classes=2, device=self.device)
        self.stage2 = _load_model(stage2_checkpoint, model_name, num_classes=2, device=self.device)

    @torch.no_grad()
    def predict(self, pre_image: str | Path, post_image: str | Path) -> PriorityPrediction:
        x = load_six_channel_tensor(pre_image, post_image, self.image_size, self.device)

        stage0_probs = _softmax_probs(self.stage0(x), ("no_building", "has_building"))
        if stage0_probs["has_building"] < self.has_building_threshold:
            return PriorityPrediction(
                priority=0,
                label=PRIORITY_LABELS[0],
                stage0_label="no_building",
                stage1_label=None,
                stage2_label=None,
                probabilities={"stage0": stage0_probs},
            )

        stage1_probs = _softmax_probs(self.stage1(x), ("non_high", "high"))
        if stage1_probs["high"] >= self.high_threshold:
            return PriorityPrediction(
                priority=3,
                label=PRIORITY_LABELS[3],
                stage0_label="has_building",
                stage1_label="high",
                stage2_label=None,
                probabilities={"stage0": stage0_probs, "stage1": stage1_probs},
            )

        stage2_probs = _softmax_probs(self.stage2(x), ("low", "medium"))
        priority = 2 if stage2_probs["medium"] >= self.medium_threshold else 1
        return PriorityPrediction(
            priority=priority,
            label=PRIORITY_LABELS[priority],
            stage0_label="has_building",
            stage1_label="non_high",
            stage2_label=PRIORITY_LABELS[priority],
            probabilities={
                "stage0": stage0_probs,
                "stage1": stage1_probs,
                "stage2": stage2_probs,
            },
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify one pre/post tile pair into scheduler priority.")
    parser.add_argument("--pre-image", type=Path, required=True)
    parser.add_argument("--post-image", type=Path, required=True)
    parser.add_argument("--stage0-checkpoint", type=Path, default=DEFAULT_STAGE0_CHECKPOINT)
    parser.add_argument("--stage1-checkpoint", type=Path, default=DEFAULT_STAGE1_CHECKPOINT)
    parser.add_argument("--stage2-checkpoint", type=Path, default=DEFAULT_STAGE2_CHECKPOINT)
    parser.add_argument("--model", choices=["resnet18", "mobilenet_v3_small"], default="resnet18")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--device", default=None)
    parser.add_argument("--has-building-threshold", type=float, default=0.5)
    parser.add_argument("--high-threshold", type=float, default=0.5)
    parser.add_argument("--medium-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classifier = PriorityClassifier(
        stage0_checkpoint=args.stage0_checkpoint,
        stage1_checkpoint=args.stage1_checkpoint,
        stage2_checkpoint=args.stage2_checkpoint,
        model_name=args.model,
        image_size=args.image_size,
        device=args.device,
        has_building_threshold=args.has_building_threshold,
        high_threshold=args.high_threshold,
        medium_threshold=args.medium_threshold,
    )
    prediction = classifier.predict(args.pre_image, args.post_image)
    print(json.dumps(asdict(prediction), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
