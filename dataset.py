from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image

TASK_CLASS_NAMES = {
    "three_class": ["low", "medium", "high"],
    "four_class_building_aware": ["no_building", "low", "medium", "high"],
    "building_vs_no_building": ["no_building", "has_building"],
    "high_vs_non_high": ["non_high", "high"],
    "medium_vs_low": ["low", "medium"],
}


def load_index(index_csv: Path) -> list[dict]:
    with index_csv.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def split_rows(rows: list[dict], mode: str) -> list[dict]:
    if mode == "train":
        valid = {"tier1", "tier3"}
    elif mode == "val":
        valid = {"hold"}
    elif mode == "test":
        valid = {"test"}
    else:
        raise ValueError(f"unsupported split mode: {mode}")
    return [row for row in rows if row["split"] in valid]


def prepare_rows_for_task(rows: list[dict], task: str) -> tuple[list[dict], list[str]]:
    if task == "three_class":
        prepared = []
        for row in rows:
            copied = dict(row)
            copied["target_class"] = int(row["risk_class"])
            prepared.append(copied)
        return prepared, TASK_CLASS_NAMES[task]

    if task == "four_class_building_aware":
        prepared = []
        for row in rows:
            copied = dict(row)
            copied["target_class"] = int(row["risk_class"])
            prepared.append(copied)
        return prepared, TASK_CLASS_NAMES[task]

    if task == "building_vs_no_building":
        prepared = []
        for row in rows:
            risk_class = int(row["risk_class"])
            copied = dict(row)
            copied["target_class"] = 0 if risk_class == 0 else 1
            prepared.append(copied)
        return prepared, TASK_CLASS_NAMES[task]

    if task == "high_vs_non_high":
        prepared = []
        high_class = 3 if any(int(row["risk_class"]) == 3 for row in rows) else 2
        for row in rows:
            risk_class = int(row["risk_class"])
            copied = dict(row)
            copied["target_class"] = 1 if risk_class == high_class else 0
            prepared.append(copied)
        return prepared, TASK_CLASS_NAMES[task]

    if task == "medium_vs_low":
        prepared = []
        for row in rows:
            risk_class = int(row["risk_class"])
            if risk_class not in {1, 2}:
                continue
            copied = dict(row)
            copied["target_class"] = 0 if risk_class == 1 else 1
            prepared.append(copied)
        return prepared, TASK_CLASS_NAMES[task]

    raise ValueError(f"unsupported task: {task}")


class XBDRiskDataset:
    def __init__(
        self,
        index_csv: Path,
        mode: str,
        image_size: int = 224,
        augment: bool = False,
        task: str = "three_class",
    ):
        split_filtered_rows = split_rows(load_index(index_csv), mode)
        self.rows, self.class_names = prepare_rows_for_task(split_filtered_rows, task)
        self.task = task
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def _read_rgb(self, path: str) -> np.ndarray:
        img = Image.open(path).convert("RGB").resize((self.image_size, self.image_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr

    def _augment_pair(self, pre: np.ndarray, post: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            pre = np.flip(pre, axis=1).copy()
            post = np.flip(post, axis=1).copy()
        if random.random() < 0.5:
            pre = np.flip(pre, axis=0).copy()
            post = np.flip(post, axis=0).copy()
        return pre, post

    def __getitem__(self, idx: int):
        import torch

        row = self.rows[idx]
        pre = self._read_rgb(row["pre_image"])
        post = self._read_rgb(row["post_image"])
        if self.augment:
            pre, post = self._augment_pair(pre, post)

        stacked = np.concatenate([pre, post], axis=2)  # HWC, 6 channels
        stacked = np.transpose(stacked, (2, 0, 1))  # CHW

        x = torch.from_numpy(stacked).float()
        y = torch.tensor(int(row["target_class"]), dtype=torch.long)
        return {
            "image": x,
            "target": y,
            "risk_score": torch.tensor(float(row["risk_score"]), dtype=torch.float32),
            "sample_id": row["sample_id"],
        }
