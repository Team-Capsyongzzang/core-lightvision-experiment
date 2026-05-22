from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from .priority_classifier import (
    DEFAULT_STAGE0_CHECKPOINT,
    DEFAULT_STAGE1_CHECKPOINT,
    DEFAULT_STAGE2_CHECKPOINT,
    PriorityClassifier,
)
from .sqs_config import QUEUE_ENV_BY_PRIORITY


def _load_boto3_client(region_name: str | None):
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("boto3 is required for SQS access. Install it with `pip install boto3`.") from exc
    return boto3.client("sqs", region_name=region_name)


def _read_jobs(path: Path) -> Iterable[dict]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    yield json.loads(stripped)
        return

    with path.open("r", encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f)


def _queue_urls_from_env() -> dict[int, str]:
    urls = {}
    missing = []
    for priority, env_name in QUEUE_ENV_BY_PRIORITY.items():
        value = os.getenv(env_name)
        if value:
            urls[priority] = value
        else:
            missing.append(env_name)
    if missing:
        raise RuntimeError(f"missing SQS queue URL environment variables: {', '.join(missing)}")
    return urls


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify tile jobs and publish them to priority SQS queues.")
    parser.add_argument("--input", type=Path, required=True, help="CSV or JSONL with pre_image and post_image fields.")
    parser.add_argument("--region", default=os.getenv("AWS_REGION"))
    parser.add_argument("--dry-run", action="store_true", help="Print classified messages without sending to SQS.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of input jobs to process.")
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

    queue_urls = {} if args.dry_run else _queue_urls_from_env()
    sqs = None if args.dry_run else _load_boto3_client(args.region)

    for idx, job in enumerate(_read_jobs(args.input)):
        if args.limit is not None and idx >= args.limit:
            break

        pre_image = job["pre_image"]
        post_image = job["post_image"]
        prediction = classifier.predict(pre_image, post_image)
        message = {
            "tile_id": job.get("tile_id") or job.get("sample_id"),
            "pre_image": pre_image,
            "post_image": post_image,
            "priority": prediction.priority,
            "label": prediction.label,
            "prediction": asdict(prediction),
            "payload": job,
        }
        body = json.dumps(message, ensure_ascii=False)

        if args.dry_run:
            print(body)
            continue

        sqs.send_message(
            QueueUrl=queue_urls[prediction.priority],
            MessageBody=body,
            MessageAttributes={
                "priority": {"DataType": "Number", "StringValue": str(prediction.priority)},
                "label": {"DataType": "String", "StringValue": prediction.label},
            },
        )


if __name__ == "__main__":
    main()
