from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

from .sqs_config import QUEUE_ENV_BY_PRIORITY


POLL_ORDER = (3, 2, 1, 0)


def _load_boto3_client(region_name: str | None):
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("boto3 is required for SQS access. Install it with `pip install boto3`.") from exc
    return boto3.client("sqs", region_name=region_name)


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


def _process_message(body: str, process_command: str | None) -> None:
    job = json.loads(body)
    if process_command is None:
        print(json.dumps({"processed": job.get("tile_id"), "priority": job.get("priority"), "label": job.get("label")}))
        return

    env = os.environ.copy()
    env["TILE_JOB_JSON"] = body
    completed = subprocess.run(process_command, shell=True, env=env, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"process command failed with exit code {completed.returncode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Poll priority SQS queues in high-to-low order and process messages.")
    parser.add_argument("--region", default=os.getenv("AWS_REGION"))
    parser.add_argument("--wait-time-seconds", type=int, default=10)
    parser.add_argument("--visibility-timeout", type=int, default=300)
    parser.add_argument("--idle-sleep-seconds", type=float, default=1.0)
    parser.add_argument(
        "--process-command",
        default=os.getenv("TILE_PROCESS_COMMAND"),
        help="Optional shell command. Receives the SQS message JSON in TILE_JOB_JSON.",
    )
    parser.add_argument("--once", action="store_true", help="Poll until one message is processed or all queues are empty.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sqs = _load_boto3_client(args.region)
    queue_urls = _queue_urls_from_env()

    while True:
        processed = False
        for priority in POLL_ORDER:
            queue_url = queue_urls[priority]
            response = sqs.receive_message(
                QueueUrl=queue_url,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=args.wait_time_seconds,
                VisibilityTimeout=args.visibility_timeout,
            )
            messages = response.get("Messages", [])
            if not messages:
                continue

            message = messages[0]
            try:
                _process_message(message["Body"], args.process_command)
            except Exception as exc:
                print(json.dumps({"status": "failed", "priority": priority, "error": str(exc)}), flush=True)
                raise

            sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=message["ReceiptHandle"])
            processed = True
            break

        if args.once:
            return
        if not processed:
            time.sleep(args.idle_sleep_seconds)


if __name__ == "__main__":
    main()
