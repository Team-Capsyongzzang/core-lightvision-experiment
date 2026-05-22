from __future__ import annotations

import argparse
import json
import os


QUEUE_NAMES = {
    "high": 3,
    "medium": 2,
    "low": 1,
    "no-building": 0,
}


def _load_boto3_client(region_name: str | None):
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("boto3 is required for SQS access. Install it with `pip install boto3`.") from exc
    return boto3.client("sqs", region_name=region_name)


def _queue_arn(sqs, queue_url: str) -> str:
    attrs = sqs.get_queue_attributes(QueueUrl=queue_url, AttributeNames=["QueueArn"])
    return attrs["Attributes"]["QueueArn"]


def _create_queue(sqs, name: str, attributes: dict[str, str] | None = None) -> str:
    response = sqs.create_queue(QueueName=name, Attributes=attributes or {})
    return response["QueueUrl"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create priority tile SQS queues and matching DLQs.")
    parser.add_argument("--region", default=os.getenv("AWS_REGION", "ap-northeast-2"))
    parser.add_argument("--prefix", default="tile", help="Queue name prefix.")
    parser.add_argument("--max-receive-count", type=int, default=5)
    parser.add_argument("--visibility-timeout", type=int, default=300)
    parser.add_argument("--message-retention-seconds", type=int, default=1209600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sqs = _load_boto3_client(args.region)

    exports = {}
    for suffix in QUEUE_NAMES:
        dlq_name = f"{args.prefix}-{suffix}-dlq"
        queue_name = f"{args.prefix}-{suffix}"

        dlq_url = _create_queue(
            sqs,
            dlq_name,
            {
                "MessageRetentionPeriod": str(args.message_retention_seconds),
            },
        )
        dlq_arn = _queue_arn(sqs, dlq_url)

        queue_url = _create_queue(
            sqs,
            queue_name,
            {
                "VisibilityTimeout": str(args.visibility_timeout),
                "MessageRetentionPeriod": str(args.message_retention_seconds),
                "RedrivePolicy": json.dumps(
                    {
                        "deadLetterTargetArn": dlq_arn,
                        "maxReceiveCount": str(args.max_receive_count),
                    }
                ),
            },
        )
        env_suffix = suffix.replace("-", "_").upper()
        exports[f"SQS_TILE_{env_suffix}_URL"] = queue_url

    print(json.dumps(exports, indent=2))
    print()
    for key, value in exports.items():
        print(f'export {key}="{value}"')


if __name__ == "__main__":
    main()
