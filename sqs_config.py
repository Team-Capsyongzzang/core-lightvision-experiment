from __future__ import annotations


QUEUE_ENV_BY_PRIORITY = {
    3: "SQS_TILE_HIGH_URL",
    2: "SQS_TILE_MEDIUM_URL",
    1: "SQS_TILE_LOW_URL",
    0: "SQS_TILE_NO_BUILDING_URL",
}
