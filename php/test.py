"""Python equivalent of php/test.php for clustering Redis embeddings with racplusplus."""
import json
import os
import sys
import struct
from pathlib import Path
from typing import List

import redis
import numpy as np
import racplusplus


def read_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def fetch_embeddings(r: redis.Redis, key_prefix: str) -> List[List[float]]:
    keys = sorted(k.decode("utf-8") if isinstance(k, bytes) else k for k in r.keys(f"{key_prefix}*"))
    points: List[List[float]] = []

    for key in keys:
        embedding_blob = r.hget(key, "embedding")
        if not embedding_blob:
            continue

        floats = [value[0] for value in struct.iter_unpack("<f", embedding_blob)]
        if floats:
            points.append(floats)

    return points


def main() -> None:
    redis_host = os.getenv("REDIS_HOST", "172.18.1.167")
    redis_port = read_env_int("REDIS_PORT", 6379)
    redis_db = read_env_int("REDIS_DB", 0)
    key_prefix = os.getenv("RAC_KEY_PREFIX", "doc:task495_40a01d425d600b4f6b3c3f05cd36f61f:")

    client = redis.Redis(host=redis_host, port=redis_port, db=redis_db)

    points = fetch_embeddings(client, key_prefix)
    if not points:
        print(f"No embeddings with key prefix {key_prefix}", file=sys.stderr)
        raise SystemExit(1)

    base_arr = np.asarray(points, dtype=np.float64)

    max_merge_distance = float(os.getenv("RAC_MAX_MERGE_DISTANCE", "0.35"))
    batch_size = read_env_int("RAC_BATCH_SIZE", 500)
    no_processors = read_env_int("RAC_NO_PROCESSORS", 8)
    distance_metric = os.getenv("RAC_DISTANCE_METRIC", "cosine")

    # labels = racplusplus.rac(
    #     base_arr,
    #     max_merge_distance,
    #     connectivity=None,
    #     batch_size=batch_size,
    #     no_processors=no_processors,
    #     distance_metric=distance_metric,
    # )

    labels = racplusplus.rac(
        base_arr,
        max_merge_distance,
        None,
        batch_size,
        no_processors,
        distance_metric,
    )
    print(labels)

    output_path = os.getenv("RAC_RESULTS_PATH")
    if output_path:
        normalized = Path(output_path).expanduser()
        normalized.parent.mkdir(parents=True, exist_ok=True)
        to_dump = labels.tolist() if hasattr(labels, "tolist") else list(labels)
        normalized.write_text(json.dumps(to_dump, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote clustering results to {normalized}", file=sys.stderr)


if __name__ == "__main__":
    main()
