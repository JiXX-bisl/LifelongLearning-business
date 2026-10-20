#!/usr/bin/env python3
"""用户入口：将已核验的共同种子转成带真实难度标注和检索索引的经验库。"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]


def main():
    sys.path.insert(0, str(ROOT))
    from src.psu_memory.validation.seed_pool import (
        DEFAULT_SEEDS, DEFAULT_BUSINESS_CONFIG, DEFAULT_RETRIEVAL_CONFIG,
        DEFAULT_POOL_CONFIG, initialize_seed_pool,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--business-config", default=DEFAULT_BUSINESS_CONFIG)
    parser.add_argument("--retrieval-config", default=DEFAULT_RETRIEVAL_CONFIG)
    parser.add_argument("--pool-config", default=DEFAULT_POOL_CONFIG)
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()
    run_dir = args.run_dir or ("data/derived/psu_memory/seeds/initial-"
                               + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    try:
        result = initialize_seed_pool(
            run_dir=run_dir, seeds_path=args.seeds, business_config=args.business_config,
            retrieval_config=args.retrieval_config, pool_config=args.pool_config,
        )
    except Exception as error:
        print(type(error).__name__ + ": " + str(error), file=sys.stderr)
        return 1
    print(json.dumps({key: result[key] for key in (
        "ready", "run_dir", "seed_store", "selected_count", "member_count",
        "group_counts", "model_difficulty_counts", "error",
    )}, ensure_ascii=False))
    return 0 if result["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
