#!/usr/bin/env python3
"""用户入口：把已确定的人工难度复核应用到新的种子库，无模型或嵌入请求。"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]


def main():
    sys.path.insert(0, str(ROOT))
    from src.psu_memory.validation.seed_pool import DEFAULT_DIFFICULTY_REVIEW, apply_difficulty_review

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", default=DEFAULT_DIFFICULTY_REVIEW)
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()
    run_dir = args.run_dir or ("data/derived/psu_memory/seeds/reviewed-"
                               + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    try:
        result = apply_difficulty_review(run_dir=run_dir, review_path=args.review)
    except Exception as error:
        print(type(error).__name__ + ": " + str(error), file=sys.stderr)
        return 1
    print(json.dumps({
        **{key: result[key] for key in ("ready", "run_dir", "seed_store", "effective_difficulty_counts", "error")},
        "manual_review": {"applied": result["manual_review"]["applied"],
                          "changes": len(result["manual_review"]["changes"])},
    }, ensure_ascii=False))
    return 0 if result["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
