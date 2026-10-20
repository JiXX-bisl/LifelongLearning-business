#!/usr/bin/env python3
"""用户入口：执行已编写 SQL、直接比较结果并收集成功经验，无模型请求。"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]


def main():
    sys.path.insert(0, str(ROOT))
    from src.psu_memory.validation.authored_formation import (
        DATABASE, DEFAULT_BUNDLE, run_authored_formation,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE)
    parser.add_argument("--run-dir", default=None,
                        help="全新目录；省略时使用带时间戳的 authored 形成目录")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3306)
    parser.add_argument("--user", default="synthetic")
    parser.add_argument("--database", default=DATABASE)
    parser.add_argument("--password-env", default="ENTERPRISE_MYSQL_PASSWORD")
    args = parser.parse_args()
    run_dir = args.run_dir or ("data/derived/psu_memory/formation/authored-"
                               + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    try:
        result = run_authored_formation(
            run_dir=run_dir, bundle_path=args.bundle, host=args.host, port=args.port,
            user=args.user, database=args.database, password_env=args.password_env,
        )
    except Exception as error:
        print(type(error).__name__ + ": " + str(error), file=sys.stderr)
        return 1
    print(json.dumps({key: result[key] for key in (
        "run_dir", "case_count", "executed_count", "pass_count", "fail_count",
        "candidate_count", "execution_complete", "candidate_save_complete",
    )}, ensure_ascii=False))
    return 0 if (result["execution_complete"] and result["candidate_save_complete"]
                 and result["fail_count"] == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
