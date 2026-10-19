from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chatflow_agent.runtime import (
    BusinessRuntimeError,
    check_business_runtime,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查业务Agent启动条件。")
    parser.add_argument("--config", type=Path, required=True, help="业务配置文件。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = check_business_runtime(args.config)
    except BusinessRuntimeError as error:
        print(f"启动预检失败：{error}", file=sys.stderr)
        return 1

    print("启动预检通过；模型仅检查配置，未请求验证。")
    print(f"数据基线：{result['data']['baseline_id']}")
    print(f"数据库：{result['database']['database']}")
    print(f"输出目录：{result['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
