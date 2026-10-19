#!/usr/bin/env python3
"""120家企业数据构建；数据库和验收仅由用户指定子命令启动。"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.enterprise_dataset_v30.build import main

if __name__ == "__main__":
    raise SystemExit(main())
