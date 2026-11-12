"""Create an independent enterprise smoke bundle for the generic SQL interface."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.sql_learning.enterprise import prepare_enterprise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=Path,
                        default=Path("data/sql/synthetic_enterprise_v30_120_v1.0.2/tables.json"))
    parser.add_argument("--output", type=Path, required=True,
                        help="New independent directory; existing directories are never overwritten")
    args = parser.parse_args()
    print(prepare_enterprise(args.tables, args.output))


if __name__ == "__main__":
    main()
