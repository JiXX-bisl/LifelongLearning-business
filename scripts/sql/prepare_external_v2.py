"""Prepare pinned Geography v4 inputs in a fresh, independent experiment folder."""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.sql_learning.datasets import download_geography_sources, prepare_geography


def main():
    parser = ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path("data/external_v2/geography/source"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--split", choices=("train", "dev", "test"), default="dev")
    parser.add_argument("--split-by", choices=("query", "question"), default="query")
    parser.add_argument("--limit", type=int, default=20, help="0 selects the complete split")
    args = parser.parse_args()
    if args.download:
        download_geography_sources(args.source_dir)
    print(json.dumps(prepare_geography(args.source_dir.resolve(), args.output_dir.resolve(),
                                      split=args.split, split_by=args.split_by,
                                      limit=None if args.limit == 0 else args.limit), indent=2))


if __name__ == "__main__":
    main()
