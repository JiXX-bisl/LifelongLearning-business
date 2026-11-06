"""Run the two core validations with the selected Windows interpreter."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
TEST_FILES = (
    "test/test_psu_submission_comparison.py",
    "test/test_psu_direct_runner.py",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("values",), default="values")
    parser.parse_args(argv)
    environment = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(
        [sys.executable, "-B", "-X", "utf8", "-m", "pytest", "-q", *TEST_FILES],
        cwd=ROOT, env=environment, check=False,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
