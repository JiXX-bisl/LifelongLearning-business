"""Run the existing offline test groups with the selected Windows interpreter."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
DIRECT_UNITTESTS = (
    "chatflow_runtime psu_seed_pool psu_pool_difficulty psu_direct_runner "
    "psu_saved_result_check psu_direct_update psu_query_integration "
    "psu_pool_classification psu_pool_selection psu_active_members psu_quality "
    "psu_model_client psu_retrieval_text psu_embedding psu_retrieval_index "
    "psu_prefilter psu_selector_prompt psu_selector psu_selector_acceptance "
    "psu_retriever psu_augmentation psu_prompt_enhancement "
    "file_lock psu_cuda psu_windows_entrypoints"
).split()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("direct", "coverage"), required=True)
    args = parser.parse_args(argv)
    steps = [
        ["pytest", "-q", "test/test_psu_direct_comparison.py", "test/test_psu_submission_comparison.py"],
        ["unittest", *(f"test.test_{name}" for name in DIRECT_UNITTESTS)],
    ]
    if args.profile == "coverage":
        steps.append([
            "pytest", "-q", "test/test_psu_coverage_comparison.py",
            "test/test_psu_coverage_evidence.py", "test/test_psu_coverage_runner.py",
        ])
    environment = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    for step in steps:
        result = subprocess.run(
            [sys.executable, "-B", "-X", "utf8", "-m", *step],
            cwd=ROOT, env=environment, check=False,
        )
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
