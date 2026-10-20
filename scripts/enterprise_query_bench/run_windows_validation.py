"""Native Windows validation entry point; use the selected Conda interpreter."""
from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
PROFILES = {
    "coverage": ("PSU_COVERAGE", "coverage_v3", "natural-v030-coverage-v3", 10),
    "direct": ("PSU_DIRECT", "direct_v1", "natural-v030-seeded-dev-v0.1", 10),
    "baseline3": ("PSU_BASELINE3", "baseline3_v1", "natural-v030-baseline3-dev-v0.1", 3),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--run-dir")
    parser.add_argument("--config")
    parser.add_argument("--case-limit", type=int)
    args = parser.parse_args(argv)
    prefix, config_name, run_name, default_limit = PROFILES[args.profile]
    environment = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    try:
        case_limit = args.case_limit
        if case_limit is None:
            case_limit = (3 if args.profile == "baseline3" else
                          int(environment.get(f"{prefix}_CASE_LIMIT") or default_limit))
        if case_limit < 1 or (args.profile == "baseline3" and case_limit != 3):
            raise ValueError("case-limit must be positive; baseline3 always uses 3")
    except ValueError as error:
        parser.error(str(error))

    run_dir = args.run_dir or environment.get(f"{prefix}_RUN")
    if not run_dir:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        run_dir = f"data/derived/psu_memory/validation/{run_name}-windows-cuda-{stamp}"
    # abspath preserves the final symlink so lexists also rejects dangling links.
    run_path = Path(os.path.abspath(ROOT / run_dir))
    if os.path.lexists(run_path):
        print(f"Run path already exists; choose a new directory: {run_path}", file=sys.stderr)
        return 1
    environment[f"{prefix}_RUN"] = str(run_path)
    config = args.config or (
        f"configs/validation/enterprise_v30_120_v1_0_2_natural_{config_name}_windows_cuda.json"
    )
    config_path = Path(os.path.abspath(ROOT / config))
    print(f"Run directory: {run_path}", flush=True)
    print(f"Profile: {args.profile}; cases per condition: {case_limit}", flush=True)
    command = [sys.executable, "-B", "-X", "utf8", "-m", "src.psu_memory.cli.validate_local"]
    steps = [
        ["prepare", "--config", str(config_path), "--run-dir", str(run_path),
         "--case-limit", str(case_limit)],
    ]
    modes = ("baseline",) if args.profile == "baseline3" else ("baseline", "fifo", "adaptive")
    steps.extend(["run", "--run-dir", str(run_path), "--mode", mode] for mode in modes)
    steps.append(["summarize", "--run-dir", str(run_path)])
    for step in steps:
        try:
            result = subprocess.run(command + step, cwd=ROOT, env=environment, check=False)
        except OSError as error:
            print(f"Could not start validation: {error}", file=sys.stderr)
            return 1
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
