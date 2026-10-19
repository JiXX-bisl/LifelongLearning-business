"""全自动SQL直接比较入口，无人工或LLM审核命令。"""
from __future__ import annotations

import argparse
import json
import sys

from ..validation.common import DEFAULT_CONFIG, MODES, load_cases, load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description="SQL直接比较与单题记录")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("list-questions", "prepare"):
        command = commands.add_parser(name)
        command.add_argument("--config", default=DEFAULT_CONFIG)
        if name == "prepare":
            command.add_argument("--run-dir", required=True)
    running = commands.add_parser("run")
    running.add_argument("--run-dir", required=True)
    running.add_argument("--mode", choices=MODES, required=True)
    summary = commands.add_parser("summarize")
    summary.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "list-questions":
            config = load_config(args.config)
            _, cases = load_cases(config["questions"])
            selected = [c for c in cases if c["id"] not in config["exclude_case_ids"]]
            for case in selected:
                print(f"{case['id']} [{case['session']} 第{case['turn']}轮] {case['question']}")
            print(f"SQL测试题数：{len(selected)}")
            return 0
        if args.command == "prepare":
            from ..validation.runner import prepare
            result = prepare(args.config, args.run_dir)
        elif args.command == "run":
            from ..validation.runner import run_condition
            result = run_condition(args.run_dir, args.mode)
        else:
            from ..validation.reporting import summarize
            result = summarize(args.run_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except KeyboardInterrupt:
        print("本轮已中断；请使用新目录和原始经验起点重跑。", file=sys.stderr)
        return 130
    except Exception as error:
        message = str(error) if isinstance(error, (ValueError, FileNotFoundError, FileExistsError)) else type(error).__name__
        print(message, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
