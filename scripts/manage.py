#!/usr/bin/env python3
"""安装、数据准备和维护命令的统一入口。"""
from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Only import the selected command: help and input installation need no model runtime.
COMMANDS = {
    "embedding": ("src.cli.embedding", "准备或检查 embedding 模型"),
    "windows-check": ("src.cli.windows_runtime", "检查 Windows/CUDA 环境"),
    "install-inputs": ("src.cli.install_inputs", "安装已校验的实验输入包"),
    "data": ("src.enterprise_dataset_v30.build", "生成、核验、导入或验收数据"),
    "query": ("src.cli.query", "普通交互或单次业务查询"),
    "check-query": ("src.cli.business_runtime", "检查业务查询启动条件"),
    "package-inputs": ("src.cli.package_inputs", "打包固定实验输入"),
    "verify-reference": ("src.cli.verify_reference", "核验当前题库参考结果"),
    "form-experiences": ("src.cli.form_experiences", "执行预编写 SQL 形成经验"),
    "initialize-seeds": ("src.cli.initialize_seeds", "初始化共同种子库"),
    "review-seeds": (
        "scripts.enterprise_query_bench.apply_seed_difficulty_review",
        "应用已确认的种子难度复核",
    ),
    "rescore": ("src.psu_memory.cli.rescore_values", "从历史 JSON 生成新的重评报告"),
}


def main(argv: list[str] | None = None) -> int | None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="命令说明：\n" + "\n".join(
            f"  {name:20} {description}" for name, (_, description) in COMMANDS.items()
        ) + "\n\n使用 <命令> --help 查看对应参数。",
    )
    parser.add_argument("command", choices=tuple(COMMANDS))
    parser.add_argument("arguments", nargs=argparse.REMAINDER, metavar="ARG")
    args = parser.parse_args(argv)

    previous_argv = sys.argv
    sys.argv = [f"{parser.prog} {args.command}", *args.arguments]
    try:
        module = import_module(COMMANDS[args.command][0])
        return module.main()
    finally:
        sys.argv = previous_argv


if __name__ == "__main__":
    raise SystemExit(main())
