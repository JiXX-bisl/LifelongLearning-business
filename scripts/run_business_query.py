from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chatflow_agent.query import QueryService
from src.chatflow_agent.runtime import BusinessRuntimeError
from src.psu_memory.integration.runtime import (
    DEFAULT_CONFIG,
    create_query_runtime,
)


BUSINESS_CONFIG = PROJECT_ROOT / "configs/business/query.yaml"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="企业数据查询助手")

    parser.add_argument(
        "--config",
        type=Path,
        default=BUSINESS_CONFIG,
        help="旧入口的业务配置；新入口使用 psu-config 中的 business_config",
    )
    parser.add_argument(
        "--psu-config",
        type=Path,
        default=None,
        help="PSU 接入配置；指定后进入通用接入流程",
    )
    parser.add_argument(
        "--mode",
        choices=("standard", "psu", "gsc"),
        default=None,
        help="保留旧入口模式；新接入不使用旧 psu/gsc 模式",
    )
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="业务记录的实验标识",
    )
    parser.add_argument(
        "--psu",
        dest="psu_enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="启用或关闭新 PSU，支持 --psu / --no-psu",
    )
    parser.add_argument(
        "--update",
        dest="update_enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="启用或关闭经验更新，支持 --update / --no-update",
    )
    parser.add_argument(
        "--show-cost",
        action="store_true",
        help="将成本和更新状态输出到 stderr",
    )

    input_mode = parser.add_mutually_exclusive_group(required=True)
    input_mode.add_argument("--question", help="执行一次提问")
    input_mode.add_argument("--interactive", action="store_true")

    args = parser.parse_args(argv)
    if args.question is not None and not args.question.strip():
        parser.error("--question 不能为空")

    return args


def error_text(error):
    if isinstance(
        error,
        (BusinessRuntimeError, ValueError, RuntimeError, FileNotFoundError),
    ):
        return str(error)
    return type(error).__name__


def show_pending(runtime):
    if runtime is None or not runtime.pending_updates:
        return

    print("经验更新尚未完成，请先核对 update.json。", file=sys.stderr)
    for index, folder in enumerate(runtime.pending_updates, start=1):
        print(f"  {index}: {folder}", file=sys.stderr)
    print("交互模式可输入 /retry 或 /retry 序号显式重试。", file=sys.stderr)


def show_cost(runtime=None, result=None, *, cumulative=False):
    if runtime is None:
        payload = {
            "condition": "legacy",
            "business_metrics": (result or {}).get("metrics"),
        }
    elif cumulative:
        payload = {
            "condition": runtime.condition,
            "run_usage": runtime.run_usage,
        }
    else:
        update = runtime.last_update or {}
        payload = {
            "condition": runtime.condition,
            "last_usage": runtime.last_usage,
            "update": {
                key: update[key]
                for key in (
                    "status",
                    "stored",
                    "admission_state",
                    "admitted",
                    "replayed",
                    "retry_required",
                )
                if key in update
            },
        }

    print(
        json.dumps(payload, ensure_ascii=False, indent=2),
        file=sys.stderr,
    )


def display_result(
    result,
    *,
    interactive=False,
    runtime=None,
    show_cost_enabled=False,
):
    # 关键输出 C：最终业务内容只从 answer 取得。
    answer = result["answer"]
    print(f"\n助手：{answer}" if interactive else answer)

    # 更新状态、成本与业务回答分开输出。
    show_pending(runtime)
    if show_cost_enabled:
        show_cost(runtime, result)

    failed = result["status"] != "completed"
    update_failed = runtime is not None and bool(runtime.pending_updates)
    return 1 if failed or update_failed else 0


def retry_pending(runtime, selection):
    if runtime is None:
        raise ValueError("/retry 仅用于新接入的更新机制")
    if not runtime.pending_updates:
        raise ValueError("当前没有待重试的更新")

    folder = None
    if selection:
        if not selection.isdecimal():
            raise ValueError("请使用 /retry 序号")
        index = int(selection)
        if not 1 <= index <= len(runtime.pending_updates):
            raise ValueError("更新序号超出范围")
        folder = runtime.pending_updates[index - 1]
    elif len(runtime.pending_updates) != 1:
        raise ValueError("存在多个待更新项目，请使用 /retry 序号")

    report = runtime.retry_update(folder)
    if report["status"] == "completed":
        print("经验更新处理完成。", file=sys.stderr)
        return 0

    print("经验更新仍未完成，请继续核对阶段记录。", file=sys.stderr)
    show_pending(runtime)
    return 1


def run_interactive(service, *, runtime=None, show_cost_enabled=False):
    print("企业查询助手已启动。")
    print("输入问题开始对话；/new 清空对话，/exit 退出。")
    if runtime is not None and runtime.update_enabled:
        print("/retry 显式重试经验更新；多项待更新时使用 /retry 序号。")

    show_pending(runtime)
    exit_code = 0

    while True:
        try:
            question = input("\n你：").strip()
        except EOFError:
            print("\n已退出。")
            return exit_code

        if not question:
            continue
        if question == "/exit":
            print("已退出。")
            return exit_code
        if question == "/new":
            service.reset()
            print("已开始新对话。")
            continue

        command, _, selection = question.partition(" ")
        if command == "/retry":
            try:
                code = retry_pending(runtime, selection.strip())
                exit_code = max(exit_code, code)
                if show_cost_enabled:
                    show_cost(runtime)
            except Exception as error:
                print(f"更新未完成：{error_text(error)}", file=sys.stderr)
                exit_code = 1
            continue

        if runtime is not None and runtime.pending_updates:
            show_pending(runtime)
            exit_code = 1
            continue

        try:
            result = service.query(question)
        except Exception as error:
            print(f"运行失败：{error_text(error)}", file=sys.stderr)
            if show_cost_enabled and runtime is not None:
                show_cost(runtime, cumulative=True)
            exit_code = 1
            continue

        code = display_result(
            result,
            interactive=True,
            runtime=runtime,
            show_cost_enabled=show_cost_enabled,
        )
        exit_code = max(exit_code, code)


def main(argv=None, *, psu_enabled=None, update_enabled=None):
    args = parse_args(argv)

    # 命令行显式开关优先，其次是主函数参数，最后是 JSON 配置。
    effective_psu = (
        args.psu_enabled
        if args.psu_enabled is not None
        else psu_enabled
    )
    effective_update = (
        args.update_enabled
        if args.update_enabled is not None
        else update_enabled
    )

    use_new_entry = (
        args.psu_config is not None
        or effective_psu is not None
        or effective_update is not None
    )

    runtime = None
    try:
        if use_new_entry:
            if args.mode not in (None, "standard"):
                raise ValueError("新接入使用独立开关，不能叠加旧 psu/gsc 模式")
            if args.config.resolve() != BUSINESS_CONFIG.resolve():
                raise ValueError(
                    "新接入请在 psu-config 的 business_config 中指定业务配置"
                )

            # 关键接入 D：主函数统一决定是否使用组件和更新机制。
            runtime = create_query_runtime(
                args.psu_config or PROJECT_ROOT / DEFAULT_CONFIG,
                psu_enabled=effective_psu,
                update_enabled=effective_update,
                experiment_id=args.experiment_id,
            )
            with runtime:
                if args.interactive:
                    return run_interactive(
                        runtime,
                        runtime=runtime,
                        show_cost_enabled=args.show_cost,
                    )

                if runtime.pending_updates:
                    show_pending(runtime)
                    return 1

                result = runtime.query(args.question)
                return display_result(
                    result,
                    runtime=runtime,
                    show_cost_enabled=args.show_cost,
                )

        # 保留原命令入口及其调用接口。
        with QueryService(
            args.config,
            mode=args.mode,
            experiment_id=args.experiment_id,
        ) as service:
            if args.interactive:
                return run_interactive(
                    service,
                    show_cost_enabled=args.show_cost,
                )

            result = service.query(args.question)
            return display_result(
                result,
                show_cost_enabled=args.show_cost,
            )

    except KeyboardInterrupt:
        print("\n已中断并退出。", file=sys.stderr)
        if args.show_cost and runtime is not None:
            show_cost(runtime, cumulative=True)
        return 130
    except Exception as error:
        print(f"运行失败：{error_text(error)}", file=sys.stderr)
        if args.show_cost and runtime is not None:
            show_cost(runtime, cumulative=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())