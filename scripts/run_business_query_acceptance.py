from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path
from typing import Any

from copy import deepcopy
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chatflow_agent.query import QueryService
from src.chatflow_agent.runtime import BusinessRuntimeError
from src.chatflow_agent.strategy import (
    EXPERIENCE_BOUNDARY,
    GSC_SELECTION_PROMPT,
)

from src.lifelong_learning.tasks.enterprise_query_bench.dataset import (
    RowOrder,
)
from src.lifelong_learning.tasks.enterprise_query_bench.evaluator import (
    compare_query_results,
)


DATASET_DIR = (
    PROJECT_ROOT / 'data/questions/enterprise_query_bench/enterprise_query_bench_dev_v0.1.1'
)


def load_acceptance_cases(
    dataset_dir: Path = DATASET_DIR,
) -> tuple[dict, list[dict], dict[int, dict]]:
    """分开保存待提问任务和验收答案，避免整条题库记录进入Agent。"""
    manifest = json.loads(
        (dataset_dir / "manifest.json").read_text(encoding="utf-8")
    )
    entries = json.loads(
        (dataset_dir / manifest["active_agent_entry"]).read_text(
            encoding="utf-8"
        )
    )

    if set(entries) != {str(index) for index in range(40)}:
        raise ValueError("验收题库必须完整包含0—39编号。")

    tasks = []
    oracles = {}

    for index in range(40):
        entry = entries[str(index)]
        tasks.append({
            "sample_index": index,
            "task_id": entry["task_id"],
            "instruction": entry["instruction"],
        })

        answer = entry["answer_info"]
        oracles[index] = {
            "expected_rows": answer["direct"],
            "row_order": RowOrder(
                answer["comparison_policy"]["row_order"]
            ),
        }

    if len({task["task_id"] for task in tasks}) != 40:
        raise ValueError("验收题库存在重复task_id。")

    return manifest, tasks, oracles


def evaluate_b1(
    result: dict[str, Any],
    oracle: dict[str, Any],
) -> dict[str, Any]:
    """只比较最后一次成功SQL结果，不评价自然语言或任务终态。"""
    last_success = next(
        (
            operation
            for operation in reversed(result.get("operations", []))
            if operation.get("kind") == "sql"
            and operation.get("success") is True
        ),
        None,
    )

    if last_success is None:
        return {
            "matched": False,
            "query_id": None,
            "reason": "no_successful_sql",
        }

    matched = compare_query_results(
        actual=last_success["rows"],
        expected=oracle["expected_rows"],
        row_order=oracle["row_order"],
    )
    return {
        "matched": matched,
        "query_id": last_success["query_id"],
        "reason": "matched" if matched else "result_mismatch",
    }

def save_acceptance(path: Path, payload: dict[str, Any]) -> None:
    """只保存验收索引，不复制完整会话轨迹。"""
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_acceptance_batch(
    service: QueryService,
    manifest: dict,
    tasks: list[dict],
    oracles: dict[int, dict],
) -> Path:
    """顺序执行任务；业务经验准入不受B1或人工复核干预。"""
    directory = Path(service.config["output_dir"]).resolve()

    expected_baseline = manifest["database_baseline"]["baseline_id"]
    if service.config["data"]["baseline_id"] != expected_baseline:
        raise BusinessRuntimeError("业务基线与验收题库不一致。")

    # Standard也检查已有会话，避免重复运行覆盖同一验收空间。
    if service.experience_store.version != 0 or (
        directory.exists() and any(directory.iterdir())
    ):
        raise BusinessRuntimeError(
            "验收空间非空，请使用新的experiment-id；不会覆盖或续跑。"
        )

    directory.mkdir(parents=True, exist_ok=True)
    acceptance_path = directory / "acceptance.json"

    payload = {
        "dataset_id": manifest["dataset_id"],
        "batch_status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        # Q02返回的解析配置只含环境变量名称，不含凭据值。
        "settings": deepcopy(service.config),
        "prompt_snapshot": {
            "decision": (
                service.system_prompt.rstrip()
                + "\n\n"
                + EXPERIENCE_BOUNDARY
            ),
            "gsc_selection": (
                GSC_SELECTION_PROMPT
                if service.learning["mode"] == "gsc"
                else None
            ),
        },
        "expected_task_count": len(tasks),
        "finished_task_count": 0,
        "results": [],
        # 独立复核结束后再形成正式汇总。
        "summary": None,
        "abort": None,
    }
    save_acceptance(acceptance_path, payload)

    for position, task in enumerate(tasks, start=1):
        index = task["sample_index"]
        print(
            f"[{position}/{len(tasks)}] "
            f"编号={index}，任务={task['task_id']}，开始查询……",
            flush=True,
        )

        try:
            # 清空对话，但保留本模式的跨题学习经验。
            service.reset()

            # 只传问题字符串，禁止传整条任务或GT。
            result = service.query(task["instruction"])

            # query返回时，会话保存和经验处理已经完成。
            b1 = evaluate_b1(result, oracles[index])
            item = {
                "sample_index": index,
                "task_id": task["task_id"],
                "record_path": result["record_path"],
                "status": result["status"],
                "b1": b1,
                "memory_version": result["memory_version"],
                "used_experience_ids": list(
                    result["used_experience_ids"]
                ),
                "metrics": deepcopy(result.get("metrics", {})),
                "independent_review": {
                    "verdict": "pending",
                    "reason": None,
                },
                # 独立复核时填写；B1不一致不直接等于业务答错。
                "failure_classification": None,
            }

        except (Exception, KeyboardInterrupt) as error:
            payload["batch_status"] = "aborted"
            payload["finished_at"] = (
                datetime.now(timezone.utc).isoformat()
            )
            payload["abort"] = {
                "sample_index": index,
                "task_id": task["task_id"],
                # 不写入可能包含敏感内容的原始异常文本。
                "error_type": type(error).__name__,
            }
            save_acceptance(acceptance_path, payload)
            raise

        # failed、limit_reached也是正常返回的任务，照实记录并继续。
        payload["results"].append(item)
        payload["finished_task_count"] = len(payload["results"])
        save_acceptance(acceptance_path, payload)

        print(
            f"  状态={item['status']}，"
            f"B1={b1['matched']}",
            flush=True,
        )

    payload["batch_status"] = "completed"
    payload["finished_at"] = datetime.now(timezone.utc).isoformat()
    save_acceptance(acceptance_path, payload)
    return acceptance_path

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="阶段一40题业务验收；每次执行一种模式"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/business/query.yaml",
        help="业务配置文件",
    )
    parser.add_argument(
        "--mode",
        choices=("standard", "psu", "gsc"),
        required=True,
        help="本次验收模式，必须明确指定",
    )
    parser.add_argument(
        "--experiment-id",
        default="stage1_batch_v1",
        help="实验标识；目标模式空间必须为空",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        manifest, tasks, oracles = load_acceptance_cases()

        with QueryService(
            args.config,
            mode=args.mode,
            experiment_id=args.experiment_id,
        ) as service:
            if service.config["model"]["model_name"] != "kimi-k2.6":
                raise BusinessRuntimeError(
                    "本轮验收固定使用kimi-k2.6，请检查业务配置。"
                )

            learning = service.learning
            if (
                learning["utilized_sample_count"] != 4
                or learning["group_count"] != 2
                or learning["sample_count_per_group"] != 2
            ):
                raise BusinessRuntimeError(
                    "本轮验收固定使用PSU-4、GSC-2×2，请检查学习配置。"
                )

            print(
                f"模式：{learning['mode']}\n"
                f"实验：{learning['experiment_id']}\n"
                f"题库：{manifest['dataset_id']}\n"
                f"输出：{service.config['output_dir']}",
                flush=True,
            )

            path = run_acceptance_batch(
                service, manifest, tasks, oracles
            )

        print(f"\n40题执行登记完成：{path}", flush=True)
        print("独立回答复核尚未完成，本次不生成正式业务正确率。")
        return 0

    except KeyboardInterrupt:
        print(
            "\n运行已中断。保留已有产物，不自动续跑。",
            file=sys.stderr,
        )
        return 130

    except BusinessRuntimeError as error:
        print(f"验收停止：{error}", file=sys.stderr)
        return 1

    except Exception as error:
        # 避免直接输出可能包含凭据或连接信息的原始异常。
        print(
            f"验收停止：{type(error).__name__}。"
            "请检查题库、配置和输出文件；已有产物不自动删除。",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
