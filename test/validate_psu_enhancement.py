"""提示词增强综合验收：回归、真实案例、零调用 CLI 与数据保护。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "test"))

# 复用上一阶段验证工具，不再定义相同的汇总函数。
from validate_psu_selector import (
    cost_totals,
    run_regression,
    timing_summary,
    vector_rows,
    write_json,
)
from test_psu_active_members import database_snapshot

from src.project_paths import resolve_project_path
from src.psu_memory.models import RetrievalRequest
from src.psu_memory.retrieval.augmentation import PROMPT_SEPARATOR
from src.psu_memory.retrieval.retrieval_models import RetrievalConfig
from src.psu_memory.retrieval.selector_llm import LocalSelectorClient
from src.psu_memory.retrieval.service import PSURetriever


def copy_database(source, target):
    with sqlite3.connect(
        source.as_uri() + "?mode=ro", uri=True
    ) as connection:
        with sqlite3.connect(target) as destination:
            connection.backup(destination)


def write_text(path, text):
    with path.open("x", encoding="utf-8", newline="") as stream:
        stream.write(text)


def base_prompt(request):
    """本地模拟上游增强内容；本轮不调用 Vanna 或生成 SQL。"""
    return (
        "【上游增强提示词本地样例】\r\n"
        "当前问题：" + request.question + "\r\n"
        "数据版本：" + request.data_version + "\r\n"
        "必要对话上下文：" + request.context_text + "\r\n"
        "数据获取建议：" + request.data_advice + "\r\n"
        "后续生成 SQL 时，以当前需求、数据结构和查询口径为准。\n"
        "历史资料只提供方法参考。  \t"
    )


def run_zero_call_cli(
    *,
    folder,
    name,
    db,
    request,
    original,
    config_path,
    expected_code,
    expected_exit,
):
    request_path = folder / (name + "-request.json")
    base_path = folder / (name + "-base.txt")
    output_path = folder / (name + "-final.txt")
    report_path = folder / (name + "-report.json")

    write_json(request_path, asdict(request))
    write_text(base_path, original)
    before = database_snapshot(db)

    # 这两个案例必须在模型初始化之前结束。
    # 使用不存在的模型配置和空 Key，防止意外进入真实择优。
    environment = dict(os.environ)
    environment["OPENAI_API_KEY"] = ""

    started = perf_counter()
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "src.psu_memory.cli.enhance_prompt",
            "--store", str(db),
            "--request", str(request_path),
            "--base-prompt", str(base_path),
            "--retrieval-config", str(config_path),
            "--model-config", str(folder / "unused-model-config.yaml"),
            "--output-prompt", str(output_path),
            "--report", str(report_path),
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
    )
    wall_seconds = perf_counter() - started

    write_text(folder / (name + "-stdout.txt"), completed.stdout)
    write_text(folder / (name + "-stderr.txt"), completed.stderr)

    if not report_path.is_file() or not output_path.is_file():
        return {
            "passed": False,
            "exit_code": completed.returncode,
            "reason": "CLI 未生成预期文件",
            "process_wall_seconds": wall_seconds,
        }

    report = json.loads(report_path.read_text(encoding="utf-8"))
    result = report["retrieval"]
    prefilter = result["diagnostics"].get("prefilter", {})
    embedding_usage = prefilter.get("usage", {})

    checks = {
        "expected_exit_code": completed.returncode == expected_exit,
        "expected_reason_code": result["reason_code"] == expected_code,
        "original_text_preserved": report["enhanced_prompt"] == original,
        "original_file_bytes_preserved": (
            output_path.read_bytes() == base_path.read_bytes()
        ),
        "zero_llm_call": result["usage"]["model_call_count"] == 0,
        "zero_new_encoding": (
            embedding_usage.get("encoded_text_count") == 0
        ),
        "no_tokenizer_load": (
            embedding_usage.get("tokenizer_load_seconds") == 0
        ),
        "no_embedding_model_load": (
            embedding_usage.get("model_load_count") == 0
        ),
        "database_unchanged": database_snapshot(db) == before,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "exit_code": completed.returncode,
        "status": report["status"],
        "cost_summary": result["diagnostics"].get("cost_summary"),
        "process_wall_seconds": wall_seconds,
        "report_file": str(report_path),
    }


def run_real(args, folder, regression):
    source = resolve_project_path(args.store)
    if not source.is_file():
        raise FileNotFoundError(source)

    previous = json.loads(
        (source.parent / "retrieve-validation.json").read_text(
            encoding="utf-8"
        )
    )
    identity = previous["first"]["items"][0]["metadata"][
        "snapshot"
    ]["identity"]
    scope = {
        "namespace": identity["namespace"],
        "data_version": identity["data_version"],
    }

    source_before = database_snapshot(source)
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    db = folder / "history.sqlite3"
    copy_database(source, db)

    config_path = resolve_project_path(args.retrieval_config)
    model_path = resolve_project_path(args.model_config)
    config = RetrievalConfig.from_json(config_path)

    clients = []

    def create_client():
        client = LocalSelectorClient.from_yaml(
            model_path,
            retrieval_config=config,
        )
        clients.append(client)
        return client

    cases = [
        {
            "id": "detail",
            "question": (
                "查一下衡途工业运输有限公司的总资产和总负债，"
                "金额按元返回。"
            ),
            "expected": "enhanced",
        },
        {
            "id": "followup",
            "question": "那北京呢？",
            "context_text": json.dumps([
                {
                    "role": "user",
                    "content": "统计未开公众号且摘要为空的企业数量。",
                }
            ], ensure_ascii=False),
            "expected": "enhanced",
        },
        {
            "id": "unclear_intent",
            "question": "请帮我处理这件事。",
            "expected": "no_selection",
        },
    ]

    records = []
    reuse_report = None
    reused_services = []
    first_runner = None
    first_http_client = None
    batch_started = perf_counter()

    with PSURetriever.from_local(
        db,
        retrieval_config=config,
        client_factory=create_client,
    ) as service:
        db_before = database_snapshot(db)
        vectors_before = vector_rows(service.index.pool_store)
        embedding_service = service.prefilter.embeddings

        first_request = None
        first_original = None
        first_final = None
        first_result = None

        for case in cases:
            print("开始真实案例：", case["id"], flush=True)
            request = RetrievalRequest(
                task_id="enhancement-acceptance-" + case["id"],
                question=case["question"],
                context_text=case.get("context_text", ""),
                **scope,
            )
            original = base_prompt(request)

            started = perf_counter()
            final, result = service.enhance_prompt(original, request)
            wall_seconds = perf_counter() - started

            augmentation = result.diagnostics.get("augmentation", {})
            summary = result.diagnostics.get("cost_summary", {})
            applied = augmentation.get("applied") is True

            checks = {
                "expected_result": result.reason_code == case["expected"],
                "at_most_one_model_call": (
                    summary.get("model_call_count") in (0, 1)
                ),
                "at_most_one_http_request": (
                    summary.get("http_request_count") in (0, 1)
                ),
                "zero_retry": summary.get("retry_count") == 0,
                "zero_extra_model_call_for_augmentation": (
                    augmentation.get("usage", {}).get("model_call_count")
                    == 0
                ),
                "new_text_within_budget": (
                    augmentation.get("added_chars", 0)
                    <= config.max_context_chars
                ),
                "same_embedding_service": (
                    service.prefilter.embeddings is embedding_service
                ),
            }

            if applied:
                selected = result.diagnostics["selection"]["selected"]
                sql = selected["content"]["sql"]
                checks.update({
                    "original_prefix_unchanged": final.startswith(original),
                    "only_reference_block_appended": (
                        final
                        == original + PROMPT_SEPARATOR + result.items[0].content
                    ),
                    "complete_sql_preserved": (
                        "历史 SQL：\n" + sql in result.items[0].content
                    ),
                    "current_version_check_passed": (
                        augmentation["snapshot_check"]["invalid"] == []
                    ),
                    "exact_added_char_count": (
                        len(final) - len(original)
                        == augmentation["added_chars"]
                    ),
                })
            else:
                checks["original_prompt_unchanged"] = final == original

            if clients and summary.get("model_call_count") == 1:
                client = clients[0]
                if first_runner is None:
                    first_runner = client._runner
                    first_http_client = client._model._async_client
                reused_services.append(
                    client._runner is first_runner
                    and client._model._async_client is first_http_client
                    and first_runner is not None
                    and first_http_client is not None
                )

            record = {
                "case": case,
                "checks": checks,
                "passed": all(checks.values()),
                "request": asdict(request),
                "original_prompt": original,
                "enhanced_prompt": final,
                "retrieval": result.to_dict(),
                "wall_seconds": wall_seconds,
            }
            records.append(record)

            write_json(folder / (case["id"] + ".json"), record)
            write_text(folder / (case["id"] + "-base.txt"), original)
            write_text(folder / (case["id"] + "-enhanced.txt"), final)

            if case["id"] == "detail":
                first_request = request
                first_original = original
                first_final = final
                first_result = result

            print(
                case["id"],
                "结果=", result.reason_code,
                "新增字符=", augmentation.get("added_chars", 0),
                "耗时=", round(wall_seconds, 3),
                "检查通过=", record["passed"],
                flush=True,
            )

        # 复用已有结果，只记录本次复核与拼接的增量成本。
        # 补丁也保证即使发生实现回归，验证不会产生第四次模型调用。
        original_result = copy.deepcopy(first_result.to_dict())
        with patch.object(
            service,
            "retrieve",
            side_effect=AssertionError("复用时禁止重新检索"),
        ):
            repeated_text, repeated = service.enhance_prompt(
                first_original,
                first_request,
                retrieval_result=first_result,
            )

        reuse_checks = {
            "same_prompt_output": repeated_text == first_final,
            "source_result_not_modified": (
                first_result.to_dict() == original_result
            ),
            "zero_incremental_model_call": (
                repeated.diagnostics["augmentation"]["usage"][
                    "model_call_count"
                ] == 0
            ),
            "zero_incremental_embedding_tokens": (
                repeated.diagnostics["augmentation"]["usage"][
                    "embedding_input_tokens"
                ] == 0
            ),
            "original_model_cost_not_added_again": (
                repeated.usage.model_call_count
                == first_result.usage.model_call_count
            ),
        }
        reuse_report = {
            "passed": all(reuse_checks.values()),
            "checks": reuse_checks,
            "incremental_usage": (
                repeated.diagnostics["augmentation"]["usage"]
            ),
            "result": repeated.to_dict(),
        }
        write_json(folder / "reuse.json", reuse_report)

        vectors_after = vector_rows(service.index.pool_store)
        db_after = database_snapshot(db)

    real_batch_seconds = perf_counter() - batch_started

    added_keys = set(vectors_after) - set(vectors_before)
    query_keys = {
        record["retrieval"]["diagnostics"].get(
            "prefilter", {}
        ).get("query", {}).get("cache_key")
        for record in records
    }
    query_keys.discard(None)

    preservation = {
        "source_records_unchanged": database_snapshot(source) == source_before,
        "source_file_unchanged": (
            hashlib.sha256(source.read_bytes()).hexdigest() == source_sha
        ),
        "same_tables": set(db_before) == set(db_after),
        "non_vector_tables_unchanged": all(
            db_before[name] == db_after.get(name)
            for name in db_before
            if name != "psu_retrieval_vectors"
        ),
        "vector_schema_unchanged": (
            db_before["psu_retrieval_vectors"]["definition"]
            == db_after["psu_retrieval_vectors"]["definition"]
        ),
        "old_vector_rows_unchanged": all(
            vectors_after.get(key) == row
            for key, row in vectors_before.items()
        ),
        "only_query_vectors_added": added_keys <= query_keys,
    }

    # 两条 CLI 流程不需要模型，独立进程验证真实命令入口。
    disabled_request = replace(
        first_request,
        task_id="enhancement-cli-disabled",
        limit=0,
    )
    cli_disabled = run_zero_call_cli(
        folder=folder,
        name="cli-disabled",
        db=db,
        request=disabled_request,
        original=first_original,
        config_path=config_path,
        expected_code="disabled",
        expected_exit=0,
    )

    # 在专用故障副本中删除索引关联，不影响正常验证库和源库。
    fault_db = folder / "missing-index.sqlite3"
    copy_database(db, fault_db)
    with sqlite3.connect(fault_db) as connection:
        connection.execute("DELETE FROM psu_retrieval_index_members")

    cli_missing_index = run_zero_call_cli(
        folder=folder,
        name="cli-missing-index",
        db=fault_db,
        request=replace(
            first_request,
            task_id="enhancement-cli-missing-index",
        ),
        original=first_original,
        config_path=config_path,
        expected_code="index_not_ready",
        expected_exit=2,
    )

    # 最后再检查源库，以及 CLI 对正常验证库的读取。
    preservation["source_unchanged_after_cli"] = (
        database_snapshot(source) == source_before
        and hashlib.sha256(source.read_bytes()).hexdigest() == source_sha
    )
    preservation["normal_database_unchanged_by_cli"] = (
        database_snapshot(db) == db_after
    )

    case_costs = [
        record["retrieval"]["diagnostics"]["cost_summary"]
        for record in records
    ]
    # reuse.result 保留原检索成本，所以这里只累计 reuse 的增量。
    all_costs = [
        *case_costs,
        reuse_report["incremental_usage"],
    ]
    for cli_result in (cli_disabled, cli_missing_index):
        if cli_result.get("cost_summary") is not None:
            all_costs.append(cli_result["cost_summary"])

    total_cost = cost_totals(all_costs)
    service_checks = {
        "one_client_instance": len(clients) == 1,
        "runner_and_http_client_reused": (
            bool(reused_services) and all(reused_services)
        ),
        "model_call_limit": (
            total_cost["model_call_count"] is not None
            and total_cost["model_call_count"] <= 3
        ),
        "zero_retry": total_cost["retry_count"] == 0,
    }

    passed = (
        regression["passed"]
        and all(record["passed"] for record in records)
        and reuse_report["passed"]
        and cli_disabled["passed"]
        and cli_missing_index["passed"]
        and all(preservation.values())
        and all(service_checks.values())
    )

    report = {
        "status": (
            "automated_checks_passed" if passed else "needs_review"
        ),
        "manual_review_required": True,
        "scope": scope,
        "source_database": str(source),
        "validation_database": str(db),
        "fault_database": str(fault_db),
        "regression": regression,
        "real_case_count": len(records),
        "real_case_pass_count": sum(record["passed"] for record in records),
        "llm_cost": total_cost,
        "new_query_vector_count": len(added_keys),
        "real_case_wall_timing": timing_summary(
            record["wall_seconds"] for record in records
        ),
        "augmentation_timing": timing_summary(
            record["retrieval"]["diagnostics"]["augmentation"][
                "usage"
            ]["elapsed_seconds"]
            for record in records
        ),
        "real_case_batch_seconds": real_batch_seconds,
        "preservation": preservation,
        "service_checks": service_checks,
        "reuse": reuse_report,
        "cli_disabled": cli_disabled,
        "cli_missing_index": cli_missing_index,
        "business_database_call_count": 0,
        "sql_generation_call_count": 0,
        "cases": records,
    }
    write_json(folder / "acceptance.json", report)

    lines = [
        "# 提示词增强验收审阅",
        "",
        "本轮验证检索和提示词增强；未执行后续 SQL 生成或业务查询。",
        "",
        "| 案例 | 结果 | 新增字符 | Token | 总耗时/秒 |",
        "|---|---|---:|---:|---:|",
    ]
    for record in records:
        result = record["retrieval"]
        lines.append(
            f"| {record['case']['id']} "
            f"| {result['reason_code']} "
            f"| {result['diagnostics']['augmentation']['added_chars']} "
            f"| {result['diagnostics']['cost_summary']['total_llm_tokens']} "
            f"| {record['wall_seconds']:.3f} |"
        )

    for record in records:
        result = record["retrieval"]
        selected = result["diagnostics"].get(
            "selection", {}
        ).get("selected")
        model_reason = result["diagnostics"].get(
            "selection", {}
        ).get("reason", "")

        lines.extend([
            "",
            "## " + record["case"]["id"],
            "",
            "当前问题：" + record["request"]["question"],
            "",
            "结果：" + result["reason_code"],
            "",
            "择优理由（仅审阅）：" + model_reason,
            "",
        ])
        if selected:
            lines.extend([
                "历史问题：" + selected["content"]["question"],
                "",
                "```sql",
                selected["content"]["sql"],
                "```",
                "",
            ])
        lines.extend([
            "最终提示词文件：" + record["case"]["id"] + "-enhanced.txt",
            "",
            "检查项：",
            "",
            "```json",
            json.dumps(record["checks"], ensure_ascii=False, indent=2),
            "```",
        ])

    write_text(folder / "review.md", "\n".join(lines) + "\n")

    print(json.dumps({
        key: value for key, value in report.items()
        if key not in ("cases", "reuse")
    }, ensure_ascii=False, indent=2))
    return passed


def main():
    parser = argparse.ArgumentParser(description="PSU 提示词增强综合验收")
    parser.add_argument("--real", action="store_true")
    parser.add_argument(
        "--store",
        type=Path,
        default=Path(
            "data/derived/psu_memory/validation/"
            "retriever-step2-a89cbd34f46245d4980dd7abdc4eedf1/"
            "history.sqlite3"
        ),
    )
    parser.add_argument(
        "--retrieval-config",
        type=Path,
        default=ROOT / (
            "src/psu_memory/retrieval/retrieval_config.example.json"
        ),
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=ROOT / "configs/business/query.yaml",
    )
    args = parser.parse_args()

    folder = (
        ROOT / "data/derived/psu_memory/validation"
        / ("enhancement-acceptance-" + uuid.uuid4().hex)
    )
    folder.mkdir(parents=True)
    print("验收目录：", folder, flush=True)

    regression = run_regression(folder)
    if not regression["passed"]:
        print("模拟回归未通过，未调用真实模型。")
        print("回归日志：", folder / "regression.log")
        return 2

    if not args.real:
        print("模拟验收通过，真实模型调用：0")
        print("回归报告：", folder / "regression.json")
        return 0

    try:
        passed = run_real(args, folder, regression)
    except Exception as error:
        write_json(folder / "execution-error.json", {
            "status": "incomplete",
            "error_type": type(error).__name__,
        })
        print("综合验收未完成，错误类型：", type(error).__name__)
        print("请保留验收目录：", folder)
        return 2

    print("完整报告：", folder / "acceptance.json")
    print("审阅表：", folder / "review.md")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())