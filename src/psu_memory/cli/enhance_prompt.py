"""本地提示词增强：保存最终文本与完整报告。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from src.project_paths import resolve_project_path
from ..models import RetrievalRequest, RetrievalResult, UsageMetrics
from ..retrieval.retrieval_models import RetrievalConfig
from ..retrieval.service import PSURetriever


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="PSU 本地提示词增强")
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--base-prompt", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output-prompt", required=True, type=Path)
    parser.add_argument(
        "--retrieval-config",
        type=Path,
        default=Path(
            "src/psu_memory/retrieval/retrieval_config.example.json"
        ),
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/business/query.yaml"),
    )
    args = parser.parse_args(argv)

    store_path = resolve_project_path(args.store)
    request_path = resolve_project_path(args.request)
    base_path = resolve_project_path(args.base_prompt)
    config_path = resolve_project_path(args.retrieval_config)
    model_path = resolve_project_path(args.model_config)
    report_path = resolve_project_path(args.report)
    output_path = resolve_project_path(args.output_prompt)

    protected = {
        store_path,
        Path(str(store_path) + "-wal"),
        Path(str(store_path) + "-shm"),
        Path(str(store_path) + "-journal"),
        request_path,
        base_path,
        config_path,
        model_path,
    }

    if (
        report_path == output_path
        or report_path in output_path.parents
        or output_path in report_path.parents
    ):
        parser.error("两个输出路径不能相同或互为父目录")

    for path in (report_path, output_path):
        if path in protected:
            parser.error("输出不能覆盖数据库、输入或配置")
        if path.exists():
            parser.error("输出已经存在，请指定新路径")

    started = perf_counter()

    try:
        # newline="" 保留原文件中的 CRLF、LF 和末尾空白。
        with base_path.open(
            "r", encoding="utf-8", newline=""
        ) as stream:
            original_prompt = stream.read()
    except Exception as error:
        print(json.dumps({
            "status": "failed",
            "error_type": type(error).__name__,
            "reason": "无法读取原提示词。",
        }, ensure_ascii=False, indent=2))
        return 2

    component_started = False
    try:
        payload = json.loads(
            request_path.read_text(encoding="utf-8")
        )
        if not isinstance(payload, dict):
            raise ValueError("请求必须是 JSON 对象")

        request = RetrievalRequest(**payload)
        config = RetrievalConfig.from_json(config_path)

        with PSURetriever.from_local(
            store_path,
            retrieval_config=config,
            model_config_path=model_path,
        ) as service:
            component_started = True
            enhanced_prompt, result = service.enhance_prompt(
                original_prompt,
                request,
            )

    except Exception as error:
        enhanced_prompt = original_prompt
        result = RetrievalResult(
            reason="本地增强命令处理失败，保留原提示词。",
            reason_code="command_failed",
            usage=(
                UsageMetrics()
                if component_started
                else UsageMetrics(
                    elapsed_seconds=0.0,
                    model_call_count=0,
                    prompt_tokens=0,
                    completion_tokens=0,
                    embedding_input_tokens=0,
                )
            ),
            diagnostics={
                "error_type": type(error).__name__,
                "augmentation": {
                    "applied": False,
                    "reason_code": "command_failed",
                    "added_chars": 0,
                },
            },
        )

    augmentation = result.diagnostics.get("augmentation", {})
    applied = augmentation.get("applied") is True
    normal_empty_codes = {
        "disabled",
        "no_selection",
        "no_members",
        "no_encodable_members",
        "no_candidates",
        "context_budget_exceeded",
    }
    status = (
        "enhanced" if applied
        else "unchanged"
        if result.reason_code in normal_empty_codes
        else "unavailable"
    )

    report = {
        "status": status,
        "inputs": {
            "store": str(store_path),
            "request": str(request_path),
            "base_prompt": str(base_path),
            "retrieval_config": str(config_path),
            "model_config": str(model_path),
        },
        "original_prompt": original_prompt,
        "enhanced_prompt": enhanced_prompt,
        "retrieval": result.to_dict(),
        "output_prompt_file": str(output_path),
        "pipeline_elapsed_seconds": perf_counter() - started,
    }

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open(
            "x", encoding="utf-8", newline=""
        ) as stream:
            stream.write(enhanced_prompt)

        with report_path.open("x", encoding="utf-8") as stream:
            json.dump(
                report,
                stream,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            stream.write("\n")

    except Exception as error:
        print(json.dumps({
            "status": "failed",
            "error_type": type(error).__name__,
            "reason": "输出保存未完成，请检查已生成文件。",
        }, ensure_ascii=False, indent=2))
        return 2

    print(json.dumps({
        "status": status,
        "reason_code": result.reason_code,
        "reason": result.reason,
        "added_chars": augmentation.get("added_chars", 0),
        "usage": result.to_dict()["usage"],
        "cost_summary": result.diagnostics.get("cost_summary"),
        "output_prompt_file": str(output_path),
        "report_file": str(report_path),
    }, ensure_ascii=False, indent=2))
    return 0 if status in {"enhanced", "unchanged"} else 2


if __name__ == "__main__":
    raise SystemExit(main())