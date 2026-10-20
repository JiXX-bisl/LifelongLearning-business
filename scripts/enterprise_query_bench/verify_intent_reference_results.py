#!/usr/bin/env python3
"""由用户运行：以一次只读数据库快照核验所选题库的全部参考结果。"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.file_lock import lock_exclusive_nonblocking


DATA_VERSION = "synthetic_enterprise_v30_120_v1.0.2"
DATABASE = "synthetic_enterprise_v30_120"
DATASET_MANIFEST = ROOT / "data/sql" / DATA_VERSION / "manifest.json"
EXCLUDED = {"A2-L1-02", "A2-L3-02"}
PUBLIC_CASE_KEYS = {
    "id", "question", "history", "session", "turn", "category",
    "learning_stage", "difficulty",
}


@dataclass(frozen=True)
class Bank:
    name: str
    version: str
    case_count: int
    reference_count: int

    @property
    def public(self):
        return ROOT / "data/questions/psu_memory" / self.name

    @property
    def private(self):
        return ROOT / "data/evaluation/psu_memory" / self.name

    @property
    def report(self):
        return self.private / "reference_validation.json"


BANKS = {
    "v0.2.2": Bank("enterprise_v30_120_v1_0_2_natural_v0_2_2_reviewed",
                   "psu_enterprise_v30_120_natural_v0.2.2", 72, 70),
    "v0.3.0": Bank("enterprise_v30_120_v1_0_2_natural_v0_3_0_expanded",
                   "psu_enterprise_v30_120_natural_v0.3.0", 134, 132),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def encode(document):
    return (json.dumps(document, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def load_bundle(compare_results, bank):
    """只检查本版本的文件、题号和结果结构，不执行旧意图裁判。"""
    snapshots = {}

    def read(path):
        content = path.read_bytes()
        if path in snapshots:
            require(snapshots[path] == content, "读取过程中输入发生变化：" + str(path))
        snapshots[path] = content
        return json.loads(content)

    manifest = read(bank.private / "bundle_manifest.json")
    required_artifacts = {
        bank.public / "questions.json", bank.public / "business_context.json",
        bank.private / "ground_truth.json", bank.private / "intent_contracts.json",
    }
    mapped_paths = {}
    for field in ("artifact_sha256", "source_sha256"):
        mapping = manifest.get(field)
        require(isinstance(mapping, dict) and mapping, field + " 缺失")
        mapped_paths[field] = set()
        for relative, expected_hash in mapping.items():
            require(isinstance(relative, str), "哈希路径必须为字符串")
            path = (ROOT / relative).resolve()
            require(not Path(relative).is_absolute() and path.is_relative_to(ROOT),
                    "哈希路径必须位于项目内：" + relative)
            content = path.read_bytes()
            require(sha256(content) == expected_hash, "文件哈希不一致：" + relative)
            if path in snapshots:
                require(snapshots[path] == content, "读取过程中输入发生变化：" + relative)
            snapshots[path] = content
            mapped_paths[field].add(path)
    require(required_artifacts.issubset(mapped_paths["artifact_sha256"]),
            "产物哈希未覆盖四个题库文件")
    require(DATASET_MANIFEST in mapped_paths["source_sha256"], "源哈希未绑定数据集 manifest")
    questions = read(bank.public / "questions.json")
    context = read(bank.public / "business_context.json")
    truth = read(bank.private / "ground_truth.json")
    contracts = read(bank.private / "intent_contracts.json")
    for document in (questions, truth, contracts, manifest):
        require(document.get("version") == bank.version, "题库版本不匹配")
        require(document.get("data_version") == DATA_VERSION, "题库数据版本不匹配")
        require(document.get("schema_version") == "3.0", "题库 Schema 不匹配")
    require(context.get("data_version") == DATA_VERSION, "公开背景数据版本不匹配")
    require(contracts.get("intent_contract_version") == "user_intent_direct_v2",
            "意图依据版本不匹配")
    require(manifest.get("reference_validation") in ("user_pending", "user_verified"),
            "参考核验状态无效")
    case_lists = [questions.get("cases"), truth.get("cases"), contracts.get("cases")]
    require(all(isinstance(cases, list) and len(cases) == bank.case_count for cases in case_lists),
            f"公开题、参考依据及意图依据均须有 {bank.case_count} 题")
    id_sets = []
    for cases in case_lists:
        require(all(isinstance(case, dict) and isinstance(case.get("id"), str)
                    and case["id"] for case in cases), "题号格式无效")
        ids = {case["id"] for case in cases}
        require(len(ids) == bank.case_count, "题号重复")
        id_sets.append(ids)
    require(id_sets[0] == id_sets[1] == id_sets[2], "三个文件的题号不一致")
    for case in questions["cases"]:
        require(set(case).issubset(PUBLIC_CASE_KEYS), "公开题含非公开字段：" + case["id"])
        require(isinstance(case.get("question"), str) and case["question"].strip(),
                "问题原文缺失：" + case["id"])
        require(isinstance(case.get("history"), list)
                and all(isinstance(item, dict) and set(item) == {"role", "content"}
                        and item["role"] == "user" and isinstance(item["content"], str)
                        for item in case["history"]),
                "公开历史只能保存用户问题文本：" + case["id"])
    references = []
    for case in truth["cases"]:
        sql, expected = case.get("reference_sql"), case.get("expected_result")
        if case["id"] in EXCLUDED:
            require(case.get("response_type") == "clarify", "排除题必须为澄清题：" + case["id"])
            continue
        require(isinstance(sql, str) and sql.strip(), "参考 SQL 缺失：" + case["id"])
        require(compare_results(expected, expected), "标准结果结构无效：" + case["id"])
        references.append(case)
    require(EXCLUDED.issubset(id_sets[0]) and len(references) == bank.reference_count,
            f"应有 {bank.reference_count} 份参考 SQL，排除两道澄清题")
    dataset = read(DATASET_MANIFEST)
    require(dataset.get("baseline_id") == DATA_VERSION and dataset.get("schema_version") == "3.0",
            "数据集版本不匹配")
    counts = dataset.get("record_counts")
    require(isinstance(counts, dict) and len(counts) == 7
            and all(isinstance(name, str) and type(count) is int and count >= 0
                    for name, count in counts.items()), "数据集表记录数无效")
    return truth, manifest, references, counts, snapshots


def database_check(references, counts, args, compare_results):
    from src.chatflow_agent.database import BusinessMySQLReadOnlyEnvironment

    checks = []
    environment = None
    stop_reason = None
    stop_diagnostic = {}
    try:
        environment = BusinessMySQLReadOnlyEnvironment(
            database=args.database, host=args.host, port=args.port, user=args.user,
            password_env=args.password_env, max_total_rows=10000,
            fetch_batch_size=500, statement_timeout_ms=20000,
        )
        connection = environment.connection
        connection.start_transaction(isolation_level="REPEATABLE READ",
                                     consistent_snapshot=True, readonly=True)
        environment.verify_record_counts(counts)
        for case in references:
            # 禁止重连：新连接无法延续这一份数据库快照。
            if environment.connection is not connection or not connection.is_connected():
                stop_reason = "connection_lost"
                break
            try:
                actual = environment.execute_query(case["reference_sql"])
                expected = case["expected_result"]
                if actual.get("is_complete") is not True:
                    check = {"id": case["id"], "verdict": "fail", "reason": "incomplete_result"}
                elif actual.get("columns") != expected["columns"]:
                    check = {"id": case["id"], "verdict": "fail", "reason": "columns_mismatch",
                             "actual_columns": actual.get("columns"),
                             "expected_columns": expected["columns"]}
                elif not compare_results(actual, expected):
                    check = {"id": case["id"], "verdict": "fail", "reason": "result_mismatch",
                             "actual_row_count": len(actual.get("rows", [])),
                             "expected_row_count": len(expected["rows"]),
                             "actual_sample": actual.get("rows", [])[:3],
                             "expected_sample": expected["rows"][:3]}
                else:
                    check = {"id": case["id"], "verdict": "pass"}
                checks.append(check)
            except Exception as error:
                checks.append({"id": case["id"], "verdict": "fail", "reason": "query_error",
                               "error_type": type(error).__name__,
                               "mysql_errno": getattr(error, "errno", None),
                               "sqlstate": getattr(error, "sqlstate", None),
                               "error_message": str(error)[:1000]})
            if environment.connection is not connection or not connection.is_connected():
                stop_reason = "snapshot_connection_lost"
                break
    except Exception as error:
        stop_reason = "database_setup_failed"
        stop_diagnostic = {"error_type": type(error).__name__,
                           "mysql_errno": getattr(error, "errno", None)}
    finally:
        if environment is not None:
            try:
                if environment.connection is not None:
                    environment.connection.rollback()
            except Exception:
                pass
            environment.close()
    checked_ids = {check["id"] for check in checks}
    for case in references:
        if case["id"] not in checked_ids:
            checks.append({"id": case["id"], "verdict": "fail",
                           "reason": "not_executed", "cause": stop_reason or "snapshot_stopped",
                           **stop_diagnostic})
    return checks


def ensure_unchanged(snapshots):
    for path, before in snapshots.items():
        require(path.read_bytes() == before, "核验期间文件已变化，拒绝写入核验结论：" + str(path))


def replace_file(path, content):
    """同目录临时文件加原子替换，不留下半写的 JSON。"""
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def approve(truth, manifest, snapshots, bank):
    truth_path = bank.private / "ground_truth.json"
    questions_path = bank.public / "questions.json"
    manifest_path = bank.private / "bundle_manifest.json"
    truth["reviewed_by_user"] = True
    truth["status"] = "USER_REFERENCE_VERIFIED"
    truth_bytes = encode(truth)
    manifest["status"] = "USER_REFERENCE_VERIFIED"
    manifest["reference_validation"] = "user_verified"
    manifest["artifact_sha256"][str(truth_path.relative_to(ROOT))] = sha256(truth_bytes)
    writes = {truth_path: truth_bytes}
    questions = json.loads(snapshots[questions_path])
    if questions.get("reviewed_by_user") is not True:
        questions.update(reviewed_by_user=True, status="USER_ENABLED",
                         review_basis="用户执行--approve确认按已授权替换方案生成的题库可启用；数据库核验不代表新增逐题人工审阅。")
        writes[questions_path] = encode(questions)
        manifest["artifact_sha256"][str(questions_path.relative_to(ROOT))] = sha256(writes[questions_path])
    manifest_bytes = encode(manifest)
    ensure_unchanged(snapshots)
    updated_snapshots = dict(snapshots)
    written = []
    try:
        for path, content in writes.items():
            ensure_unchanged(updated_snapshots)
            replace_file(path, content)
            updated_snapshots[path] = content
            written.append(path)
        ensure_unchanged(updated_snapshots)
        replace_file(manifest_path, manifest_bytes)
    except Exception:
        # 清单写入失败时，只回退仍是本进程内容的已写文件。
        for path in reversed(written):
            if path.read_bytes() == writes[path]:
                replace_file(path, snapshots[path])
        raise
    snapshots.update({**writes, manifest_path: manifest_bytes})
    return sha256(truth_bytes)


def run(args, bank):
    sys.path.insert(0, str(ROOT))
    from src.psu_memory.validation.comparison import compare_results

    report = {
        "version": bank.version, "data_version": DATA_VERSION, "verdict": "fail",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "database": DATABASE, "readonly": True, "approved": False,
        "excluded_ids": sorted(EXCLUDED), "checks": [],
    }
    snapshots = {}
    previous_report = bank.report.read_bytes() if bank.report.exists() else None
    try:
        truth, manifest, references, counts, snapshots = load_bundle(compare_results, bank)
        truth_hash = sha256(snapshots[bank.private / "ground_truth.json"])
        report.update(ground_truth_sha256_before=truth_hash, ground_truth_sha256_after=truth_hash)
        report["checks"] = database_check(references, counts, args, compare_results)
        passed = sum(check["verdict"] == "pass" for check in report["checks"])
        report.update(pass_count=passed, fail_count=bank.reference_count - passed)
        if passed == bank.reference_count:
            report["verdict"] = "pass"
        ensure_unchanged(snapshots)
        current_report = bank.report.read_bytes() if bank.report.exists() else None
        require(current_report == previous_report, "核验期间已有其他进程写入报告，拒绝覆盖")
        if args.approve and passed == bank.reference_count:
            report["ground_truth_sha256_after"] = approve(truth, manifest, snapshots, bank)
            report["approved"] = True
        ensure_unchanged(snapshots)
        replace_file(bank.report, encode(report))
    except Exception as error:
        # 输入无效或并发变更时不覆盖输入，也不发布可误认作当前结果的旧报告。
        print(json.dumps({"verdict": "fail", "error_type": type(error).__name__,
                          "message": str(error) if isinstance(error, (ValueError, FileNotFoundError))
                          else "核验未完成；请检查文件和依赖。",
                          "approved": report["approved"], "report_written": False}, ensure_ascii=False))
        return 2
    print(json.dumps({"verdict": report["verdict"], "pass": report["pass_count"],
                      "fail": report["fail_count"], "excluded": 2,
                      "approved": report["approved"], "report": str(bank.report)}, ensure_ascii=False))
    return 0 if report["verdict"] == "pass" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", choices=tuple(BANKS), default="v0.2.2",
                        help="v0.2.2核验70题；v0.3.0核验含62道变体在内的132题")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3306)
    parser.add_argument("--user", default="synthetic")
    parser.add_argument("--database", choices=[DATABASE], default=DATABASE)
    parser.add_argument("--password-env", default="ENTERPRISE_MYSQL_PASSWORD")
    parser.add_argument("--approve", action="store_true",
                        help="全部参考SQL通过后，确认所选题库及标准可启用；不修改数据库数据")
    args = parser.parse_args()
    bank = BANKS[args.bank]
    # 锁文件保留在原位，避免删除后其他进程对不同 inode 加锁。
    with (bank.private / ".reference_validation.lock").open("a+b") as lock:
        try:
            lock_exclusive_nonblocking(lock)
        except BlockingIOError:
            print(json.dumps({"verdict": "fail", "reason": "verification_already_running",
                              "approved": False, "report_written": False}, ensure_ascii=False))
            return 2
        return run(args, bank)


if __name__ == "__main__":
    raise SystemExit(main())
