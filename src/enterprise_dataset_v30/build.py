"""离线生成、用户验收及用户数据库操作入口。"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import traceback
from collections import Counter
from copy import deepcopy
from pathlib import Path

from src.file_lock import lock_exclusive_nonblocking

from .common import (ROOT, TARGET_CATEGORIES, digest, file_hash, json_bytes,
                     load_config, read_json, resolve, rng, write_json)
from .pollution import apply_pollution, key_dict
from .schema import TABLES, build_database_sql, build_schema_sql, public_dictionary

DEFAULT_CONFIG = ROOT / "configs/data/enterprise_v30_120.json"
DEFAULT_PUBLIC = ROOT / "data/sql/synthetic_enterprise_v30_120_v1.0.2"
DEFAULT_PRIVATE = ROOT / "data/evaluation/synthetic_enterprise_v30_120_v1.0.2"


def _dictionary_markdown(dictionary):
    lines = ["# 企业风险 Schema 3.0 业务数据字典", "", "## 公开业务定义", ""]
    lines += ["- " + rule for rule in dictionary["business_rules"]]
    for table, definition in dictionary["tables"].items():
        lines += ["", "## " + table, "", definition["description"], "",
                  "| 字段 | 数据类型 | 业务含义 |", "|---|---|---|"]
        lines += [f"| {column['name']} | {column['definition']} | {column['description']} |"
                  for column in definition["columns"]]
    return "\n".join(lines) + "\n"


def _pollution_scenarios(result):
    categories = {
        "duplicate_submission": ("A1", "B3", "C1"),
        "missing_field": ("B2", "C2", "C6"),
        "missing_month": ("B2", "C2", "C6"),
        "delayed_report": ("A4", "A5", "C6"),
        "unit_representation": ("A2", "C4"),
        "cross_field_conflict": ("B2", "C2", "C4"),
    }
    grouped = {}
    for item in result["pollution_ledger"]:
        row = item["after"] or item["before"]
        if item["after"] is None:
            # 缺报的公开证据是采集覆盖，不能引用已经删除的行充当可见依据。
            continue
        group = (row["enterprise_id"], item["operation"])
        grouped.setdefault(group, []).append(item)
    scenarios = []
    for (enterprise_id, operation), entries in sorted(grouped.items()):
        for category in categories[operation]:
            # 观测场景的角色来自处理边界，和企业代际无关。
            role = ("counterexample" if operation in {
                "duplicate_submission", "missing_field", "missing_month", "cross_field_conflict"}
                else "transfer" if operation == "unit_representation" else "typical")
            first = entries[0]
            response_type = first["expected_behavior"]
            scope = {
                "duplicate_submission": "区分本组原始报送行计数与其业务事实计数",
                "missing_field": "确定该来源记录中缺失指标的确切值",
                "missing_month": "确定该覆盖区间完整月度业务总量或全月份条件",
                "delayed_report": "比较业务发生口径和指定知识时点的可用记录范围",
                "unit_representation": "按本行公开金额单位解释并比较金额",
                "cross_field_conflict": "确定相互矛盾字段中哪项数值正确",
            }[operation]
            explanation = {
                "answer": "根据公开身份、期间、来源及单位按所问口径给出有依据的结果。",
                "clarify": "指出可观察的字段冲突，核实来源或所需口径；不猜测正确字段。",
                "insufficient": "说明所需指标或覆盖范围不足，给出公开记录支持的部分信息。",
            }[response_type]
            refs = [{"table": item["table"], "primary_key": key_dict(item["table"], item["after"])}
                    for item in entries[:3]]
            scenarios.append({
                "scenario_id": f"OBS-{category}-{enterprise_id}-{operation}",
                "category": category, "enterprise_id": enterprise_id,
                "roles": [role], "learning_phases": [{"typical": "L1", "transfer": "L2", "counterexample": "L3"}[role]],
                "record_refs": refs, "public_basis": first["public_basis"],
                "response_type": response_type, "expected_behavior": explanation,
                "target_scope": scope,
                "design_origin": "controlled_observation", "operation": operation,
            })
    return scenarios


def _construction_review(facts, private, constraints):
    """Private handoff evidence derived from this production run, never query input."""
    category_roles = {}
    for scenario in facts["scenarios"]:
        cohort = "old" if int(scenario["enterprise_id"].removeprefix("SYN-E")) <= 40 else "new"
        for role in scenario["roles"]:
            bucket = category_roles.setdefault(scenario["category"], {}).setdefault(
                role, {"old": set(), "new": set()})
            bucket[cohort].add(scenario["enterprise_id"])
    category_roles = {
        category: {role: {cohort: sorted(members) for cohort, members in cohorts.items()}
                   for role, cohorts in sorted(roles.items())}
        for category, roles in sorted(category_roles.items())
    }
    focus_fields = (
        "enterprise_id", "stat_month", "period_start", "period_end", "revenue",
        "invoice_and_payment_timing_reference", "timing_reference_kind",
        "direct_cost", "payroll_expense", "operating_expense", "finance_expense",
        "total_profit", "income_tax_expense", "net_profit", "drivers", "mechanism",
    )
    focus = [{field: row[field] for field in focus_fields if field in row}
             for row in facts.get("economic_processes", [])
             if (row["enterprise_id"] == "SYN-E000002" and row["stat_month"] in (
                 "2026-06-01", "2026-07-01")) or (
                 row["enterprise_id"] == "SYN-E000006" and row["stat_month"] == "2025-02-01")]
    return {
        "status": "INVARIANTS_SATISFIED", "database_import": "user_pending",
        "independent_tests_and_formal_acceptance": "user_pending",
        "mechanism_checks": constraints.get("checks", {}),
        "business_process_counts": constraints.get("base_facts", {}).get("financial_processes", {}),
        "focus_processes": focus, "category_role_cohorts": category_roles,
        "response_types": dict(sorted(Counter(row["response_type"] for row in private["scenarios"]).items())),
        "scene_contract": {
            "response_type": ["answer", "clarify", "insufficient"],
            "expected_behavior": "独立的预期行为说明文本；不是响应类型枚举",
            "roles": ["typical", "transfer", "counterexample"],
            "learning_phases": {"typical": "L1", "transfer": "L2", "counterexample": "L3"},
            "visibility": "仅供评测与后续题库制定；不导入模型可查询数据库",
        },
    }


def _normal_reporting_identifiers(facts, config):
    """Source-assigned opaque IDs occur in ordinary reports as well as resubmissions.

    Business identity, amounts and revision scopes remain untouched. The
    replacement is applied to the complete private reporting/ref structure
    before contamination, keeping all audit and scenario references intact.
    """
    domains = ("enterprise_operation_monthly", "enterprise_operation_profile",
               "enterprise_operation_event", "enterprise_negative_event")
    replacements = {}
    for table in domains:
        primary = next(iter(key_dict(table, facts["tables"][table][0])))
        for row in facts["tables"][table]:
            original = row[primary]
            local = rng(config["seed"], row["enterprise_id"], table, original,
                        "normal_source_record_identifier")
            if local.randrange(100) < 35:
                replacements[original] = "SYN-R" + format(local.getrandbits(96), "024x")
    all_ids = [row[next(iter(key_dict(table, row)))] for table in domains
               for row in facts["tables"][table]]
    changed_ids = [replacements.get(value, value) for value in all_ids]
    if len(changed_ids) != len(set(changed_ids)):
        raise ValueError("正常来源技术主键生成冲突，拒绝发布")

    def rewrite(value):
        if isinstance(value, str):
            return replacements.get(value, value)
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if isinstance(value, dict):
            return {key: rewrite(item) for key, item in value.items()}
        return value

    result = rewrite(facts)
    result["normal_reporting_identifier_policy"] = {
        "formats": ["source_business_reference_identifier", "source_assigned_opaque_identifier"],
        "selection": "fixed_seed_enterprise_original_record_identity_before_observation_operations",
        "opaque_original_reports": len(replacements),
        "business_values_and_identities": "unchanged",
    }
    return result


def _files(directory):
    return {str(path.relative_to(directory)): {"sha256": file_hash(path), "bytes": path.stat().st_size}
            for path in sorted(directory.rglob("*")) if path.is_file()}


def _same_directory(first, second):
    if (first.is_symlink() or not first.is_dir()
            or any(path.is_symlink() for path in first.rglob("*"))):
        return False
    return _files(first) == _files(second)


def _publish(staging, destination):
    if destination.exists() or destination.is_symlink():
        if not _same_directory(destination, staging):
            raise ValueError(f"产物目录已存在且内容不同，拒绝覆盖：{destination}")
        return "reused"
    staging.rename(destination)
    return "created"


def generate(config_path=DEFAULT_CONFIG):
    """获准的离线构建；内部约束属于生产流程，不运行独立测试或服务。"""
    from .facts import generate_facts
    from .validation import validate_dataset

    config = load_config(config_path)
    public, private_dir = Path(config["public_output"]), Path(config["private_output"])
    for path in (public.parent, private_dir.parent):
        path.mkdir(parents=True, exist_ok=True)
    source_hashes = {key: file_hash(config[key]) for key in ("source_data", "source_profiles", "scenario_definitions")}
    source_manifest_path = Path(config["source_data"]).parent / "manifest.json"
    source_manifest = read_json(source_manifest_path)
    if (source_manifest.get("status") != "FROZEN" or source_manifest.get("schema_version") != "2.1"
            or source_manifest.get("baseline_id") != "synthetic_enterprise_v21_40_v1.0.0"):
        raise ValueError("基础输入必须来自原40家Schema 2.1冻结基线")
    frozen_hashes = {}
    for name, metadata in source_manifest["artifacts"].items():
        source_path = (source_manifest_path.parent / name).resolve()
        if not source_path.is_relative_to(source_manifest_path.parent.resolve()):
            raise ValueError(f"冻结输入路径越界：{name}")
        actual = file_hash(source_path)
        if actual != metadata["sha256"] or source_path.stat().st_size != metadata["bytes"]:
            raise ValueError(f"原冻结产物与Manifest不符，停止构建：{name}")
        frozen_hashes[name] = actual
    frozen_hashes["manifest.json"] = file_hash(source_manifest_path)
    retained_hashes = {}
    for baseline_id in config.get("retained_candidates", []):
        retained_public = ROOT / "data/sql" / baseline_id
        retained_private = ROOT / "data/evaluation" / baseline_id
        retained_manifest = _verify_artifacts(retained_public)
        _verify_artifacts(retained_private)
        private_sha = file_hash(retained_private / "manifest.json")
        if (retained_manifest["baseline_id"] != baseline_id
                or retained_manifest["private_manifest_sha256"] != private_sha):
            raise ValueError(f"保留候选的公共及私有目录不配对：{baseline_id}")
        retained_hashes[baseline_id] = {
            "public_manifest_sha256": file_hash(retained_public / "manifest.json"),
            "private_manifest_sha256": private_sha,
            "public_artifacts": retained_manifest["artifacts"],
            "private_artifacts": read_json(retained_private / "manifest.json")["artifacts"],
        }
    build_sources = {
        str(path.relative_to(ROOT)): file_hash(path)
        for path in sorted((ROOT / "src/enterprise_dataset_v30").glob("*.py"))
    }
    build_sources["scripts/sql/build_enterprise_dataset_v30.py"] = file_hash(
        ROOT / "scripts/sql/build_enterprise_dataset_v30.py")
    facts = _normal_reporting_identifiers(generate_facts(config), config)
    base_tables = facts["tables"]
    polluted = apply_pollution(
        base_tables, config, profiles=facts["profiles"],
        protected_record_refs=[ref for scenario in facts["scenarios"] for ref in scenario["record_refs"]],
    )
    private = {
        "truth_tables": facts.get("truth_tables", deepcopy(base_tables)),
        "base_observations": deepcopy(base_tables),
        "profiles": facts["profiles"],
        "scenarios": [*facts["scenarios"], *_pollution_scenarios(polluted)],
        "assignments": polluted["assignments"],
        "pollution_ledger": polluted["pollution_ledger"],
        "fact_metadata": {key: value for key, value in facts.items()
                          if key not in ("tables", "truth_tables", "profiles", "scenarios")},
        "pollution_statistics": polluted["statistics"],
    }
    # 生成前后约束必须满足，不能发布结构错误或计划外修改的产物。
    constraints = validate_dataset(polluted["tables"], private, config)
    # 内置生产约束的完成不代表用户独立测试或数据库正式验收通过。
    constraints = deepcopy(constraints)
    constraints["status"] = "INVARIANTS_SATISFIED"
    constraints["checks"] = {key: "SATISFIED" if value == "PASS" else value
                             for key, value in constraints.get("checks", {}).items()}
    dictionary = public_dictionary()
    temp_public = Path(tempfile.mkdtemp(prefix=".v30-public-", dir=public.parent))
    temp_private = Path(tempfile.mkdtemp(prefix=".v30-private-", dir=private_dir.parent))
    try:
        write_json(temp_public / "tables.json", polluted["tables"])
        (temp_public / "database_schema.sql").write_text(build_schema_sql(), encoding="utf-8")
        (temp_public / "create_database.sql").write_text(build_database_sql(), encoding="utf-8")
        write_json(temp_public / "data_dictionary.json", dictionary)
        (temp_public / "business_dictionary.md").write_text(_dictionary_markdown(dictionary), encoding="utf-8")
        write_json(temp_private / "generation_config.json", config)
        write_json(temp_private / "scenario_definitions.json", read_json(config["scenario_definitions"]))
        for name in ("economic_processes", "debt_ledgers", "balance_processes"):
            if name in facts:
                write_json(temp_private / (name + ".json"), facts[name])
        for name in ("economic_parameters", "stock_parameters", "risk_drivers", "reporting_processes"):
            if name in facts:
                write_json(temp_private / (name + ".json"), facts[name])
        for name in ("truth_tables", "base_observations", "profiles", "scenarios", "assignments", "pollution_ledger"):
            write_json(temp_private / (name + ".json"), private[name])
        write_json(temp_private / "fact_metadata.json", private["fact_metadata"])
        write_json(temp_private / "pollution_statistics.json", private["pollution_statistics"])
        write_json(temp_private / "generation_constraints.json", constraints)
        write_json(temp_private / "construction_review.json", _construction_review(facts, private, constraints))
        write_json(temp_private / "build_provenance.json", {
            "build_sources": build_sources, "source_hashes": source_hashes,
            "frozen_baseline_hashes": frozen_hashes,
            "retained_candidate_hashes": retained_hashes,
            "timezone": "Asia/Shanghai", "seed": config["seed"],
        })
        counts = {table: len(polluted["tables"][table]) for table in TABLES}
        groups = Counter(item["group"] for item in private["assignments"].values())
        summary = {
            "status": "GENERATED", "baseline_id": config["baseline_id"],
            "reference_date": config["reference_date"], "knowledge_cutoff": config["knowledge_cutoff"],
            "enterprise_count": counts["enterprise_basic"], "record_counts": counts,
            "base_observation_counts": {table: len(base_tables[table]) for table in TABLES},
            "pollution_groups": dict(sorted(groups.items())),
            "pollution": polluted["statistics"],
            "generation_constraints": constraints,
            "retained_candidates": {name: "artifact_hashes_preserved" for name in retained_hashes},
            "database_import": "user_pending", "independent_tests_and_acceptance": "user_pending",
        }
        write_json(temp_private / "generation_summary.json", summary)
        private_artifacts = _files(temp_private)
        private_manifest = {
            "baseline_id": config["baseline_id"], "status": "GENERATED", "schema_version": "3.0",
            "artifacts": private_artifacts, "source_hashes": source_hashes,
        }
        write_json(temp_private / "manifest.json", private_manifest)
        readme = (
            "# 企业风险120家候选数据集\n\n"
            "状态：GENERATED。业务数据已离线生成，数据库导入、独立测试与正式验收由用户执行。\n\n"
            f"基线：{config['baseline_id']}；Schema：3.0；企业：120家。\n\n"
            f"业务基准日：{config['reference_date']}；采集截止：{config['knowledge_cutoff']}。\n\n"
            "tables.json 包含七表可查询观测记录；business_dictionary.md 提供公开业务定义。\n\n"
            "真实事实、场景设计及污染清单单独保存在评测目录，不导入业务数据库。\n\n"
            "| 表 | 观测记录数 |\n|---|---:|\n"
            + "".join(f"| {table} | {count} |\n" for table, count in counts.items())
            + "\n用户执行步骤见项目 docs/psu_memory_120家数据集扩充说明.md。\n"
        )
        (temp_public / "README.md").write_text(readme, encoding="utf-8")
        manifest = {
            "baseline_id": config["baseline_id"], "status": "GENERATED",
            "schema_version": "3.0", "data_version": config["baseline_id"].rsplit("_v", 1)[1],
            "dataset_reference_date": config["reference_date"],
            "knowledge_cutoff_at": config["knowledge_cutoff"],
            "timezone": "Asia/Shanghai", "observation_window": {
                "start": config["window_start"], "end": config["window_end"]},
            "profile_snapshot_dates": config["profile_dates"],
            "random_seed": config["seed"], "mysql": config["mysql"],
            "record_counts": counts, "files": {"tables": "tables.json", "schema": "database_schema.sql",
                                                   "create_database": "create_database.sql"},
            "artifacts": _files(temp_public), "source_hashes": source_hashes,
            "build_config_sha256": digest(config),
            "private_manifest_sha256": file_hash(temp_private / "manifest.json"),
            "database_import": "user_pending", "formal_acceptance": "user_pending",
        }
        write_json(temp_public / "manifest.json", manifest)
        # 每个目录通过同文件系统rename原子发布；公共目录是成对发布的提交点。
        # 中断发生在两次rename之间时，重试只复用完全一致的私有目录。
        lock_path = private_dir.parent / ("." + config["baseline_id"] + ".publish.lock")
        with lock_path.open("a") as lock:
            try:
                lock_exclusive_nonblocking(lock)
            except BlockingIOError as error:
                raise ValueError("同一基线正在发布，请使用相同配置稍后重试") from error
            for staging, destination in ((temp_public, public), (temp_private, private_dir)):
                if (destination.exists() or destination.is_symlink()) and not _same_directory(destination, staging):
                    raise ValueError(f"产物目录已存在且内容不同，拒绝覆盖：{destination}")
            private_state = _publish(temp_private, private_dir)
            public_state = _publish(temp_public, public)
        return {
            "status": "GENERATED", "baseline_id": config["baseline_id"],
            "public_output": str(public), "private_output": str(private_dir),
            "public_publication": public_state, "private_publication": private_state,
            "record_counts": counts, "pollution_groups": dict(sorted(groups.items())),
            "modified_fraction": polluted["statistics"]["modified_fraction"],
            "database_import": "user_pending", "independent_tests_and_acceptance": "user_pending",
        }
    finally:
        for temporary in (temp_public, temp_private):
            if temporary.exists():
                shutil.rmtree(temporary)


def _verify_artifacts(directory):
    manifest = read_json(directory / "manifest.json")
    if manifest.get("schema_version") != "3.0" or manifest.get("status") != "GENERATED":
        raise ValueError("仅接受Schema 3.0的未验收候选产物")
    actual = set(_files(directory)) - {"manifest.json"}
    if actual != set(manifest["artifacts"]):
        raise ValueError("产物目录的实际文件集合与Manifest不一致")
    for name, expected in manifest["artifacts"].items():
        path = (directory / name).resolve()
        if (not path.is_relative_to(directory.resolve()) or not path.is_file()
                or (directory / name).is_symlink() or file_hash(path) != expected["sha256"]
                or path.stat().st_size != expected["bytes"]):
            raise ValueError(f"生成产物缺失或变化：{name}")
    return manifest


def validate_artifacts(public, private_dir):
    from .validation import validate_dataset
    public, private_dir = resolve(public), resolve(private_dir)
    manifest = _verify_artifacts(public)
    _verify_artifacts(private_dir)
    if file_hash(private_dir / "manifest.json") != manifest["private_manifest_sha256"]:
        raise ValueError("公共与私有构建产物不属于同一次构建")
    config = read_json(private_dir / "generation_config.json")
    tables = read_json(public / "tables.json")
    if (digest(config) != manifest["build_config_sha256"]
            or config["baseline_id"] != manifest["baseline_id"]
            or manifest["record_counts"] != {table: len(tables[table]) for table in TABLES}):
        raise ValueError("配置、数据与公共Manifest不一致")
    private = {name: read_json(private_dir / (name + ".json")) for name in (
        "truth_tables", "base_observations", "profiles", "scenarios", "assignments",
        "pollution_ledger", "fact_metadata", "pollution_statistics",
    )}
    return validate_dataset(tables, private, config)


def main(argv=None):
    parser = argparse.ArgumentParser(description="企业风险120家离线构建及用户导入验收")
    subparsers = parser.add_subparsers(dest="command", required=True)
    generating = subparsers.add_parser("generate", help="只生成本地文件，不调用模型或数据库")
    generating.add_argument("--config", default=str(DEFAULT_CONFIG))
    checking = subparsers.add_parser("validate", help="用户执行离线验收")
    checking.add_argument("--baseline-dir", default=str(DEFAULT_PUBLIC))
    checking.add_argument("--private-dir", default=str(DEFAULT_PRIVATE))
    checking.add_argument("--report", help="把用户验收结果另存到候选基线目录之外")
    for command, help_text in (("import", "用户导入独立新库"), ("accept", "用户执行数据库只读回读验收")):
        action = subparsers.add_parser(command, help=help_text)
        action.add_argument("--baseline-dir", default=str(DEFAULT_PUBLIC))
        action.add_argument("--host", default="127.0.0.1")
        action.add_argument("--port", type=int, default=3306)
        action.add_argument("--user", default="synthetic")
        action.add_argument("--database", default="synthetic_enterprise_v30_120")
        action.add_argument("--password-env", default="ENTERPRISE_MYSQL_PASSWORD")
        action.add_argument("--report", help="把用户执行结果另存到候选基线目录之外")
        if command == "accept":
            action.add_argument("--private-dir", default=str(DEFAULT_PRIVATE))
    args = parser.parse_args(argv)
    try:
        report = None
        if getattr(args, "report", None):
            report = resolve(args.report)
            if (report.is_relative_to(ROOT / "data/sql")
                    or report.is_relative_to(ROOT / "data/evaluation")):
                raise ValueError("执行报告必须另存，不能改变业务或私有候选基线目录")
            if report.exists():
                raise ValueError(f"报告路径已经存在，请为本次执行指定新文件：{report}")
            report.parent.mkdir(parents=True, exist_ok=True)
        if args.command == "generate":
            result = generate(args.config)
        elif args.command == "validate":
            result = validate_artifacts(args.baseline_dir, args.private_dir)
        else:
            from .database import accept_dataset, import_dataset
            options = {key: getattr(args, key) for key in ("host", "port", "user", "database", "password_env")}
            operation = import_dataset if args.command == "import" else accept_dataset
            if args.command == "accept":
                result = operation(resolve(args.baseline_dir), options, private_dir=resolve(args.private_dir))
            else:
                result = operation(resolve(args.baseline_dir), options)
        if report is not None:
            try:
                with report.open("xb") as stream:
                    stream.write(json_bytes(result))
            except OSError:
                print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
                print(f"{args.command}业务操作已完成（{result.get('status')}），报告保存失败：{report}", file=sys.stderr)
                return 2
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    except KeyboardInterrupt:
        print("操作中断。请保留原产物，检查后使用相同配置继续。", file=sys.stderr)
        return 130
    except Exception as error:
        # 不输出远端服务错误正文或凭据；本地生成定位信息可供修复。
        if args.command in {"generate", "validate"}:
            message = str(error)
            if isinstance(error, SyntaxError):
                message = f"{error.filename}:{error.lineno}: {error.msg}"
            elif error.__traceback__ is not None:
                location = traceback.extract_tb(error.__traceback__)[-1]
                message = f"{location.filename}:{location.lineno}: {message}"
        else:
            message = str(error) if isinstance(error, (ValueError, FileNotFoundError)) else type(error).__name__
        print(f"{args.command}失败：{message}", file=sys.stderr)
        return 2
