"""纯规则层：不读文件、不访问数据库、不调用模型。"""
from __future__ import annotations

import hashlib
import json

from .quality_models import QualityDecision, digest


def operations(record):
    value = json.loads(record["task"].get("evidence_text") or "[]")
    if not isinstance(value, list) or any(not isinstance(o, dict) for o in value):
        raise ValueError("evidence_text 必须是操作对象列表")
    return value


def original_content(record, profile):
    task = record["task"]
    queries = [o for o in operations(record) if o.get("kind") == "sql"]
    ids = {e.get("details", {}).get("legacy_b1", {}).get("query_id")
           for e in profile.get("evidence", [])
           if e.get("signals", {}).get("reference_result_match") is True}
    matched = [o for o in queries if o.get("query_id") and o.get("query_id") in ids]
    # 新任务可能分多条 SQL 完成；独立审核必须明确绑定到被审核的那条。
    reviews = [e.get("details", {}).get("independent_review", {})
               for e in profile.get("evidence", [])]
    reviewed_ids = {r.get("query_id") for r in reviews
                    if r.get("verdict") == "pass" and r.get("scope") == "sql"}
    reviewed = [o for o in queries if o.get("query_id")
                and o.get("query_id") in reviewed_ids and o.get("success") is True]
    successful = [o for o in queries if o.get("success") is True]
    selected = (matched or reviewed or successful or queries or [{}])[-1]
    return {
        "execution_id": task["task_id"], "kind": "original", "parent_version": None,
        "question": task["question"], "context_text": task.get("context_text", ""),
        "context_available": "context_text" in task,
        "sql": selected.get("sql", ""), "operation": selected,
        "notes": "", "applicability": "", "error_reason": "",
        "confirmed_by_user": False, "source_sha256": record["source_sha256"],
    }


def _review_matches(review, content, content_version):
    if review.get("scope") not in ("sql", "sql_pattern"):
        return False
    operation = content.get("operation", {})
    return bool(
        (review.get("content_version") and review["content_version"] == content_version)
        or (review.get("query_id") and review["query_id"] == operation.get("query_id"))
        or (review.get("sql") and review["sql"].strip() == content["sql"].strip())
        or (review.get("sql_sha256") and review["sql_sha256"] ==
            hashlib.sha256(content["sql"].strip().encode()).hexdigest())
    )


def _verified_sql_result_review(review, operation, record):
    return bool(
        review.get("admission_policy") in ("sql_result_only_v1", "sql_result_direct_v1")
        and review.get("scope") == "sql" and review.get("verdict") == "pass"
        and operation.get("query_id")
        and review.get("query_id") == operation.get("query_id")
        and operation.get("success") is True and operation.get("is_complete") is True
        and len([candidate for candidate in operations(record)
                 if candidate.get("kind") == "sql"
                 and candidate.get("query_id") == operation.get("query_id")]) == 1
    )


def evaluate_candidate(record, profile, content, *, source, evidence_version=None, feedback=()):
    """评价指定内容版本。历史负面反馈只影响它指向的版本。"""
    task = record["task"]
    version = digest(content)
    corrected = content["kind"] == "correction"  # 是否被修正过
    events = [e for e in feedback if e["target_version"] == version]
    reasons, basis = [], []
    correctness = None
    reference_ok = review_ok = False
    operation = content.get("operation", {})
    # 当前实验只评价已实际执行的SQL结果。外部结果审核绑定唯一查询后，
    # 最终回答状态及回答自评不再否定该SQL；原始任务状态仍保留供审计。
    sql_result_approved = any(
        _verified_sql_result_review(review, operation, record)
        for item in profile.get("evidence", [])
        for review in [item.get("details", {}).get("independent_review", {})]
    )
    visible = dict(content)
    for event in events:
        for field in ("notes", "applicability", "error_reason"):
            if event.get(field):
                visible[field] = event[field]

    if source["status"] == "conflict":
        reasons.extend(source.get("conflicts") or ["source_conflict"])
    if not content.get("question", "").strip() or not content.get("sql", "").strip():
        reasons.append("query_content_missing")
    negative = any(e["rating"] in ("problem", "correction") for e in events)
    if negative:
        reasons.append("negative_feedback")
        correctness = False

    if not corrected:
        # 验收与历史反馈属于原内容，不自动传播到修正版。
        if task["status"] != "completed" and not sql_result_approved:
            reasons.append("execution_not_completed")
        if task.get("evaluation_verdict") == "revise" and not sql_result_approved:
            reasons.append("internal_revision_required")
        if operation.get("success") is False or operation.get("is_complete") is False:
            reasons.append("query_execution_incomplete")
        if profile.get("user_feedback") == "problem":
            reasons.append("negative_feedback")
            correctness = False
        conflicts = profile.get("signal_conflicts", {})
        if conflicts:
            reasons.append("evaluation_conflict")
        if "problem" in conflicts.get("user_feedback", []):
            reasons.append("negative_feedback")
            correctness = False
        if profile.get("reference_result_match") is False:
            reasons.append("reference_mismatch")
            correctness = False

        positive_ids = set()
        for item in profile.get("evidence", []):
            if item.get("signals", {}).get("reference_result_match") is True:
                positive_ids.add(item.get("details", {}).get("legacy_b1", {}).get("query_id"))
            review = item.get("details", {}).get("independent_review", {})
            if _review_matches(review, content, version):
                if review.get("verdict") == "pass":
                    if (review.get("admission_policy") in ("sql_result_only_v1", "sql_result_direct_v1")
                            and not _verified_sql_result_review(review, operation, record)):
                        reasons.append("sql_result_review_unverified")
                    else:
                        review_ok = True
                elif review.get("verdict") in ("fail", "revise", "rejected"):
                    reasons.append("independent_review_failed")
                    correctness = False

        if profile.get("reference_result_match") is True:
            qid = operation.get("query_id")
            matching_operations = [o for o in operations(record) if qid and o.get("query_id") == qid]
            reference_ok = (
                positive_ids == {qid} and qid is not None and len(matching_operations) == 1
                and operation.get("success") is True and operation.get("is_complete") is True
            )
            if not reference_ok:
                reasons.append("reference_query_unverified")
        if reference_ok:
            basis.append("reference_result_match")
        if review_ok:
            basis.append("scoped_independent_review")
        helpful = profile.get("user_feedback") == "helpful" or any(e["rating"] == "helpful" for e in events)
        if task.get("evaluation_verdict") == "pass" and helpful:
            basis.append("internal_pass_and_positive_feedback")
    elif content.get("confirmed_by_user"):
        basis.append("user_confirmed_correction")

    if not basis:
        reasons.append("admission_basis_missing")
    if correctness is None and (reference_ok or review_ok or "user_confirmed_correction" in basis):
        correctness = True
    level = 3 if reference_ok and review_ok else (2 if (
        reference_ok or review_ok or "user_confirmed_correction" in basis
    ) else (1 if basis else 0))
    scope = operation.get("scope", {})
    completeness = {
        "execution_details": bool(operation.get("success") is True
                                  and operation.get("is_complete") is True
                                  and "columns" in operation and "rows" in operation),
        "query_scope": bool(visible.get("applicability") or
                            (isinstance(scope, dict) and scope.get("enterprise") and scope.get("time_range"))),
        "context": content.get("context_available") is True,
    }
    preview = "\n".join([
        "问题：" + content.get("question", ""),
        "必要上下文：" + (content.get("context_text") or "无"),
        "SQL：\n" + content.get("sql", ""),
        "注意事项：" + (visible.get("notes") or "未提供"),
        "适用范围：" + (visible.get("applicability") or "以当前问题及数据结构为准"),
    ])
    return QualityDecision(
        execution_id=task["task_id"], content_version=version,
        admission_state="suspended" if reasons else "eligible",
        effective_correctness=correctness, reasons=sorted(set(reasons)), basis=basis,
        validation_level=level, completeness=completeness, preview_text=preview,
        source=source, evidence_version=evidence_version, content=visible,
        historical_usage=task.get("usage", {}),
        feedback_event_ids=[e["event_id"] for e in events],
        execution_verified=bool(not corrected and operation.get("success") is True
                                and operation.get("is_complete") is True),
    )
