"""题库构建阶段的既有静态契约；不参与直接SQL评分。"""
RESPONSE_TYPES = ("answer", "clarify", "insufficient")
LEGACY_RESPONSE_TYPES = {"query": "answer", "clarification": "clarify"}
RESPONSE_KINDS = {"answer": "query", "clarify": "clarification", "insufficient": "insufficient"}
SQL_REQUIREMENTS = ("required", "optional", "not_applicable")
NATURAL_INTENT_VERSION = "natural_intent_v1"

def bind_case_contracts(cases, rules):
    """仅在验证端附加响应及SQL契约；不改写公开问题清单。"""
    ids = {case["id"] for case in cases}
    expected = rules.get("expected_types", {})
    requirements = rules.get("sql_requirements", {})
    intent_version = rules.get("intent_contract_version")
    intent_rubrics = rules.get("intent_rubrics", {})
    if intent_version is not None:
        if intent_version != NATURAL_INTENT_VERSION:
            raise ValueError("未知意图审核契约版本")
        if not isinstance(intent_rubrics, dict) or set(intent_rubrics) != ids:
            raise ValueError("intent_rubrics 必须完整覆盖全部题号")
    if not isinstance(expected, dict) or (expected and set(expected) != ids):
        raise ValueError("expected_types 必须完整覆盖全部题号")
    if not isinstance(requirements, dict) or set(requirements) - ids:
        raise ValueError("sql_requirements 必须使用当前题号")
    result = []
    for case in cases:
        kind = case.get("kind")
        legacy = LEGACY_RESPONSE_TYPES.get(kind, kind if kind in RESPONSE_TYPES else None)
        response_type = expected.get(case["id"], legacy)
        if response_type not in RESPONSE_TYPES:
            raise ValueError(f"缺少或无效的私有响应类型：{case['id']}")
        if legacy is not None and legacy != response_type:
            raise ValueError(f"旧kind与私有响应类型冲突：{case['id']}")
        requirement = requirements.get(case["id"], {
            "answer": "required", "clarify": "not_applicable", "insufficient": "optional",
        }[response_type])
        if requirement not in SQL_REQUIREMENTS:
            raise ValueError(f"无效SQL执行契约：{case['id']}")
        result.append({**case, "kind": RESPONSE_KINDS[response_type],
                       "expected_response_type": response_type, "sql_requirement": requirement,
                       "response_contract_version": ("three_way_v1" if expected or kind in RESPONSE_TYPES
                                                     else "legacy"),
                       **({"intent_contract_version": intent_version} if intent_version else {})})
    return result


def validate_intent_contracts(document, cases, *, data_version=None):
    """文件完整性检查不等于用户认可隐含目标；执行门槛另行检查。"""
    if document.get("intent_contract_version") != NATURAL_INTENT_VERSION:
        raise ValueError("意图合同协议必须为natural_intent_v1")
    if data_version is not None and document.get("data_version") != data_version:
        raise ValueError("意图合同与数据版本不同")
    contracts = document.get("cases")
    ids = {case["id"] for case in cases}
    if (not isinstance(contracts, list) or len(contracts) != len(ids)
            or {item.get("id") for item in contracts if isinstance(item, dict)} != ids):
        raise ValueError("意图合同必须完整且唯一对应全部提问")
    if not isinstance(document.get("pending_reviews"), list):
        raise ValueError("意图合同必须显式记录pending_reviews")
    target_ids = {}
    unresolved = set()
    for contract in contracts:
        case_id = contract["id"]
        targets = contract.get("required_business_targets")
        if not isinstance(targets, list) or not targets:
            raise ValueError("每题必须列出业务目标及意图还原依据：" + case_id)
        target_ids[case_id] = set()
        for target in targets:
            if not isinstance(target, dict):
                raise ValueError("业务目标必须为结构化对象：" + case_id)
            target_id = target.get("target_id")
            if not isinstance(target_id, str) or not target_id.strip() or target_id in target_ids[case_id]:
                raise ValueError("每题业务目标target_id必须非空且唯一：" + case_id)
            target_ids[case_id].add(target_id)
            if (not isinstance(target.get("description"), str) or not target["description"].strip()
                    or type(target.get("implicit")) is not bool):
                raise ValueError("业务目标须描述目的并标注implicit：" + case_id)
            bases = target.get("restoration_basis")
            if not isinstance(bases, list) or not bases:
                raise ValueError("业务目标缺少还原依据：" + case_id + ":" + target_id)
            for basis in bases:
                if (not isinstance(basis, dict)
                        or basis.get("kind") not in ("question_wording", "public_dictionary", "public_records",
                                                     "conversation_context", "business_convention")
                        or any(not isinstance(basis.get(key), str) or not basis[key].strip()
                               for key in ("text", "source"))
                        or type(basis.get("observable")) is not bool):
                    raise ValueError("意图还原依据须记录类型、说明、来源及模型可见性：" + case_id)
            alignment = target.get("alignment_status")
            if alignment not in ("supported", "convention_needs_review", "scope_conflict"):
                raise ValueError("未知意图还原依据状态：" + case_id)
            if alignment == "supported" and not any(basis["observable"] for basis in bases):
                raise ValueError("supported业务目标至少需要一项模型可见的还原依据：" + case_id + ":" + target_id)
            review_status = target.get("review_status")
            if review_status not in (None, "APPROVED"):
                raise ValueError("未知业务目标人工审核状态：" + case_id + ":" + target_id)
            approved = review_status == "APPROVED"
            if approved:
                resolution = target.get("review_resolution")
                if (document.get("reviewed_by_user") is not True
                        or contract.get("basis_reviewed_by_user") is not True
                        or not isinstance(resolution, dict)
                        or any(not isinstance(resolution.get(key), str)
                               or not resolution[key].strip() for key in ("reason", "source"))):
                    raise ValueError("已通过业务目标必须记录用户审核、理由及来源：" + case_id + ":" + target_id)
            # 人工审核是独立决议，不把原先的业务习惯或范围诊断改写为supported。
            if alignment != "supported" and not approved:
                unresolved.add((case_id, target_id))
    pending_ids = set()
    for pending in document["pending_reviews"]:
        if (not isinstance(pending, dict) or pending.get("id") not in ids
                or pending.get("target_id") not in target_ids[pending["id"]]):
            raise ValueError("待确认隐含依据含未知题号")
        pair = (pending["id"], pending["target_id"])
        if pair in pending_ids:
            raise ValueError("待确认隐含依据不能重复")
        pending_ids.add(pair)
    if not unresolved.issubset(pending_ids):
        raise ValueError("未支持或冲突的业务目标必须列入pending_reviews")
    return document


