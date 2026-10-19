"""Private, observable scenario contracts; no questions, SQL or model requests.

Business variants and evaluation roles share one family/category rotation.
Production checks bind each claim to concrete public rows before publication.
"""
from __future__ import annotations

from collections import defaultdict

from .common import TARGET_CATEGORIES, decimal
from .schema import PRIMARY_KEYS

SCENARIO_FAMILIES = (2, 8, 14, 20, 26, 32)
EQUAL_REVENUE_FAMILIES = (1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20,
                         22, 24, 26, 28, 30, 32, 34, 36, 38)
ROLES = ("typical", "transfer", "counterexample")
ROLE_OFFSETS = {"A1": 0, "A2": 1, "A4": 2, "A5": 0, "B2": 1,
                "B3": 2, "C1": 0, "C2": 1, "C4": 2, "C6": 0}
RESPONSE_TYPES = ("answer", "clarify", "insufficient")
ROLE_PHASES = {"typical": "L1", "transfer": "L2", "counterexample": "L3"}


def check_definitions(definitions):
    assignment = definitions.get("role_assignment", {})
    if (tuple(definitions.get("scenario_families", ())) != SCENARIO_FAMILIES or
            tuple(definitions.get("equal_revenue_families", ())) != EQUAL_REVENUE_FAMILIES or
            set(definitions.get("categories", {})) != set(TARGET_CATEGORIES) or
            assignment.get("method") != "family_category_rotation_v1" or
            tuple(assignment.get("ordered_roles", ())) != ROLES or
            assignment.get("category_offsets") != ROLE_OFFSETS or
            definitions.get("learning_phases") != ROLE_PHASES or
            tuple(definitions.get("response_types", ())) != RESPONSE_TYPES):
        raise ValueError("场景契约与已实现的family/category交叉角色及响应类型不一致")
    for category in TARGET_CATEGORIES:
        definition = definitions["categories"][category]
        if not isinstance(definition.get("mechanism"), str) or not definition["mechanism"].strip():
            raise ValueError(f"{category}:缺少可观察业务依据")
        for role in ROLES:
            variant = definition.get(role, {})
            if (variant.get("response_type") not in RESPONSE_TYPES or
                    not isinstance(variant.get("target_scope"), str) or
                    not variant["target_scope"].strip() or
                    not isinstance(variant.get("expected_behavior"), str) or
                    not variant["expected_behavior"].strip()):
                raise ValueError(f"{category}/{role}:场景响应与范围契约不完整")


def variant_for(number, category, definitions=None):
    """The business generator and evidence binder use this same role function."""
    if category not in ROLE_OFFSETS or not 1 <= number <= 120:
        raise ValueError(f"非法场景角色输入:{number}/{category}")
    family, generation = (number - 1) % 40 + 1, (number - 1) // 40
    if family not in SCENARIO_FAMILIES:
        raise ValueError(f"SYN-E{number:06d}/{category}:不属于已配置场景承载画像")
    if definitions is not None:
        check_definitions(definitions)
    position = SCENARIO_FAMILIES.index(family)
    return ROLES[(position + generation + ROLE_OFFSETS[category]) % len(ROLES)]


variant_role = variant_for


def _require(condition, enterprise_id, category, message):
    if not condition:
        raise ValueError(f"{enterprise_id}/{category}:业务场景依据不满足:{message}")


def _refs(bindings):
    result, seen = [], set()
    for table, rows in bindings:
        for row in rows:
            key = tuple(row[field] for field in PRIMARY_KEYS[table])
            identity = (table, key)
            if identity not in seen:
                seen.add(identity)
                result.append({"table": table,
                               "primary_key": {field: row[field] for field in PRIMARY_KEYS[table]}})
    return result


def _latest_business_month(rows, month):
    candidates = [row for row in rows if row["stat_month"] == month]
    groups = {(row["source_system"], row["source_record_ref"], row["stat_month"])
              for row in candidates}
    if len(groups) != 1:
        raise ValueError(f"{month}:场景机制要求单一明确业务身份，实际为{len(groups)}组")
    return max(candidates, key=lambda row: row["report_revision"])


def _role_evidence(category, role, eid, selected, global_people):
    people, monthly = selected["enterprise_person"], selected["enterprise_operation_monthly"]
    if category in {"A1", "C1"}:
        entities, names = defaultdict(list), defaultdict(set)
        for row in people:
            entities[row["person_entity_id"]].append(row)
            names[row["person_name"]].add(row["person_entity_id"])
        if role == "typical":
            _require(any(len({row["assignment_id"] for row in rows}) > 1 for rows in entities.values()),
                     eid, category, "未发现同一自然人的真实多次任职")
            return "one_person_multiple_real_assignments"
        if role == "transfer":
            _require(any(any(row["enterprise_id"] != eid for row in global_people[entity])
                         for entity in entities), eid, category, "未发现有公开双边记录的跨企业自然人")
            return "one_person_observed_in_multiple_enterprises"
        _require(any(len(identities) > 1 for identities in names.values()),
                 eid, category, "未发现同名异人公开身份对")
        return "same_name_distinct_person_entities"
    if category == "A2":
        loans = [row for row in selected["enterprise_operation_event"] if row["event_type"] == "银行借款"]
        _require(loans, eid, category, "缺少借款合同与状态记录")
        history = [row for row in loans if row["business_object_ref"] == loans[0]["business_object_ref"]]
        latest = max(history, key=lambda row: (row["state_effective_at"], row["ingested_at"]))
        _require(any(decimal(row["amount"]) > 0 for row in history), eid, category, "合同本金不为正")
        _require(any(decimal(row["overdue_principal"]) > 0 for row in history), eid, category, "缺少历史逾期依据")
        if role == "counterexample":
            _require(decimal(latest["outstanding_principal"]) > 0 and decimal(latest["overdue_principal"]) > 0,
                     eid, category, "反例合同在基准日缺少实际未清偿逾期")
            return "historical_overdue_and_current_unsettled_balance"
        _require(decimal(latest["outstanding_principal"]) == 0 and decimal(latest["overdue_principal"]) == 0,
                 eid, category, "结清变体在基准日仍有未偿或逾期余额")
        return "historical_principal_and_overdue_with_current_zero_balance"
    if category == "B2":
        row = _latest_business_month(monthly, "2026-02-01")
        _require(row["tax_amount"] is not None, eid, category, "缺少可确认的税款字段")
        value = decimal(row["tax_amount"])
        _require(value > 0 if role == "counterexample" else value == 0,
                 eid, category, "确认零或正数的业务变体与角色不符")
        return "confirmed_positive_tax_payment" if role == "counterexample" else "confirmed_zero_tax_payment"
    if category == "B3":
        first = _latest_business_month(monthly, "2025-11-01")
        second = _latest_business_month(monthly, "2025-12-01")
        _require(first["source_record_ref"] != second["source_record_ref"] and
                 decimal(first["operating_revenue"]) == decimal(second["operating_revenue"]),
                 eid, category, "未绑定两次真实同值经营事实")
        if role in {"transfer", "counterexample"}:
            revisions = [row for row in monthly if row["stat_month"] == "2026-03-01"]
            _require(len({row["report_revision"] for row in revisions}) >= 2,
                     eid, category, "缺少用于版本与报送行口径的真实修订")
        return "equal_values_in_distinct_months_with_lawful_revisions"
    if category == "C2":
        values = [_latest_business_month(monthly, month) for month in
                  ("2026-04-01", "2026-05-01", "2026-06-01")]
        signs = tuple(decimal(row["net_profit"]) < 0 for row in values)
        expected = {"typical": (True, True, False), "transfer": (True, True, True),
                    "counterexample": (True, False, True)}[role]
        _require(signs == expected and all(decimal(row["net_profit"]) != 0 for row in values),
                 eid, category, "实际相邻月损益过程与角色机制不一致")
        return {"typical": "two_adjacent_loss_months_then_recovery", "transfer": "three_adjacent_loss_months",
                "counterexample": "two_losses_separated_by_profitable_month"}[role]
    if category == "C4":
        first = _latest_business_month(monthly, "2026-06-01")
        second = _latest_business_month(monthly, "2026-07-01")
        r1, r2 = decimal(first["operating_revenue"]), decimal(second["operating_revenue"])
        _require(r1 > 0 and r2 > 0 and r1 != r2, eid, category, "两月未形成不同收入权重")
        _require(decimal(first["total_profit"]) / r1 != decimal(second["total_profit"]) / r2,
                 eid, category, "两月利润率相同，无法观察计算公式差别")
        return "different_revenue_weights_and_different_monthly_profit_rates"
    if category == "A4":
        august = _latest_business_month(monthly, "2026-08-01")
        _require(august["period_end"] == "2026-08-31" and august["ingested_at"] > "2026-08-31 23:59:59",
                 eid, category, "缺少跨采集月的业务事实")
        return "august_business_reported_and_ingested_in_september"
    if category == "A5":
        revisions = [row for row in monthly if row["stat_month"] == "2026-03-01"]
        _require(len({row["report_revision"] for row in revisions}) >= 2 and
                 len({row["operating_revenue"] for row in revisions}) >= 2,
                 eid, category, "缺少数值不同的同业务时点合法修订")
        return "distinct_original_and_revised_values_at_same_business_time"
    return "observable_period_and_state_conditions_for_context_changes"


def _context_transition(role, eid, monthly, negatives):
    """Concrete private scope transitions, with exact public-row witnesses."""
    if role == "typical":
        initial = {"enterprise_id": eid, "period_start": "2026-04-01", "period_end": "2026-06-30",
                   "measure": "monthly_net_profit"}
        followup = {**initial, "period_end": "2026-05-31"}
        before = [row for row in monthly if "2026-04-01" <= row["stat_month"] <= "2026-06-01"]
        after = [row for row in before if row["stat_month"] <= "2026-05-01"]
        table, changed, retained = "enterprise_operation_monthly", ["period_end"], ["enterprise_id", "period_start", "measure"]
        cancelled = []
    else:
        table = "enterprise_negative_event"
        _require(len(negatives) >= 2, eid, "C6", "状态条件变更缺少两个公开状态观察")
        if role == "transfer":
            initial = {"enterprise_id": eid, "event_type": "高管涉案", "selection": "historical_occurrence"}
            followup = {"enterprise_id": eid, "event_type": "高管涉案", "selection": "business_state_as_of",
                        "state_as_of": "2026-08-31", "knowledge_cutoff": "2026-09-15 23:59:59"}
            earliest, latest = {}, {}
            for row in negatives:
                identity = row["business_event_ref"]
                if identity not in earliest or row["state_effective_at"] < earliest[identity]["state_effective_at"]:
                    earliest[identity] = row
                if (row["state_effective_at"] <= "2026-08-31 23:59:59" and
                        row["ingested_at"] <= "2026-09-15 23:59:59" and
                        (identity not in latest or row["state_effective_at"] > latest[identity]["state_effective_at"])):
                    latest[identity] = row
            before, after = list(earliest.values()), list(latest.values())
            changed, retained, cancelled = ["selection", "state_as_of", "knowledge_cutoff"], ["enterprise_id", "event_type"], []
        else:
            first_state = min(row["state_effective_at"] for row in negatives)
            initial = {"enterprise_id": eid, "event_type": "高管涉案", "selection": "state_history",
                       "state_effective_at": first_state}
            followup = {key: value for key, value in initial.items() if key != "state_effective_at"}
            before = [row for row in negatives if row["state_effective_at"] == first_state]
            after = list(negatives)
            changed, retained, cancelled = [], ["enterprise_id", "event_type", "selection"], ["state_effective_at"]
    before_refs, after_refs = _refs([(table, before)]), _refs([(table, after)])
    _require(before_refs and after_refs and before_refs != after_refs, eid, "C6", "条件变更未形成可核对的记录范围差异")
    return {"initial_scope": initial, "followup_scope": followup,
            "changed_conditions": changed, "inherited_conditions": retained, "cancelled_conditions": cancelled,
            "initial_record_refs": before_refs, "followup_record_refs": after_refs,
            "visibility": "public_records_and_dictionary", "requires_hidden_truth": False}


def build_fact_scenarios(tables, enterprise_features, definitions):
    check_definitions(definitions)
    by_enterprise = {table: defaultdict(list) for table in tables}
    global_people = defaultdict(list)
    for table, rows in tables.items():
        for row in rows:
            by_enterprise[table][row["enterprise_id"]].append(row)
            if table == "enterprise_person":
                global_people[row["person_entity_id"]].append(row)
    result = []
    for number, features in sorted(enterprise_features.items()):
        if (number - 1) % 40 + 1 not in SCENARIO_FAMILIES:
            continue
        eid = f"SYN-E{number:06d}"
        selected = {table: values[eid] for table, values in by_enterprise.items()}
        monthly, people = selected["enterprise_operation_monthly"], selected["enterprise_person"]
        profiles = selected["enterprise_operation_profile"]
        loans = [row for row in selected["enterprise_operation_event"] if row["event_type"] == "银行借款"]
        contract = loans[0]["business_object_ref"] if loans else None
        loan_history = [row for row in loans if row["business_object_ref"] == contract]
        negatives = [row for row in selected["enterprise_negative_event"] if row["event_type"] == "高管涉案"]
        _require(negatives, eid, "C1", "缺少与本企业人员可靠关联的事项")
        referenced_ids = {row["related_person_id"] for row in negatives}
        _require(any(row["person_id"] in referenced_ids for row in people),
                 eid, "C1", "事项人员引用缺少本企业公开任职记录")

        def months(*values):
            return [row for row in monthly if row["stat_month"] in values]

        bindings = {
            "A1": [("enterprise_person", people)],
            "A2": [("enterprise_operation_event", loan_history), ("enterprise_operation_profile", profiles[-1:])],
            "A4": [("enterprise_operation_monthly", months("2026-08-01"))],
            "A5": [("enterprise_operation_monthly", months("2026-03-01")),
                   ("enterprise_operation_profile", [row for row in profiles if row["as_of_date"] == "2025-12-31"]),
                   ("enterprise_negative_event", negatives)],
            "B2": [("enterprise_operation_monthly", months("2026-02-01")),
                   ("enterprise_data_coverage", [row for row in selected["enterprise_data_coverage"]
                                                 if row["table_name"] == "enterprise_operation_monthly"])],
            "B3": [("enterprise_operation_monthly", months("2025-11-01", "2025-12-01", "2026-03-01"))],
            "C1": [("enterprise_person", people), ("enterprise_negative_event", negatives),
                   ("enterprise_operation_monthly", months("2026-01-01"))],
            "C2": [("enterprise_operation_monthly", months("2026-04-01", "2026-05-01", "2026-06-01")),
                   ("enterprise_negative_event", negatives)],
            "C4": [("enterprise_operation_monthly", months("2026-06-01", "2026-07-01")),
                   ("enterprise_operation_profile", profiles)],
            "C6": [("enterprise_operation_monthly", months("2026-04-01", "2026-05-01", "2026-06-01")),
                   ("enterprise_negative_event", negatives)],
        }
        for category in TARGET_CATEGORIES:
            role = variant_for(number, category, definitions)
            definition, variant = definitions["categories"][category], definitions["categories"][category][role]
            predicate = _role_evidence(category, role, eid, selected, global_people)
            evidence_bindings, related = list(bindings[category]), set()
            if role == "transfer" and category in {"A1", "C1"}:
                local_entities = {row["person_entity_id"] for row in people}
                matching = [row for entity in sorted(local_entities) for row in global_people[entity]
                            if row["enterprise_id"] != eid]
                related.update(row["enterprise_id"] for row in matching)
                evidence_bindings.append(("enterprise_person", matching))
                if category == "C1":
                    for other_id in sorted(related):
                        other_events = by_enterprise["enterprise_negative_event"][other_id]
                        other_ids = {row["related_person_id"] for row in other_events}
                        other_people = [row for row in by_enterprise["enterprise_person"][other_id]
                                        if row["person_id"] in other_ids]
                        evidence_bindings.extend((("enterprise_negative_event", other_events),
                                                  ("enterprise_person", other_people)))
            scenario = {
                "scenario_id": f"{category}-{eid}", "category": category, "enterprise_id": eid,
                "roles": [role], "learning_phases": [ROLE_PHASES[role]],
                "record_refs": _refs(evidence_bindings), "related_enterprise_ids": sorted(related),
                "public_basis": definition["mechanism"], "response_type": variant["response_type"],
                "expected_behavior": variant["expected_behavior"], "target_scope": variant["target_scope"],
                "business_features": list(features), "design_origin": "genuine_business_fact",
                "role_assignment": "family_category_rotation_v1",
                "evidence": {"business_predicate": predicate, "observed_dimensions": definition["observed_dimensions"],
                             "contrasting_scopes": variant["contrasting_scopes"],
                             "visibility": "public_records_and_dictionary", "requires_hidden_truth": False},
            }
            if category == "C6":
                scenario["context_transition"] = _context_transition(role, eid, monthly, negatives)
                scenario["context_contract"] = {
                    "base_enterprise_id": eid,
                    "condition_change": {"typical": "replace_period_keep_enterprise",
                                         "transfer": "replace_state_time_keep_enterprise_and_event_type",
                                         "counterexample": "remove_explicitly_cancelled_filter"}[role],
                    "unchanged_conditions_are_inherited": True, "cancelled_conditions_are_not_inherited": True,
                }
            result.append(scenario)
        role = variant_for(number, "A2", definitions)
        result.append({
            "scenario_id": f"AMB-A2-{eid}", "category": "A2", "enterprise_id": eid,
            "roles": [role], "learning_phases": [ROLE_PHASES[role]],
            "record_refs": _refs(bindings["A2"]), "related_enterprise_ids": [],
            "public_basis": definitions["categories"]["A2"]["mechanism"], "response_type": "clarify",
            "expected_behavior": "当借款规模未指定指合同本金、未偿本金还是逾期本金时，列明合理口径并澄清所需指标。",
            "target_scope": "未指定余额时点和指标含义的借款规模",
            "business_features": list(features), "design_origin": "public_scope_ambiguity",
            "role_assignment": "family_category_rotation_v1",
            "evidence": {"business_predicate": "multiple_defined_loan_metrics_with_different_values",
                         "observed_dimensions": ["amount", "outstanding_principal", "overdue_principal", "balance_as_of_date"],
                         "contrasting_scopes": ["合同本金", "截至指定时点未偿本金", "截至指定时点逾期本金"],
                         "visibility": "public_records_and_dictionary", "requires_hidden_truth": False},
        })
    return result
