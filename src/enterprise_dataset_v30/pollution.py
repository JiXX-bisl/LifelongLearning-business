"""在已确定的业务事实上施加有台账的受控观测扰动。"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timedelta
from decimal import Decimal

from .common import OPERATIONS, decimal, money, rng
from .schema import MONEY_FIELDS, PRIMARY_KEYS, TABLES


def key_of(table, row):
    return tuple(row[name] for name in PRIMARY_KEYS[table])


def key_dict(table, row):
    return {name: row[name] for name in PRIMARY_KEYS[table]}


def _stratum_statistics(basic_rows, assignments, profiles):
    scales = {row["enterprise_id"]: row["scale"] for row in (profiles or [])}
    strata = defaultdict(lambda: {"none": 0, "single": 0, "double": 0})
    for row in basic_rows:
        cohort = "old" if int(row["enterprise_id"].removeprefix("SYN-E")) <= 40 else "new"
        key = (cohort, row["industry"], scales.get(row["enterprise_id"], "未知规模"))
        strata[key][assignments[row["enterprise_id"]]["group"]] += 1
    return [{"cohort": key[0], "industry": key[1], "scale": key[2], "groups": value}
            for key, value in sorted(strata.items())]


def _stratified_order(rows, seed, cohort, scales=None):
    buckets = defaultdict(list)
    for row in rows:
        stratum = (row["industry"], (scales or {}).get(row["enterprise_id"], "未知规模"))
        buckets[stratum].append(row["enterprise_id"])
    industries = sorted({stratum[0] for stratum in buckets})
    rng(seed, "pollution-industries", cohort).shuffle(industries)
    stratum_order = {}
    for industry in industries:
        strata = sorted(stratum for stratum in buckets if stratum[0] == industry)
        rng(seed, "pollution-scale-order", cohort, industry).shuffle(strata)
        stratum_order[industry] = strata
    for stratum, members in buckets.items():
        members.sort()
        rng(seed, "pollution-members", cohort, *stratum).shuffle(members)
    ordered = []
    while any(buckets.values()):
        for industry in industries:
            strata = stratum_order[industry]
            for _ in range(len(strata)):
                stratum = strata.pop(0)
                strata.append(stratum)
                if buckets[stratum]:
                    ordered.append(buckets[stratum].pop())
                    break
    return ordered


def assign_pollution(basic_rows, config, profiles=None, equal_carriers=None):
    assignments = {}
    scales = {row["enterprise_id"]: row["scale"] for row in (profiles or [])}
    for cohort, predicate, distribution in (
        ("old", lambda row: int(row["enterprise_id"].removeprefix("SYN-E")) <= 40,
         config["pollution"]["old_distribution"]),
        ("new", lambda row: int(row["enterprise_id"].removeprefix("SYN-E")) > 40,
         config["pollution"]["new_distribution"]),
    ):
        ordered = _stratified_order([row for row in basic_rows if predicate(row)], config["seed"], cohort, scales)
        if sum(distribution.values()) != len(ordered):
            raise ValueError("企业污染分组配额与来源企业数不符")
        start = 0
        for group in ("none", "single", "double"):
            for enterprise_id in ordered[start:start + distribution[group]]:
                assignments[enterprise_id] = {"group": group, "classes": []}
            start += distribution[group]
    # 固定均衡类别配额；操作类别与企业编号没有连续段对应关系。
    singles = sorted(e for e, a in assignments.items() if a["group"] == "single")
    doubles = sorted(e for e, a in assignments.items() if a["group"] == "double")
    rng(config["seed"], "single-classes").shuffle(singles)
    rng(config["seed"], "double-classes").shuffle(doubles)
    for index, enterprise_id in enumerate(singles):
        assignments[enterprise_id]["classes"] = [OPERATIONS[index % len(OPERATIONS)]]
    for index, enterprise_id in enumerate(doubles):
        first = index % len(OPERATIONS)
        assignments[enterprise_id]["classes"] = [OPERATIONS[first], OPERATIONS[(first + 1) % len(OPERATIONS)]]
    # Pairing is a benign fact-identity requirement, independent of risk state
    # and the none/single/double grouping.  Exchange complete class lists only
    # inside an existing group so all class and company quotas are preserved.
    require_pairing = equal_carriers is not None
    equal_carriers = set(equal_carriers or ())
    minimum = config["coverage_requirements"]["minimum_equal_real_and_duplicate_enterprises"] if require_pairing else 0
    def paired():
        return {enterprise_id for enterprise_id in equal_carriers
                if "duplicate_submission" in assignments[enterprise_id]["classes"]}
    while len(paired()) < minimum:
        choices = []
        for donor, assignment in assignments.items():
            if donor in equal_carriers or "duplicate_submission" not in assignment["classes"]:
                continue
            for target in equal_carriers:
                other = assignments[target]
                if other["group"] == assignment["group"] and "duplicate_submission" not in other["classes"]:
                    choices.append((donor, target))
        choices.sort()
        if not choices:
            raise ValueError("既定污染分组不能承载至少12家真实同值与重复报送配对，拒绝更改真实风险或企业分组")
        donor, target = rng(config["seed"], "equal-duplicate-class-pair", len(paired())).choice(choices)
        assignments[donor]["classes"], assignments[target]["classes"] = assignments[target]["classes"], assignments[donor]["classes"]
    return assignments


def apply_pollution(base_tables, config, *, profiles=None, protected_record_refs=None):
    tables = deepcopy(base_tables)
    equal_months = defaultdict(set)
    by_amount = defaultdict(lambda: defaultdict(set))
    for row in base_tables["enterprise_operation_monthly"]:
        amount = decimal(row["operating_revenue"]) * (Decimal(10000) if row["amount_unit"] == "万元" else Decimal(1))
        by_amount[row["enterprise_id"]][amount].add(row["stat_month"])
    for enterprise_id, amounts in by_amount.items():
        for months in amounts.values():
            if len(months) >= 2:
                equal_months[enterprise_id].update(months)
    assignments = assign_pollution(tables["enterprise_basic"], config, profiles, equal_months)
    indexes = {table: {key_of(table, row): row for row in rows} for table, rows in tables.items()}
    original_keys = {table: set(index) for table, index in indexes.items()}
    protected = {(entry["table"], tuple(entry["primary_key"][key] for key in PRIMARY_KEYS[entry["table"]]))
                 for entry in (protected_record_refs or [])}
    zero_months = {}
    for row in base_tables["enterprise_operation_monthly"]:
        if decimal(row["tax_amount"]) == 0:
            enterprise_id = row["enterprise_id"]
            zero_months[enterprise_id] = min(zero_months.get(enterprise_id, row["stat_month"]), row["stat_month"])
    for row in base_tables["enterprise_operation_monthly"]:
        # Preserve a genuine zero business month per enterprise, plus all
        # explicitly referenced scenes. Legitimate prolonged zero payments
        # must not remove the entire company from missing-month candidates.
        if (row["stat_month"] in equal_months[row["enterprise_id"]] or
                row["stat_month"] == zero_months.get(row["enterprise_id"])):
            protected.add(("enterprise_operation_monthly", key_of("enterprise_operation_monthly", row)))
    touches = defaultdict(set)
    ledger = []
    cutoff = datetime.fromisoformat(config["knowledge_cutoff"])
    rates = {name: decimal(value) for name, value in config["pollution"]["rates"].items()}
    max_operations = config["pollution"]["max_operations_per_record"]
    sampling = defaultdict(lambda: {"eligible": 0, "guaranteed": 0, "sampled_candidates": 0,
                                    "sampled_selected": 0, "applied_original_records": 0})

    def record(operation, table, before, after, *, behavior="answer", basis="", nature="数据质量"):
        identity = key_of(table, before)
        if len(touches[(table, identity)] | {operation}) > max_operations:
            raise ValueError("单行污染操作超过配置上限")
        touches[(table, identity)].add(operation)
        ledger.append({
            "operation": operation, "table": table, "source_key": key_dict(table, before),
            "before": deepcopy(before), "after": deepcopy(after),
            "expected_behavior": behavior, "public_basis": basis, "nature": nature,
        })

    def replace(operation, table, row, new, **notes):
        identity = key_of(table, row)
        record(operation, table, row, new, **notes)
        indexes[table][identity] = new

    def available(table, row, operation):
        identity = key_of(table, row)
        return (identity in original_keys[table]
                and ((table, identity) not in protected or operation in ("unit_representation", "duplicate_submission"))
                and len(touches[(table, identity)] | {operation}) <= max_operations)

    def candidates(enterprise_id, operation):
        result = []
        for table in TABLES:
            for identity in sorted(original_keys[table]):
                row = indexes[table].get(identity)
                if row is None or row["enterprise_id"] != enterprise_id or not available(table, row, operation):
                    continue
                if operation == "duplicate_submission" and table in (
                    "enterprise_operation_monthly", "enterprise_operation_profile",
                    "enterprise_operation_event", "enterprise_negative_event",
                ):
                    # 不复制人员表，保留负面事件外键的稳定对应。
                    result.append((table, row))
                elif operation == "missing_field" and table in (
                    "enterprise_operation_monthly", "enterprise_operation_profile", "enterprise_person",
                ):
                    fields = _missing_fields(table, row)
                    if fields:
                        result.append((table, row))
                elif operation == "delayed_report" and "ingested_at" in row:
                    if datetime.fromisoformat(row["ingested_at"]) + timedelta(days=1) <= cutoff:
                        result.append((table, row))
                elif operation == "unit_representation" and table in MONEY_FIELDS and row.get("amount_unit") == "元":
                    if any(row.get(field) is not None and decimal(row[field]) != 0 for field in MONEY_FIELDS[table]):
                        result.append((table, row))
                elif operation == "cross_field_conflict" and table == "enterprise_operation_monthly":
                    revenue_eligible = (row.get("operating_revenue") is not None and row.get("main_business_revenue") is not None
                                        and decimal(row["operating_revenue"]) > 0)
                    invoice_eligible = all(row.get(field) is not None for field in
                                           ("invoice_purchase_amount", "utility_invoice_amount", "travel_hospitality_invoice_amount"))
                    if revenue_eligible or invoice_eligible:
                        result.append((table, row))
                elif operation == "cross_field_conflict" and table == "enterprise_person":
                    if ((row.get("is_current") == 0 and row.get("end_date") is not None
                         and row["end_date"] < config["reference_date"])
                            or (row.get("is_current") == 1 and row.get("end_date") is None
                                and row["start_date"] < config["reference_date"])):
                        result.append((table, row))
        return result

    def mutate(enterprise_id, operation, table, row):
        row = indexes[table].get(key_of(table, row))
        if row is None or not available(table, row, operation):
            return False
        local = rng(config["seed"], enterprise_id, table, *key_of(table, row), operation)
        new = deepcopy(row)
        if operation == "missing_field":
            fields = _missing_fields(table, row)
            if not fields:
                return False
            field = fields[local.randrange(len(fields))]
            new[field] = None
            replace(operation, table, row, new, behavior="insufficient",
                    basis=f"{field}为NULL，字段字典和其他报送版本可查询；不能把未知直接替代为零。")
        elif operation == "unit_representation":
            if row.get("amount_unit") != "元":
                return False
            new["amount_unit"] = "万元"
            for field in MONEY_FIELDS[table]:
                if row.get(field) is not None:
                    new[field] = format(decimal(row[field]) / Decimal(10000), ".6f")
            replace(operation, table, row, new, basis="amount_unit明确为万元，原始金额的换算保持精确。", nature="表示变化")
        elif operation == "delayed_report":
            original = datetime.fromisoformat(row["ingested_at"])
            available_days = (cutoff - original).days
            if available_days < 1:
                return False
            delay = local.randint(1, min(available_days, 45))
            reported = datetime.fromisoformat(row["reported_at"])
            new["reported_at"] = (reported + timedelta(days=delay)).strftime("%Y-%m-%d %H:%M:%S")
            new["ingested_at"] = (original + timedelta(days=delay)).strftime("%Y-%m-%d %H:%M:%S")
            replace(operation, table, row, new, basis="业务日期、状态生效、reported_at与ingested_at均可观察；迟报不改变发生日期，旧版本较晚收到也不自动替代同统计时点的更高修订。", nature="来源时效")
        elif operation == "cross_field_conflict":
            if table == "enterprise_person":
                if row["end_date"] is not None:
                    new["is_current"] = 1
                else:
                    # NULL结束日期自身不能证明仍在任。用明确已经结束的
                    # 日期与来源在任标志构造可观察矛盾，不误标未知离任时间。
                    new["end_date"] = (datetime.fromisoformat(config["reference_date"])
                                       - timedelta(days=1)).date().isoformat()
                basis = "报送明确显示任职在基准日前已经结束，同时仍将基准日在任标志报为1；应指出冲突并核实来源，无法据此猜测哪一字段正确。"
            else:
                unit = Decimal(10000) if row.get("amount_unit") == "万元" else Decimal(1)
                modes = []
                if row.get("operating_revenue") is not None and row.get("main_business_revenue") is not None:
                    modes.append("revenue")
                if all(row.get(field) is not None for field in ("invoice_purchase_amount", "utility_invoice_amount", "travel_hospitality_invoice_amount")):
                    modes.append("invoice")
                mode = local.choice(modes)
                if mode == "revenue":
                    new_value = money(decimal(row["operating_revenue"]) * unit * Decimal("1.08") + Decimal(1))
                    new["main_business_revenue"] = format(decimal(new_value) / unit, ".6f") if unit == 10000 else new_value
                    basis = "同口径同期间主营收入大于营业总收入；无公开更正依据时应报告矛盾，不能猜测真实值。"
                else:
                    new_value = money(decimal(row["invoice_purchase_amount"]) * unit * Decimal("1.03") + Decimal(1))
                    new["utility_invoice_amount"] = format(decimal(new_value) / unit, ".6f") if unit == 10000 else new_value
                    basis = "同口径进项发票已定义子项之和大于进项总额；应报告冲突并核实来源，不猜测哪个原始字段正确。"
            replace(operation, table, row, new, behavior="clarify",
                    basis=basis)
        elif operation == "duplicate_submission":
            primary = PRIMARY_KEYS[table][0]
            # SHA256标识不包含污染标签，业务身份和来源版本保持原样。
            token = format(local.getrandbits(96), "024x")
            new[primary] = "SYN-R" + token
            if key_of(table, new) in indexes[table]:
                raise ValueError("派生报送记录主键冲突")
            record(operation, table, row, new, basis="技术记录ID不同，业务身份、统计期间和来源记录引用相同。")
            indexes[table][key_of(table, new)] = new
        else:
            raise ValueError(f"未知操作：{operation}")
        return True

    # 缺月按整个企业月份删除，不能仅删掉一条修订而留下其余版本。
    for enterprise_id in sorted(assignments):
        if "missing_month" not in assignments[enterprise_id]["classes"]:
            continue
        months = sorted({row["stat_month"] for row in indexes["enterprise_operation_monthly"].values()
                         if row["enterprise_id"] == enterprise_id and
                         all(("enterprise_operation_monthly", key_of("enterprise_operation_monthly", other)) not in protected
                             for other in indexes["enterprise_operation_monthly"].values()
                             if other["enterprise_id"] == enterprise_id and other["stat_month"] == row["stat_month"])})
        order = list(months)
        rng(config["seed"], enterprise_id, "missing-month-order").shuffle(order)
        chosen = []
        sampling["missing_month"]["eligible"] += len(months)
        for month in order:
            proposed = sorted([*chosen, month])
            longest = run = 1
            for previous, following in zip(proposed, proposed[1:]):
                a, b = datetime.fromisoformat(previous), datetime.fromisoformat(following)
                run = run + 1 if b.year * 12 + b.month == a.year * 12 + a.month + 1 else 1
                longest = max(longest, run)
            if longest > config["pollution"]["max_consecutive_missing_months"]:
                continue
            guaranteed = not chosen
            selected = guaranteed or rng(config["seed"], enterprise_id, month, "missing_month").random() < rates["missing_month"]
            if not guaranteed:
                sampling["missing_month"]["sampled_candidates"] += 1
            if selected:
                chosen.append(month)
                sampling["missing_month"]["guaranteed" if guaranteed else "sampled_selected"] += 1
        if not chosen:
            raise ValueError(f"{enterprise_id}没有可用于缺报的适用月份")
        for identity, row in list(indexes["enterprise_operation_monthly"].items()):
            if row["enterprise_id"] == enterprise_id and row["stat_month"] in chosen:
                record("missing_month", "enterprise_operation_monthly", row, None,
                       behavior="insufficient", basis="采集覆盖为部分；缺失月份不是已确认的零经营。")
                del indexes["enterprise_operation_monthly"][identity]
                sampling["missing_month"]["applied_original_records"] += 1
        for identity, row in list(indexes["enterprise_data_coverage"].items()):
            if row["enterprise_id"] == enterprise_id and row["table_name"] == "enterprise_operation_monthly":
                new = {**row, "coverage_status": "部分"}
                replace("missing_month", "enterprise_data_coverage", row, new,
                        behavior="insufficient", basis="采集覆盖状态明确说明月度数据没有完整覆盖。")

    # 单位先转换、再缺失/矛盾/时效，最后复制；每个操作至少落地一次。
    order = ("unit_representation", "missing_field", "cross_field_conflict", "delayed_report", "duplicate_submission")
    for operation in order:
        for enterprise_id in sorted(assignments):
            if operation not in assignments[enterprise_id]["classes"]:
                continue
            eligible = candidates(enterprise_id, operation)
            rng(config["seed"], enterprise_id, operation, "candidate-order").shuffle(eligible)
            if operation == "duplicate_submission" and bool(equal_months[enterprise_id]):
                eligible.sort(key=lambda item: not (item[0] == "enterprise_operation_monthly" and
                                                    item[1]["stat_month"] in equal_months[enterprise_id]))
            sampling[operation]["eligible"] += len(eligible)
            applied = 0
            # 首条强制满足配额，其他条按该类追加概率抽样。
            for table, row in eligible:
                guaranteed = applied == 0
                if not guaranteed:
                    sampling[operation]["sampled_candidates"] += 1
                selected = guaranteed or rng(config["seed"], enterprise_id, table,
                                              *key_of(table, row), operation, "sample").random() < rates[operation]
                if selected and mutate(enterprise_id, operation, table, row):
                    applied += 1
                    sampling[operation]["guaranteed" if guaranteed else "sampled_selected"] += 1
                    sampling[operation]["applied_original_records"] += 1
            if applied == 0:
                raise ValueError(f"{enterprise_id}没有实现指定的{operation}操作")

    tables = {table: [index[key] for key in sorted(index)] for table, index in indexes.items()}
    report_tables = {table for table in TABLES if base_tables[table] and "report_revision" in base_tables[table][0]}
    modified = {identity for identity, operations in touches.items()
                if identity[0] in report_tables and operations - {"duplicate_submission"}}
    base_count = sum(len(base_tables[table]) for table in report_tables)
    if Decimal(len(modified)) / Decimal(base_count) > decimal(config["pollution"]["max_modified_fraction"]):
        raise ValueError("实际修改/缺报比例超过15%；请审核配置，拒绝自动删减关键样本")
    return {
        "tables": tables, "assignments": assignments, "pollution_ledger": ledger,
        "statistics": {
            "base_observation_count": base_count, "modified_or_missing_original_records": len(modified),
            "modified_fraction": str(Decimal(len(modified)) / Decimal(base_count)),
            "added_duplicate_records": sum(item["operation"] == "duplicate_submission" for item in ledger),
            "operations": {operation: sum(item["operation"] == operation for item in ledger) for operation in OPERATIONS},
            "sampling": {operation: {**values,
                                      "configured_probability": str(rates[operation]),
                                      "observed_selected_fraction": str(Decimal(values["sampled_selected"]) / Decimal(values["sampled_candidates"]))
                                      if values["sampled_candidates"] else None}
                         for operation, values in sorted(sampling.items())},
            "strata": _stratum_statistics(tables["enterprise_basic"], assignments, profiles),
            "equal_real_and_duplicate_enterprises": len({item["before"]["enterprise_id"] for item in ledger
                                                           if item["operation"] == "duplicate_submission" and bool(equal_months[item["before"]["enterprise_id"]])}),
        },
    }


def _missing_fields(table, row):
    fields = {
        "enterprise_operation_monthly": ("net_profit", "tax_amount", "social_security_count", "main_business_revenue"),
        "enterprise_operation_profile": ("total_assets", "total_liabilities", "has_wechat_account"),
        "enterprise_person": ("age", "household_region"),
    }.get(table, ())
    return [field for field in fields if row.get(field) is not None]
