"""Build reproducible business facts and lawful reporting history.

This module has no database, model, command-line or import-time side effects.
Random observation damage is applied separately by the dataset builder. The
scenario map and generation profiles are evaluation artifacts, not query tables.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta
from decimal import Decimal

from .common import decimal, dt, money, read_json, rng
from .schema import MONEY_FIELDS, PRIMARY_KEYS, TABLE_COLUMNS, TABLES
from .economics import operating_months, snapshot_stocks, feature_plan, opening_stocks, contract_interest_rate, initial_reporting_estimate
from .scenarios import variant_for, build_fact_scenarios, check_definitions


_FAMILIES = (2, 8, 14, 20, 26, 32)
_PREFIXES = ("霁禾", "澄川", "岚序", "景衡", "沐承", "栖远", "清宸", "珩益",
             "弘澜", "锦原", "承曜", "启砚", "和晟", "晏庭", "泽屹", "宁拓")
_PROFILE_DATES = ("2025-08-31", "2025-12-31", "2026-03-31", "2026-08-31")


def _fraction(random, low, high):
    # Decimal parameters and integer sampling never pass amounts through float.
    quantum = Decimal(1000000)
    first = int(decimal(low) * quantum)
    last = int(decimal(high) * quantum)
    return Decimal(random.randint(first, last)) / quantum


def _metadata(config, table, source_ref, business_date, revision=1, delay=3):
    reported = dt(business_date, delay)
    return {
        "source_system": config.get("generation", {}).get("source_system", "企业报送系统"),
        "source_record_ref": source_ref,
        "reported_at": reported,
        "ingested_at": dt(reported, 1),
        "report_revision": revision,
        "report_status": "正式",
    }


def _row(table, **values):
    result = {column: None for column in TABLE_COLUMNS[table]}
    result.update(values)
    return result


def _ref(table, row):
    return {"table": table,
            "primary_key": {key: row[key] for key in PRIMARY_KEYS[table]}}


def _profile_definitions(payload):
    if isinstance(payload, list):
        rows = payload
    else:
        rows = payload.get("画像配置清单", payload.get("profiles", []))
    result = {}
    for position, row in enumerate(rows, 1):
        basic = row.get("basic_info", row)
        key = basic.get("profile_type_id", basic.get("画像类型ID", f"P{position:03d}"))
        result[key] = {
            "profile_type_id": key,
            "scale": basic.get("scale", basic.get("规模", "中型")),
            "development_stage": basic.get("development_stage", basic.get("发展阶段", "成熟期")),
            "operating_trend": basic.get("operating_trend", basic.get("经营趋势", "基本平稳")),
            "business_model": basic.get("business_model", basic.get("业务模式", "一般经营")),
            "operating_characteristics": basic.get("operating_characteristics", basic.get("典型经营特点", "")),
            "source_business_variation": basic.get("source_business_variation", basic.get("企业差异设定", "")),
        }
    return result


def _basic(config, source, number, profile):
    row = deepcopy(source)
    original = number <= 40
    random = rng(config["seed"], "basic", number)
    enterprise_id = f"SYN-E{number:06d}"
    row["enterprise_id"] = enterprise_id
    if not original:
        family = (number - 1) % 40 + 1
        generation = (number - 1) // 40
        prefix = _PREFIXES[(family + generation * 5) % len(_PREFIXES)]
        row["enterprise_name"] = prefix + ("锦" if generation == 1 else "宸") + source["enterprise_name"][2:]
        row["synthetic_credit_code"] = f"SYN-USCC-{number:06d}"
        stage = profile["development_stage"]
        years = ((2023, 2024) if "初创" in stage else
                 (2018, 2022) if "成长" in stage else (2006, 2016))
        year = random.randint(*years)
        month = random.randint(1, 8 if year == 2024 else 12)
        row["established_date"] = f"{year:04d}-{month:02d}-{random.randint(1, 25):02d}"
        row["registered_capital"] = money(decimal(source["registered_capital"]) * _fraction(random, ".72", "1.38"))
        row["registered_email"] = f"office{number:03d}@enterprise{number:03d}.example.test"
        row["registered_phone"] = f"SYN-TEL-{number:06d}"
        row["registered_address"] = f"上海市衡川区虚构商务路{100 + number}号{generation + 1}幢{100 + family}室（虚构地址）"
        variant = ("重点服务区域内稳定客户，按批次验收结算，扩张人员在订单增长后逐步到岗。" if generation == 1 else
                   "同时服务区域客户和跨区域项目，采用分阶段交付与集中结算，人员投入先于收入确认。")
        row["main_business_description"] = source["main_business_description"] + variant
    row["registered_capital"] = money(row["registered_capital"])
    row["as_of_date"] = config["reference_date"]
    row["registration_status_effective_date"] = row["established_date"]
    row["source_updated_at"] = dt(config["reference_date"], 2)
    return _row("enterprise_basic", **row)


def _monthly_values(config, source_months, source_profile, basic, profile, number, ledgers, operations):
    return operating_months(config, source_months, source_profile, basic, profile,
                            number, variant_for, _row, _metadata, ledgers, operations)


def _monthly_reports(config, rows, number, processes):
    result = []
    revise = (number - 1) % 40 + 1 in _FAMILIES
    process_by_month = {process["stat_month"]: process for process in processes}
    for final in rows:
        if revise and final["stat_month"] == "2026-03-01":
            first = deepcopy(final)
            process = process_by_month[final["stat_month"]]
            estimated_values, reporting_process = initial_reporting_estimate(config, process, first["monthly_record_id"])
            first.update(estimated_values)
            process["lawful_reporting_estimates"] = [reporting_process]
            revised = deepcopy(final)
            revised["monthly_record_id"] = final["monthly_record_id"].removesuffix("R1") + "R2"
            revised.update(_metadata(config, "enterprise_operation_monthly", final["source_record_ref"],
                                     final["period_end"], revision=2, delay=11))
            process["report_revision"] = 2
            process["monthly_record_id"] = revised["monthly_record_id"]
            result.extend((first, revised))
        else:
            result.append(deepcopy(final))
    return result


def _persons(config, source_people, basic, number):
    rows = []
    family, generation = (number - 1) % 40 + 1, (number - 1) // 40
    role = variant_for(number, "A1") if family in _FAMILIES else None
    for position, original in enumerate(source_people, 1):
        values = deepcopy(original)
        person_id = original["person_id"] if number <= 40 else f"SYN-P{number * 100 + position:06d}"
        entity_id = f"PE-{number:06d}-{position:02d}"
        if number > 40:
            local = rng(config["seed"], "person", number, position)
            values["age"] = max(24, min(68, original["age"] + local.randint(-5, 5)))
            values["person_name"] = original["person_name"][0] + ("景" if generation == 1 else "予") + original["person_name"][-1]
            original_start = date.fromisoformat(original["start_date"])
            start = max(date.fromisoformat(basic["established_date"]), original_start)
            values["start_date"] = start.isoformat()
            if original["end_date"] is not None and date.fromisoformat(original["end_date"]) < start:
                values["end_date"] = (start + timedelta(days=90)).isoformat()
        values.update(person_id=person_id, enterprise_id=basic["enterprise_id"],
                      person_entity_id=entity_id, assignment_id=f"AS-{number:06d}-{position:02d}-C")
        row = _row("enterprise_person", **values)
        source_ref = f"PERSON-{basic['enterprise_id']}-{position:02d}-C"
        row.update(_metadata(config, "enterprise_person", source_ref, config["reference_date"], delay=2))
        if role == "typical" and position == 2:
            # One person left and later returned: two genuine assignments.
            start = max(date.fromisoformat(row["start_date"]), date.fromisoformat(basic["established_date"]))
            return_date = max(date(2025, 6, 1), start + timedelta(days=120))
            if return_date < date.fromisoformat(config["reference_date"]):
                historical = deepcopy(row)
                historical["person_id"] = person_id + "H1"
                historical["assignment_id"] = f"AS-{number:06d}-{position:02d}-H"
                historical["end_date"] = (return_date - timedelta(days=15)).isoformat()
                historical["is_current"] = 0
                historical.update(_metadata(config, "enterprise_person", source_ref + "-H", historical["end_date"]))
                rows.append(historical)
                row["start_date"], row["end_date"], row["is_current"] = return_date.isoformat(), None, 1
        rows.append(row)
        if role == "typical" and position == 3:
            revised = deepcopy(row)
            revised["person_id"] = person_id + "R2"
            revised["age"] = row["age"] + 1
            revised.update(_metadata(config, "enterprise_person", source_ref, config["reference_date"], 2, 6))
            rows.append(revised)
    if role == "transfer":
        anchor = source_people[-1]
        partner = family + 40 * ((generation + 1) % 3)
        partner_generation = (partner - 1) // 40
        partner_name, partner_age = anchor["person_name"], anchor["age"]
        if partner > 40:
            partner_random = rng(config["seed"], "person", partner, len(source_people))
            partner_age = max(24, min(68, partner_age + partner_random.randint(-5, 5)))
            partner_name = anchor["person_name"][0] + ("景" if partner_generation == 1 else "予") + anchor["person_name"][-1]
        row = _row("enterprise_person", person_id=f"SYN-P{number * 100 + 90:06d}",
            enterprise_id=basic["enterprise_id"], person_entity_id=f"PE-{partner:06d}-{len(source_people):02d}",
            assignment_id=f"AS-{number:06d}-90-C", person_name=partner_name,
            age=partner_age, household_region=anchor["household_region"], role="董事",
            start_date=max(basic["established_date"], "2024-09-01"), end_date=None, is_current=1)
        row.update(_metadata(config, "enterprise_person", f"PERSON-{basic['enterprise_id']}-90-C", config["reference_date"], delay=2))
        rows.append(row)
    # A counterexample has two different natural persons with one same name,
    # including original enterprises; equality of names does not merge them.
    if role == "counterexample":
        rows[1]["person_name"] = rows[0]["person_name"]
    return rows


def _operation(config, basic, number, token, event_type, event_date, status,
               object_ref=None, state_date=None, revision=1, **fields):
    business_event_ref = f"EV-{basic['enterprise_id']}-{token}"
    effective = state_date or event_date
    description = fields.pop("description", f"{basic['enterprise_name']}办理{event_type}事项，事项日期与来源报送日期分别留存。")
    if event_type in ("银行借款", "债券发行"):
        rate = contract_interest_rate(config, basic["enterprise_id"], event_type, object_ref or business_event_ref)
        description += f"合同约定年利率{rate * 100:.4f}%，按实际本金余额及实际天数/365计息，月末现金结息，未另约定额外罚息。"
    row = _row("enterprise_operation_event", operation_event_id=f"OE-{number:06d}-{token}-R{revision}",
        enterprise_id=basic["enterprise_id"], business_event_ref=business_event_ref,
        business_object_ref=object_ref or business_event_ref, event_date=event_date,
        state_effective_at=dt(effective), event_type=event_type, event_status=status,
        description=description,
        amount_unit="元", **fields)
    # A new business state is a new source observation time, not a correction
    # to an older statistical time. Technical row sequence remains separate.
    state_ref = f"{business_event_ref}-STATE-{str(effective)[:10]}"
    row.update(_metadata(config, "enterprise_operation_event", state_ref, effective, 1))
    for field in MONEY_FIELDS["enterprise_operation_event"]:
        if row[field] is not None:
            row[field] = money(row[field])
    return row


def _operations(config, source_events, basic, number, snapshots):
    result = []
    random = rng(config["seed"], "contracts", number)
    family, generation = (number - 1) % 40 + 1, (number - 1) // 40
    cohort = family in _FAMILIES
    established = date.fromisoformat(basic["established_date"])
    final = config["reference_date"]
    current_profile = max(snapshots, key=lambda row: (row["as_of_date"], row["report_revision"]))
    liability = decimal(current_profile["total_liabilities"])
    assets = decimal(current_profile["total_assets"])
    banks = [event for event in source_events if event["event_type"] == "银行借款"]
    overdue_sources = [event for event in source_events if event["event_type"] == "借款逾期"]
    if not banks and (cohort or overdue_sources):
        banks = [{"event_date": "2025-10-15", "amount": max(Decimal("10000"), liability * Decimal(".12")), "term_months": 24, "event_status": "正常还款"}]
    # Contract funding follows the source's equipment/working-capital plans;
    # each new enterprise changes those contract weights independently.
    weights = [decimal(template["amount"]) *
               (_fraction(rng(config["seed"], "loan-weight", number, position), ".68", "1.42") if number > 40 else Decimal(1))
               for position, template in enumerate(banks, 1)]
    funding_budget = sum(weights) * _fraction(random, ".90", "1.12")
    for position, template in enumerate(banks, 1):
        principal = decimal(money(funding_budget * weights[position - 1] / sum(weights)))
        origin = max(established + timedelta(days=14), date.fromisoformat(template["event_date"]))
        if cohort:
            origin = min(origin, date(2025, 12, 15))
        origin = min(origin, date(2026, 7, 1))
        token, object_ref = f"LOAN{position:02d}", f"CONTRACT-{basic['enterprise_id']}-{position:02d}"
        term = template.get("term_months") or 24
        result.append(_operation(config, basic, number, token, "银行借款", origin.isoformat(), "正常还款",
            object_ref, amount=principal, term_months=term, outstanding_principal=principal,
            overdue_principal=0, balance_as_of_date=origin.isoformat(),
            description="已签订借款合同并取得本金，合同本金与各观察日未偿本金分别记录。"))
        is_overdue = (cohort and position == 1) or bool(overdue_sources and position == 1)
        closed = (cohort and variant_for(number, "A2") != "counterexample") or (not cohort and template.get("event_status") == "已结清")
        balance = Decimal(0) if closed else principal * _fraction(random, ".32", ".54")
        overdue = balance * Decimal(".45") if is_overdue else Decimal(0)
        if is_overdue:
            overdue_date = max(origin + timedelta(days=45), date(2026, 4, 15))
            overdue_date = min(overdue_date, date(2026, 7, 20))
            overdue_at_occurrence = principal * Decimal(".18")
            event_token = f"OVERDUE{position:02d}"
            result.append(_operation(config, basic, number, token, "银行借款", origin.isoformat(), "已逾期",
                object_ref, overdue_date.isoformat(), 2, amount=principal, term_months=term,
                outstanding_principal=principal * Decimal(".70"), overdue_principal=overdue_at_occurrence,
                balance_as_of_date=overdue_date.isoformat(),
                description="借款合同已出现到期未偿本金，来源登记本期余额及逾期状态。"))
            result.append(_operation(config, basic, number, event_token, "借款逾期", overdue_date.isoformat(), "已逾期",
                object_ref, amount=overdue_at_occurrence, term_months=term,
                outstanding_principal=principal * Decimal(".70"), overdue_principal=overdue_at_occurrence,
                balance_as_of_date=overdue_date.isoformat(),
                description="合同约定的部分本金到期未偿，登记该次逾期发生额。"))
            result.append(_operation(config, basic, number, event_token, "借款逾期", overdue_date.isoformat(),
                "已结清" if closed else "已逾期", object_ref, final, 2,
                amount=overdue_at_occurrence, term_months=term, outstanding_principal=balance,
                overdue_principal=overdue, balance_as_of_date=final,
                description="已核对合同截至观察日的未偿本金和逾期本金，历史逾期发生额保持留存。"))
        result.append(_operation(config, basic, number, token, "银行借款", origin.isoformat(),
            "已结清" if closed else "已逾期" if is_overdue else "正常还款", object_ref, final, 3 if is_overdue else 2,
            amount=principal, term_months=term, outstanding_principal=balance,
            overdue_principal=overdue, balance_as_of_date=final,
            description=("截至观察日，合同本金已全部偿还，财务记录已更新。" if closed else
                         "截至观察日，合同仍有本金未偿，财务已登记本期余额。")))
    for position, template in enumerate(source_events, 1):
        kind = template["event_type"]
        if kind not in ("融资", "债券发行"):
            continue
        when = max(established + timedelta(days=30), date.fromisoformat(template["event_date"]))
        amount = (decimal(template["amount"]) if number <= 40 else
                  max(Decimal("10000"), assets * _fraction(random, ".055", ".16")))
        fields = {"amount": amount}
        if kind == "融资":
            fields.update(valuation=max(amount * Decimal(5), assets * _fraction(random, "1.7", "3.8")),
                          financing_stage=template.get("financing_stage") or "战略融资")
        else:
            fields.update(term_months=template.get("term_months") or 36)
        result.append(_operation(config, basic, number, f"CAPITAL{position:02d}", kind,
                                 when.isoformat(), "已完成", **fields))
    patent_sources = sorted((event for event in source_events if event["event_type"] in ("专利申请", "专利授权")),
                            key=lambda event: (event["event_date"], event["event_type"] != "专利申请"))
    pending = []
    serial = 0
    for template in patent_sources:
        when = max(established + timedelta(days=30), date.fromisoformat(template["event_date"]))
        for _ in range(int(template.get("quantity") or 1)):
            if template["event_type"] == "专利申请":
                serial += 1
                object_ref = f"PATENT-{basic['enterprise_id']}-{serial:03d}"
                pending.append((object_ref, when))
                result.append(_operation(config, basic, number, f"PATAPP{serial:03d}", "专利申请", when.isoformat(),
                    "未授权", object_ref, quantity=1, description="提交一件技术方案专利申请，尚未取得授权。"))
            else:
                match = next((item for item in pending if item[1] <= when), None)
                if match is None:
                    serial += 1
                    object_ref = f"PATENT-{basic['enterprise_id']}-{serial:03d}"
                    application_date = max(established + timedelta(days=7), when - timedelta(days=180))
                    result.append(_operation(config, basic, number, f"PATAPP{serial:03d}", "专利申请",
                        application_date.isoformat(), "未授权", object_ref, quantity=1,
                        description="已提交技术方案申请，申请日在授权前，原申请信息保留。"))
                else:
                    pending.remove(match)
                    object_ref = match[0]
                    application_date = match[1]
                token = object_ref.rsplit("-", 1)[1]
                result.append(_operation(config, basic, number, f"PATAPP{token}", "专利申请",
                    application_date.isoformat(), "已授权", object_ref, when.isoformat(), 2, quantity=1,
                    description="前期申请已获得后续授权，原申请事实及发生日期保持留存。"))
                result.append(_operation(config, basic, number, f"PATGRANT{token}", "专利授权", when.isoformat(),
                    "已授权", object_ref, quantity=1, description="该技术方案已获得授权，授权与原申请对应同一件专利。"))
    if cohort and not patent_sources:
        object_ref = f"PATENT-{basic['enterprise_id']}-901"
        result.append(_operation(config, basic, number, "PATAPP901", "专利申请", "2025-11-15",
                                 "未授权", object_ref, quantity=1, description="就企业内部工艺改进提交一件技术方案申请。"))
        result.append(_operation(config, basic, number, "PATAPP901", "专利申请", "2025-11-15",
                                 "已授权", object_ref, "2026-02-20", 2, quantity=1,
                                 description="此前提交的工艺技术方案申请已获得授权，申请信息继续保留。"))
        result.append(_operation(config, basic, number, "PATGRANT901", "专利授权", "2026-02-20",
                                 "已授权", object_ref, quantity=1, description="前期提交的工艺技术方案已获得授权。"))
    return result


def _debt_ledgers(events):
    contracts = {}
    for row in events:
        if row["event_type"] == "银行借款":
            contracts.setdefault(row["business_object_ref"], []).append(row)
    ledgers = []
    for contract, rows in sorted(contracts.items()):
        rows.sort(key=lambda row: (row["balance_as_of_date"], row["report_revision"]))
        original = decimal(rows[0]["amount"])
        previous = original
        repayments, checkpoints = [], []
        for row in rows:
            balance = decimal(row["outstanding_principal"])
            if balance > previous or balance < 0:
                raise ValueError(f"{contract} {row['balance_as_of_date']}:未偿本金变化缺少借款依据")
            if previous > balance:
                repayments.append({"payment_date": row["balance_as_of_date"],
                                   "principal_amount": money(previous - balance)})
            checkpoints.append({"balance_as_of_date": row["balance_as_of_date"],
                                "cumulative_principal_repaid": money(original - balance),
                                "outstanding_principal": money(balance),
                                "overdue_principal": row["overdue_principal"],
                                "record_ref": _ref("enterprise_operation_event", row)})
            previous = balance
        ledgers.append({"enterprise_id": rows[0]["enterprise_id"],
                        "business_object_ref": contract, "original_principal": money(original),
                        "checkpoints": checkpoints, "repayments": repayments,
                        "mechanism": "合同本金减去截至余额时点累计实还本金等于未偿本金；逾期本金为未偿本金的一部分，历史逾期发生额不与余额相加。"})
    return ledgers


def _active_person(people, when):
    applicable = [row for row in people if row["start_date"] <= when and
                  (row["end_date"] is None or row["end_date"] >= when)]
    return max(applicable, key=lambda row: (row["role"] == "法定代表人", row["report_revision"])) if applicable else None


def _negatives(config, templates, basic, people, number):
    result = []
    family, generation = (number - 1) % 40 + 1, (number - 1) // 40
    if family in _FAMILIES and not templates:
        templates = [{"event_type": "投诉", "event_date": "2026-01-20",
                      "case_category": "交付争议", "handling_result": "已撤回", "related_person_id": None}]
    if family in _FAMILIES and not any(row["event_type"] == "高管涉案" for row in templates):
        templates = [*templates, {"event_type": "高管涉案", "event_date": "2026-01-25",
                                 "case_category": "合同签署职责争议",
                                 "handling_result": "已结案" if variant_for(number, "A5") != "counterexample" else "调查中"}]
    for position, template in enumerate(templates, 1):
        event_date = max(basic["established_date"], template["event_date"])
        person = _active_person(people, event_date)
        related = None
        if template["event_type"] == "高管涉案":
            if person is None:
                event_date = max(event_date, min(row["start_date"] for row in people))
                person = _active_person(people, event_date)
            related = person["person_id"]
        ref = f"NEG-{basic['enterprise_id']}-{position:02d}"
        token = f"NE-{number:06d}-{position:02d}"
        category = template.get("case_category")
        description = (f"有关{category or template['event_type']}事项已由来源登记，涉及本企业的业务活动。")
        first = _row("enterprise_negative_event", negative_event_id=token + "-R1",
            enterprise_id=basic["enterprise_id"], related_person_id=related,
            business_event_ref=ref, business_object_ref=f"CASE-{basic['enterprise_id']}-{position:02d}",
            event_date=event_date, state_effective_at=dt(event_date), event_type=template["event_type"],
            case_category=category, description=description,
            handling_result="调查中" if template["event_type"] == "高管涉案" else "待处理")
        first.update(_metadata(config, "enterprise_negative_event", f"{ref}-STATE-{event_date}", event_date))
        result.append(first)
        last_date = min(date.fromisoformat(config["reference_date"]) - timedelta(days=12),
                        date.fromisoformat(event_date) + timedelta(days=70))
        if last_date <= date.fromisoformat(event_date):
            continue
        latest = deepcopy(first)
        latest["negative_event_id"] = token + "-R2"
        latest["state_effective_at"] = dt(last_date)
        latest["handling_result"] = template.get("handling_result") or "调查中"
        if family in _FAMILIES and variant_for(number, "A5") == "counterexample":
            latest["handling_result"] = "待处理"
        outcome = {
            "已撤回": "当事方确认补充交付或处置安排后主动撤回该事项，来源未记录违法认定。",
            "调解": "相关当事方已达成调解协议，交付、费用及责任分担按协议处理。",
            "已结案": ("调查核对了本企业任职人员的合同签署职责，未形成个人违法认定，事项已处理完毕。"
                         if template["event_type"] == "高管涉案" else "相关处置已完成，当事方已确认处理结果，事项已结案。"),
            "调查中": "相关事实、职责及处理依据仍在调查，尚无最终认定。",
            "待处理": "来源尚未记录最终处置结论，事项仍待办理。",
            "其他": ("来源已完成相关人员职责与事实核查，未形成违法认定，结论按其他处理方式登记。"
                       if template["event_type"] == "高管涉案" else
                       "企业已完成补交付、退款或现场处置安排，当事方确认事项处理完毕，来源按其他处理方式登记。"),
        }.get(latest["handling_result"])
        if outcome is None:
            raise ValueError(f"{basic['enterprise_id']} {ref}:未定义处理结果的客观业务结论")
        latest["description"] = f"有关{category or template['event_type']}事项的处理记录为{latest['handling_result']}。{outcome}"
        latest.update(_metadata(config, "enterprise_negative_event", f"{ref}-STATE-{last_date}", last_date, 1))
        result.append(latest)
    return result


def _coverage(config, basic, number):
    rows = []
    start = max(config["window_start"], basic["established_date"])
    source = config.get("generation", {}).get("source_system", "企业报送系统")
    for table in TABLES:
        if table == "enterprise_data_coverage":
            continue
        rows.append(_row("enterprise_data_coverage", coverage_id=f"CV-{number:06d}-{len(rows) + 1:02d}",
            enterprise_id=basic["enterprise_id"], table_name=table, source_system=source,
            coverage_start=start, coverage_end=config["window_end"], coverage_status="完整",
            last_synced_at=config["knowledge_cutoff"]))
    return rows


def _classified_features(features, snapshots):
    """Describe final outcomes without altering their business mechanisms."""
    canonical = {}
    for row in sorted(snapshots, key=lambda item: (item["as_of_date"], item["report_revision"])):
        canonical[row["as_of_date"]] = row
    ordered = [canonical[day] for day in sorted(canonical)]
    first, final = ordered[0], ordered[-1]
    first_assets, final_assets = decimal(first["total_assets"]), decimal(final["total_assets"])
    if first_assets <= 0 or final_assets <= 0:
        raise ValueError(f"{final['enterprise_id']}: final leverage classification requires positive asset stocks")
    result = list(features)
    if (decimal(final["total_liabilities"]) / final_assets >
            decimal(first["total_liabilities"]) / first_assets):
        result.append("leverage_increase")
    if decimal(final["total_liabilities"]) > final_assets:
        result.append("negative_owners_equity")
    return result


def generate_facts(config):
    """Return seven unpolluted observed tables plus private planning artifacts."""
    source = read_json(config["source_data"])
    if not isinstance(source, list) or len(source) != 40:
        raise ValueError("基础事实生成要求冻结的40家企业六表数组")
    source = sorted(source, key=lambda item: item["enterprise_basic"][0]["enterprise_id"])
    for enterprise in source:
        for table, identity in (("enterprise_person", "person_id"),
                                ("enterprise_operation_event", "operation_event_id"),
                                ("enterprise_negative_event", "negative_event_id"),
                                ("enterprise_operation_monthly", "stat_month")):
            enterprise[table].sort(key=lambda row: row[identity])
    profile_defs = _profile_definitions(read_json(config["source_profiles"]))
    definitions = read_json(config["scenario_definitions"])
    check_definitions(definitions)
    count = config.get("enterprise_count", 120)
    if count != 120:
        raise ValueError("本版事实生成固定为120家企业")
    tables = {table: [] for table in TABLES}
    profiles, enterprise_features = [], {}
    economic_processes, debt_ledgers, balance_processes = [], [], []
    canonical_months = 0
    generation_errors = []
    for number in range(1, count + 1):
        family = (number - 1) % 40 + 1
        anchor = source[family - 1]
        profile = profile_defs.get(f"P{family:03d}")
        if profile is None:
            raise ValueError(f"缺少P{family:03d}的源画像定义")
        try:
            basic = _basic(config, anchor["enterprise_basic"][0], number, profile)
            basic["main_business_description"] += "本企业月度税款报送范围为所得税预缴现金支付，损益所得税费用按当期计提。"
            features = feature_plan(number, variant_for)
            opening = opening_stocks(config, anchor["enterprise_operation_profile"][0], basic, number, features)
            people = _persons(config, anchor["enterprise_person"], basic, number)
            funding_outline = [{"as_of_date": opening["opening_stock_date"], "report_revision": 1,
                                "total_assets": opening["opening_total_assets"],
                                "total_liabilities": opening["opening_total_liabilities"]}]
            operations = _operations(config, anchor["enterprise_operation_event"], basic, number, funding_outline)
            enterprise_debt = _debt_ledgers(operations)
            debt_ledgers.extend(enterprise_debt)
            monthly, processes, features, opening = _monthly_values(
                config, anchor["enterprise_operation_monthly"], anchor["enterprise_operation_profile"][0],
                basic, profile, number, enterprise_debt, operations)
            economic_processes.extend(processes)
            if any(feature in features for feature in ("continuous_loss", "two_month_loss_then_recovery",
                                                       "three_month_continuous_loss", "isolated_loss_without_continuity")):
                basic["main_business_description"] += ("部分已交付商品批次按质量处置协议开展无偿替换、逆向物流及复检，正常销售与后续处理分别记录。" if basic["industry"] == "批发零售" else
                                                       "部分批次开展返工和复验，既有人员与预留外包产能按合同安排，收入按实际验收确认。")
            if "revenue_contraction" in features:
                basic["main_business_description"] += "部分客户调整交付验收安排，人员和场地合同在调整期间继续履行。"
            if "business_exit_and_asset_review" in features:
                basic["main_business_description"] += "退出非核心业务后保留最低场地及供应承诺，对设备用途和可回收性开展复核。"
                if decimal(opening["pre_window_owner_distribution"]) > 0:
                    basic["main_business_description"] += "股东按此前审议的累计留存收益安排执行部分利润分配。"
            canonical_months += len(monthly)
            snapshots, stock_processes = snapshot_stocks(
                config, anchor["enterprise_operation_profile"][0], basic, number, features,
                monthly, processes, enterprise_debt, operations, opening, _row, _metadata)
            balance_processes.extend(stock_processes)
            features = _classified_features(features, snapshots)
            for process in processes:
                process["business_features"] = list(features)
                process["business_feature_scope"] = "enterprise_window_plan_and_final_stock_classification"
            negatives = _negatives(config, anchor["enterprise_negative_event"], basic, people, number)
            values = {
                "enterprise_basic": [basic], "enterprise_person": people,
                "enterprise_operation_profile": snapshots,
                "enterprise_operation_monthly": _monthly_reports(config, monthly, number, processes),
                "enterprise_operation_event": operations, "enterprise_negative_event": negatives,
                "enterprise_data_coverage": _coverage(config, basic, number),
            }
            for table, rows in values.items():
                tables[table].extend(rows)
            profiles.append({
                "enterprise_id": basic["enterprise_id"], "anchor_enterprise_id": anchor["enterprise_basic"][0]["enterprise_id"],
                "profile_type_id": profile["profile_type_id"], "generation": (number - 1) // 40,
                "industry": basic["industry"], "scale": profile["scale"],
                "development_stage": profile["development_stage"], "operating_trend": profile["operating_trend"],
                "business_model": profile["business_model"], "business_features": features,
                "operating_characteristics": profile["operating_characteristics"],
                "anchor_business_variation": profile["source_business_variation"],
                "current_business_description": basic["main_business_description"],
                "management_metrics": {
                    "latest_employee_count": monthly[-1]["employee_count"],
                    "latest_month_revenue": monthly[-1]["operating_revenue"],
                    "window_revenue": money(sum(decimal(row["operating_revenue"]) for row in monthly)),
                    "window_net_profit": money(sum(decimal(row["net_profit"]) for row in monthly)),
                    "loss_months": [row["stat_month"] for row in monthly if decimal(row["net_profit"]) < 0],
                    "current_assets": snapshots[-1]["total_assets"],
                    "current_liabilities": snapshots[-1]["total_liabilities"],
                },
                "window_month_count": len(monthly), "partial_periods": [
                    row["stat_month"] for row in monthly if row["period_start"] != row["stat_month"]],
            })
            enterprise_features[number] = features
        except ValueError as exc:
            generation_errors.append(f"SYN-E{number:06d}/P{family:03d}: {exc}")
    if generation_errors:
        raise ValueError("离线构建发现商业计划冲突，完整120家产物未发布：\n" + "\n".join(generation_errors))
    scenarios = build_fact_scenarios(tables, enterprise_features, definitions)
    if canonical_months > 2875:
        raise ValueError(f"成立条件下的事实月数超过2875上限：{canonical_months}；检查保留成立日期和月覆盖")
    feature_enterprises = {
        feature: [f"SYN-E{number:06d}" for number, features in enterprise_features.items() if feature in features]
        for feature in sorted({feature for features in enterprise_features.values() for feature in features})
    }
    facts = {
        "reference_date": config["reference_date"], "knowledge_cutoff": config["knowledge_cutoff"],
        "window_start": config["window_start"], "window_end": config["window_end"],
        "canonical_month_count": canonical_months,
        "canonical_snapshot_count": count * len(config.get("profile_dates", _PROFILE_DATES)),
        "feature_enterprises": feature_enterprises,
        "derived_feature_scopes": {
            "leverage_increase": "最后规范快照的负债/资产比例高于最初规范快照；不影响预先选定的业务过程。",
            "negative_owners_equity": "最后规范快照负债大于资产；是真实权益负值，不是观测污染。",
            "business_exit_and_asset_review": "预先选定的业务退出、最低产能承诺和资产用途/可回收性复核机制；不预设最终负债率或权益符号。",
        },
        "layers": ["preselected_operating_and_funding_drivers", "business_base_and_genuine_risk", "lawful_reporting_history"],
        "truth_semantics": "truth_tables与base_observations均保留合法报送版本和历史状态；规范业务身份、版本与时间字段确定具体口径，非所有物理行简单相加。",
        "amount_encoding": "所有业务金额为人民币元的两位小数字符串；观测单位转换由独立污染/表示步骤进行。",
        "no_history_before_establishment": True,
        "notes": [
            "源40家身份及画像保留，指定业务场景可在新版重构原有数值；原冻结文件不修改。",
            "成立当月只生成成立日起的部分经营期间，成立前没有经营月报。",
            "已确认零税款、真实同值月份、合法修订、差异规模比例均在事实层生成。",
            "异常经营事实不作为记录污染；污染操作在之后独立执行。",
            "画像、场景与事实说明仅用于评测，业务可查询范围只有七表。",
        ],
    }
    return {"tables": tables, "truth_tables": deepcopy(tables), "profiles": profiles,
            "scenarios": scenarios, "facts": facts, "economic_processes": economic_processes,
            "debt_ledgers": debt_ledgers, "balance_processes": balance_processes}
