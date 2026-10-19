"""Production invariants for the controlled 120-enterprise dataset.

This module reads in-memory data only.  It does not import a database driver,
connect to a database, run a model, or execute SQL.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import re
from copy import deepcopy
from datetime import date, datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Mapping

from .schema import COLUMN_TYPES, MONEY_FIELDS, NULLABLE_COLUMNS, PRIMARY_KEYS, TABLE_COLUMNS, TABLES


CATEGORIES = ("A1", "A2", "A4", "A5", "B2", "B3", "C1", "C2", "C4", "C6")
OPERATIONS = {
    "duplicate_submission", "missing_field", "missing_month", "delayed_report",
    "unit_representation", "cross_field_conflict",
}
GROUP_COUNTS = {"none": 30, "single": 60, "double": 30}
OLD_GROUP_COUNTS = {"none": 10, "single": 20, "double": 10}
NEW_GROUP_COUNTS = {"none": 20, "single": 40, "double": 20}
EXPECTED_IDS = {f"SYN-E{number:06d}" for number in range(1, 121)}
RESPONSE_TYPES = {"answer", "clarify", "insufficient"}
ROLE_PHASES = {"typical": "L1", "transfer": "L2", "counterexample": "L3"}


def _reviewed_contract(config: Mapping[str, Any]) -> bool:
    """Keep archived candidates readable under their original process contract."""
    version = str(config.get("baseline_id", "")).rsplit("_v", 1)[-1]
    return tuple(int(part) for part in version.split(".")) >= (1, 0, 2)


def _cents(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _key(table: str, row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row[field] for field in PRIMARY_KEYS[table])


def _day(value: Any, label: str) -> date:
    try:
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        return date.fromisoformat(str(value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label}: invalid date {value!r}") from error


def _instant(value: Any, label: str) -> datetime:
    try:
        if isinstance(value, datetime):
            parsed = value
        elif isinstance(value, date):
            parsed = datetime.combine(value, datetime.min.time())
        else:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        _require(parsed.tzinfo is None, f"{label}: dataset timestamps must use the documented local time without an offset")
        return parsed
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label}: invalid timestamp {value!r}") from error


def _decimal(value: Any, label: str, scale: int) -> Decimal | None:
    if value is None:
        return None
    _require(isinstance(value, (str, Decimal)), f"{label}: money must be exchanged as a decimal string")
    if isinstance(value, str):
        _require(re.fullmatch(r"-?\d+(?:\.\d+)?", value) is not None, f"{label}: invalid decimal string")
    try:
        result = value if isinstance(value, Decimal) else Decimal(value)
    except (TypeError, ValueError, ArithmeticError) as error:
        raise ValueError(f"{label}: invalid decimal") from error
    _require(result.is_finite(), f"{label}: nonfinite money")
    normalized = result.normalize() if result else Decimal(0)
    _require(normalized.as_tuple().exponent >= -scale, f"{label}: exceeds {scale} fractional digits")
    integer_digits = max(len(result.as_tuple().digits) + result.as_tuple().exponent, 0)
    _require(integer_digits <= 16, f"{label}: exceeds monetary storage range")
    return result


def _money(row: Mapping[str, Any], field: str, label: str) -> Decimal | None:
    unit = row.get("amount_unit", "元")
    _require(unit in {"元", "万元"}, f"{label}: unsupported amount_unit {unit!r}")
    value = _decimal(row.get(field), label, 6 if unit == "万元" else 2)
    return None if value is None else value * (Decimal(10000) if unit == "万元" else Decimal(1))


def _maps(tables: Mapping[str, Any], label: str) -> dict[str, dict[tuple[Any, ...], dict[str, Any]]]:
    _require(set(tables) == set(TABLES), f"{label}: table set differs from the V3.0 schema")
    result: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = {}
    for table in TABLES:
        rows = tables[table]
        _require(isinstance(rows, list), f"{label}.{table}: expected an array")
        mapped: dict[tuple[Any, ...], dict[str, Any]] = {}
        for index, row in enumerate(rows):
            _require(isinstance(row, dict), f"{label}.{table}[{index}]: expected an object")
            _require(set(row) == set(TABLE_COLUMNS[table]), f"{label}.{table}[{index}]: incorrect fields")
            for field, value in row.items():
                _require(value is not None or field in NULLABLE_COLUMNS[table], f"{label}.{table}[{index}].{field}: null in a required field")
                if value is None:
                    continue
                column_type = COLUMN_TYPES[table][field].upper()
                if column_type.startswith("VARCHAR"):
                    limit = int(re.search(r"\((\d+)\)", column_type).group(1))
                    _require(isinstance(value, str) and len(value) <= limit, f"{label}.{table}[{index}].{field}: invalid text/length")
                elif column_type == "TEXT":
                    _require(isinstance(value, str) and len(value.encode("utf-8")) <= 65535, f"{label}.{table}[{index}].{field}: invalid text")
                elif column_type == "DATE":
                    _day(value, f"{label}.{table}.{field}")
                elif column_type == "DATETIME":
                    _instant(value, f"{label}.{table}.{field}")
                elif "INT" in column_type:
                    _require(isinstance(value, int) and not isinstance(value, bool), f"{label}.{table}[{index}].{field}: expected integer")
                    bits = 8 if column_type.startswith("TINYINT") else 16 if column_type.startswith("SMALLINT") else 32
                    lower, upper = (0, 2 ** bits - 1) if "UNSIGNED" in column_type else (-2 ** (bits - 1), 2 ** (bits - 1) - 1)
                    _require(lower <= value <= upper, f"{label}.{table}[{index}].{field}: integer outside storage range")
            key = _key(table, row)
            _require(all(value is not None for value in key), f"{label}.{table}: null primary key")
            _require(key not in mapped, f"{label}.{table}: duplicate technical primary key {key}")
            for field in MONEY_FIELDS.get(table, ()):
                _money(row, field, f"{label}.{table}{key}.{field}")
            if "report_revision" in row:
                _require(row["report_revision"] >= 1 and row["report_status"] in {"正式", "撤回"}, f"{label}.{table}: invalid report metadata")
                _require(_instant(row["reported_at"], f"{label}.{table}.reported_at") <= _instant(row["ingested_at"], f"{label}.{table}.ingested_at"), f"{label}.{table}: reporting time exceeds ingestion time")
                _require(bool(row["source_system"].strip()) and bool(row["source_record_ref"].strip()), f"{label}.{table}: empty source identity")
            for flag in ("is_current", "has_wechat_account"):
                if flag in row:
                    _require(row[flag] in (None, 0, 1), f"{label}.{table}: invalid {flag}")
            if table == "enterprise_person":
                _require(row["end_date"] is None or _day(row["start_date"], "start_date") <= _day(row["end_date"], "end_date"), f"{label}: reversed appointment interval")
            elif table == "enterprise_operation_monthly":
                month, start, end = (_day(row[field], field) for field in ("stat_month", "period_start", "period_end"))
                _require(month.day == 1 and start <= end and (start.year, start.month) == (month.year, month.month) == (end.year, end.month), f"{label}: invalid monthly coverage interval")
                for field in ("employee_count", "social_security_count"):
                    _require(row[field] is None or row[field] >= 0, f"{label}: negative {field}")
            elif table == "enterprise_basic":
                _require(_day(row["established_date"], "established_date") <= _day(row["as_of_date"], "as_of_date") and _day(row["registration_status_effective_date"], "registration_status_effective_date") <= _day(row["as_of_date"], "as_of_date"), f"{label}: invalid registration chronology")
            elif table == "enterprise_data_coverage":
                _require(row["table_name"] in set(TABLES) - {"enterprise_data_coverage"} and row["coverage_status"] in {"完整", "部分", "未采集", "不适用"}, f"{label}: invalid coverage domain/status")
                _require(_day(row["coverage_start"], "coverage_start") <= _day(row["coverage_end"], "coverage_end"), f"{label}: reversed coverage interval")
            mapped[key] = row
        result[table] = mapped
    return result


def _identity_and_relations(tables: Mapping[str, list[dict[str, Any]]], label: str) -> None:
    basic = tables["enterprise_basic"]
    ids = [row["enterprise_id"] for row in basic]
    _require(len(ids) == 120 and set(ids) == EXPECTED_IDS, f"{label}: expected the complete 120-enterprise identity set")
    industries = Counter(row["industry"] for row in basic)
    _require(len(industries) == 8 and set(industries.values()) == {15}, f"{label}: expected eight industries with 15 enterprises each")
    for field in ("synthetic_credit_code",):
        values = [row[field] for row in basic]
        _require(len(values) == len(set(values)), f"{label}: duplicate {field}")
    people = {row["person_id"]: row for row in tables["enterprise_person"]}
    for table in TABLES:
        for row in tables[table]:
            if "enterprise_id" in row:
                _require(row["enterprise_id"] in EXPECTED_IDS, f"{label}.{table}: unknown enterprise")
    for row in tables["enterprise_negative_event"]:
        person_id = row.get("related_person_id")
        if person_id is not None:
            _require(person_id in people and people[person_id]["enterprise_id"] == row["enterprise_id"], f"{label}: negative-event person is not from the same enterprise")
        if label == "truth":
            _require(row["event_type"] != "高管涉案" or person_id is not None, f"{label}: executive case has no person")


def _normal_facts(tables: Mapping[str, list[dict[str, Any]]], config: Mapping[str, Any]) -> None:
    basic = {row["enterprise_id"]: row for row in tables["enterprise_basic"]}
    reference = _day(config.get("reference_date", "2026-08-31"), "reference_date")
    knowledge_cutoff = _instant(config.get("knowledge_cutoff", "2026-09-15 23:59:59"), "knowledge_cutoff")
    start_month = _day(config.get("window_start", "2024-09-01"), "window_start").replace(day=1)
    final_month = _day(config.get("window_end", "2026-08-31"), "window_end").replace(day=1)
    expected_months = set()
    month_cursor = start_month
    while month_cursor <= final_month:
        expected_months.add(month_cursor.isoformat())
        month_cursor = date(month_cursor.year + (month_cursor.month == 12), month_cursor.month % 12 + 1, 1)
    for row in basic.values():
        _require(date(1990, 1, 1) <= _day(row["established_date"], "established_date") <= reference, "truth: invalid enterprise establishment date")
        capital = _money(row, "registered_capital", "registered_capital")
        _require(capital is not None and capital >= 0, "truth: missing or negative registered capital")
    for row in tables["enterprise_person"]:
        start = _day(row["start_date"], "person.start_date")
        end = None if row["end_date"] is None else _day(row["end_date"], "person.end_date")
        _require(_day(basic[row["enterprise_id"]]["established_date"], "established_date") <= start <= reference, "truth: invalid appointment date")
        _require(end is None or start <= end <= reference, "truth: invalid appointment end date")
        _require((row["is_current"] == 1 and end is None) or (row["is_current"] == 0 and end is not None), "truth: inconsistent employment status")
    monthly = defaultdict(set)
    for row in tables["enterprise_operation_monthly"]:
        monthly[row["enterprise_id"]].add(str(row["stat_month"])[:10])
        start = _day(row["period_start"], "period_start")
        end = _day(row["period_end"], "period_end")
        month = _day(row["stat_month"], "stat_month")
        _require(start <= end and start.year == month.year and start.month == month.month and end.year == month.year and end.month == month.month, "truth: monthly reporting period differs from stat_month")
        _require(month.day == 1 and start >= _day(basic[row["enterprise_id"]]["established_date"], "established_date"), "truth: monthly period starts before enterprise establishment")
        main = _money(row, "main_business_revenue", "main_business_revenue")
        revenue = _money(row, "operating_revenue", "operating_revenue")
        purchase = _money(row, "invoice_purchase_amount", "invoice_purchase_amount")
        utility = _money(row, "utility_invoice_amount", "utility_invoice_amount")
        travel = _money(row, "travel_hospitality_invoice_amount", "travel_hospitality_invoice_amount")
        _require(None not in (main, revenue, purchase, utility, travel), "truth: missing applicable monthly monetary fact")
        _require(main <= revenue, "truth: main business revenue exceeds operating revenue")
        _require(utility + travel <= purchase, "truth: invoice subitems exceed the purchase total")
        for field in MONEY_FIELDS.get("enterprise_operation_monthly", ()):
            value = _money(row, field, field)
            _require(value is not None, f"truth: missing monthly {field}")
            if field not in {"total_profit", "net_profit"}:
                _require(value >= 0, f"truth: negative monthly {field}")
        _require(row["employee_count"] >= 0 and row["social_security_count"] >= 0, "truth: negative headcount")
    _require(set(monthly) == EXPECTED_IDS, "truth: missing enterprise monthly coverage")
    for enterprise_id, months in monthly.items():
        established_month = _day(basic[enterprise_id]["established_date"], "established_date").replace(day=1).isoformat()
        expected_for_enterprise = {month for month in expected_months if month >= established_month}
        _require(months == expected_for_enterprise, f"truth: incomplete monthly coverage for {enterprise_id}")
    for row in tables["enterprise_operation_profile"]:
        _require(row["has_wechat_account"] in (0, 1), "truth: invalid wechat flag")
        _require((row["has_wechat_account"] == 0 and row["wechat_content_summary"] is None) or (row["has_wechat_account"] == 1 and row["wechat_content_summary"] is not None), "truth: inconsistent wechat facts")
        for field in ("total_assets", "total_liabilities"):
            value = _money(row, field, field)
            _require(value is not None and value >= 0, f"truth: invalid {field}")
        # Liabilities above assets and losses are genuine possible facts, not errors.
    for table in ("enterprise_operation_event", "enterprise_negative_event"):
        for row in tables[table]:
            occurred = _day(row["event_date"], f"{table}.event_date")
            _require(occurred <= reference, "truth: future event")
            if table == "enterprise_operation_event" or row.get("related_person_id") is None:
                _require(occurred >= _day(basic[row["enterprise_id"]]["established_date"], "established_date"), "truth: enterprise event before establishment")
            effective = _instant(row["state_effective_at"], "state_effective_at")
            _require(occurred <= effective.date() <= reference, "truth: invalid event status time")
            _require(bool(row.get("business_event_ref")) and bool(row.get("business_object_ref")), "truth: missing event or matter identity")
            if table == "enterprise_operation_event":
                kind = row["event_type"]
                _require(bool(row.get("business_event_ref")) and bool(row.get("business_object_ref")), "truth: missing event or matter identity")
                if kind in {"专利申请", "专利授权"}:
                    _require(row["quantity"] is not None and row["quantity"] > 0 and row["amount"] is None and row["valuation"] is None, "truth: invalid patent-specific fields")
                else:
                    amount = _money(row, "amount", "amount")
                    _require(amount is not None and amount >= 0, "truth: invalid financial event amount")
                outstanding = _money(row, "outstanding_principal", "outstanding_principal") if "outstanding_principal" in row else None
                overdue = _money(row, "overdue_principal", "overdue_principal") if "overdue_principal" in row else None
                if outstanding is not None:
                    _require(outstanding >= 0 and (overdue is None or 0 <= overdue <= outstanding), "truth: invalid principal snapshot")
                _require(overdue is None or outstanding is not None, "truth: overdue snapshot has no outstanding principal")
                balance_date = row.get("balance_as_of_date")
                _require((balance_date is not None) == (outstanding is not None), "truth: principal snapshot/date mismatch")
                if balance_date is not None:
                    _require(occurred <= _day(balance_date, "balance_as_of_date") <= reference, "truth: invalid principal snapshot date")
    for table in TABLES:
        source_identities = {}
        for row in tables[table]:
            if "report_revision" in row:
                _require(isinstance(row["report_revision"], int) and not isinstance(row["report_revision"], bool) and row["report_revision"] >= 1, "truth: invalid report revision")
                _require(row["report_status"] in {"正式", "撤回"}, "truth: invalid report status")
            if "reported_at" in row:
                reported = _instant(row["reported_at"], "reported_at")
                ingested = _instant(row["ingested_at"], "ingested_at")
                _require(reported <= ingested <= knowledge_cutoff, "truth: invalid source reporting chronology")
                source_identity = (row["enterprise_id"], row["source_system"], row["source_record_ref"])
                business_fields = {
                    "enterprise_person": ("person_entity_id", "assignment_id"),
                    "enterprise_operation_profile": ("as_of_date",),
                    "enterprise_operation_monthly": ("stat_month", "period_start", "period_end"),
                    "enterprise_operation_event": ("business_event_ref", "business_object_ref", "event_date"),
                    "enterprise_negative_event": ("business_event_ref", "business_object_ref", "event_date"),
                }[table]
                business_identity = tuple(row[field] for field in business_fields)
                _require(source_identity not in source_identities or source_identities[source_identity] == business_identity, f"truth.{table}: source revision scope crosses business facts or statistical times")
                source_identities[source_identity] = business_identity
    for table in ("enterprise_operation_event", "enterprise_negative_event"):
        identity = {}
        for row in tables[table]:
            ref = row["business_event_ref"]
            fields = (row["enterprise_id"], row["business_object_ref"], row["event_type"], str(row["event_date"])[:10])
            _require(ref not in identity or identity[ref] == fields, "truth: event identity describes multiple different real events")
            identity[ref] = fields
    for row in tables["enterprise_data_coverage"]:
        _require(row["table_name"] in set(TABLES) - {"enterprise_data_coverage"}, "coverage: unknown business table")
        _require(row["coverage_status"] in {"完整", "部分", "未采集", "不适用"}, "coverage: invalid status")
        _require(_day(row["coverage_start"], "coverage_start") <= _day(row["coverage_end"], "coverage_end"), "coverage: reversed interval")
        _require(_instant(row["last_synced_at"], "last_synced_at") <= knowledge_cutoff, "coverage: timestamp beyond available knowledge")


def _assignment_checks(assignments: Mapping[str, Any]) -> dict[str, int]:
    _require(set(assignments) == EXPECTED_IDS, "assignments: incomplete enterprise set")
    groups, old, new = Counter(), Counter(), Counter()
    for enterprise_id, assignment in assignments.items():
        _require(isinstance(assignment, dict), "assignments: expected objects")
        group, classes = assignment.get("group"), assignment.get("classes")
        _require(group in GROUP_COUNTS and isinstance(classes, list), "assignments: invalid group/classes")
        _require(len(classes) == {"none": 0, "single": 1, "double": 2}[group] and len(set(classes)) == len(classes), "assignments: incorrect class count")
        _require(set(classes) <= OPERATIONS, "assignments: unsupported  class")
        groups[group] += 1
        (old if int(enterprise_id[-6:]) <= 40 else new)[group] += 1
    _require(dict(groups) == GROUP_COUNTS and dict(old) == OLD_GROUP_COUNTS and dict(new) == NEW_GROUP_COUNTS, "assignments: expected 30/60/30 with old 10/20/10 and new 20/40/20")
    return dict(groups)


def _ledger_checks(tables: Mapping[str, Any], base: Mapping[str, Any], ledger: list[dict[str, Any]], assignments: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    original = _maps(base, "base_observations")
    replay = deepcopy(original)
    touched: set[tuple[str, tuple[Any, ...]]] = set()
    classes_by_row: dict[tuple[str, tuple[Any, ...]], set[str]] = defaultdict(set)
    actual_company_classes: dict[str, set[str]] = defaultdict(set)
    missing_months: dict[str, set[str]] = defaultdict(set)
    operation_counts = Counter()
    operation_enterprises = defaultdict(set)
    duplicates = 0
    for index, entry in enumerate(ledger):
        _require(isinstance(entry, dict), f"ledger[{index}]: expected an object")
        operation, table = entry.get("operation"), entry.get("table")
        _require(operation in OPERATIONS and table in TABLES, f"ledger[{index}]: unsupported operation/table")
        before, after, source_key = entry.get("before"), entry.get("after"), entry.get("source_key")
        _require(isinstance(before, dict) and isinstance(source_key, dict), f"ledger[{index}]: missing original record")
        _require(set(source_key) == set(PRIMARY_KEYS[table]), f"ledger[{index}]: incomplete source primary key")
        key = tuple(source_key[field] for field in PRIMARY_KEYS[table])
        _require(_key(table, before) == key and replay[table].get(key) == before, f"ledger[{index}]: original record does not match the current replay state")
        enterprise_id = before.get("enterprise_id")
        _require(enterprise_id in assignments and operation in assignments[enterprise_id]["classes"], f"ledger[{index}]:  not assigned to this enterprise")
        _require(entry.get("expected_behavior") in {"answer", "clarify", "insufficient"}, f"ledger[{index}]: missing expected response type")
        _require(isinstance(entry.get("public_basis"), str) and bool(entry["public_basis"].strip()), f"ledger[{index}]: no observable basis")
        _require(isinstance(entry.get("nature"), str) and bool(entry["nature"].strip()), f"ledger[{index}]: missing  nature")
        actual_company_classes[enterprise_id].add(operation)
        operation_counts[operation] += 1
        operation_enterprises[operation].add(enterprise_id)
        if operation == "duplicate_submission":
            _require(isinstance(after, dict), f"ledger[{index}]: duplicate has no new record")
            _require(key in original[table], f"ledger[{index}]: recursively copied a derived observation")
            target_key = _key(table, after)
            _require(target_key != key and target_key not in replay[table], f"ledger[{index}]: duplicate reuses a technical primary key")
            _require(after.get("enterprise_id") == enterprise_id, f"ledger[{index}]: duplicate moved enterprise")
            _require(all(after[field] == value for field, value in before.items() if field not in PRIMARY_KEYS[table]), f"ledger[{index}]: duplicate changed nontechnical fields")
            for field in ("business_event_ref", "business_object_ref", "person_entity_id", "assignment_id"):
                if field in before:
                    _require(after[field] == before[field], f"ledger[{index}]: duplicate changed business identity")
            replay[table][target_key] = deepcopy(after)
            classes_by_row[(table, key)].add(operation)
            classes_by_row[(table, target_key)] = set(classes_by_row[(table, key)])
            duplicates += 1
        elif operation == "missing_month" and table == "enterprise_operation_monthly":
            _require(after is None, f"ledger[{index}]: invalid monthly omission")
            del replay[table][key]
            missing_months[enterprise_id].add(str(before["stat_month"])[:10])
            touched.add((table, key))
            classes_by_row[(table, key)].add(operation)
        else:
            _require(isinstance(after, dict) and _key(table, after) == key and after.get("enterprise_id") == enterprise_id, f"ledger[{index}]: mutation changed identity")
            _require(before != after, f"ledger[{index}]: no-op mutation")
            changed = {field for field in before if before[field] != after.get(field)}
            if operation == "missing_month":
                _require(table == "enterprise_data_coverage", f"ledger[{index}]: missing-month metadata update is not a coverage record")
                _require(changed == {"coverage_status"} and before["table_name"] == "enterprise_operation_monthly" and after["coverage_status"] == "部分", f"ledger[{index}]: invalid omitted-month coverage update")
            elif operation == "missing_field":
                _require(changed <= NULLABLE_COLUMNS[table] and all(before[field] is not None and after[field] is None for field in changed), f"ledger[{index}]: missing-field mutation changed more than nullable values")
            elif operation == "unit_representation":
                _require(changed <= set(MONEY_FIELDS.get(table, ())) | {"amount_unit"}, f"ledger[{index}]: unit conversion changed unrelated fields")
                _require(before.get("amount_unit") != after.get("amount_unit"), f"ledger[{index}]: unit representation did not change unit")
                for field in MONEY_FIELDS.get(table, ()):
                    _require(_money(before, field, field) == _money(after, field, field), f"ledger[{index}]: unit conversion changed the underlying money")
            elif operation == "delayed_report":
                _require(changed <= {"reported_at", "ingested_at"} and "ingested_at" in changed, f"ledger[{index}]: delay changed business values")
                _require(_instant(after["ingested_at"], "ingested_at") > _instant(before["ingested_at"], "ingested_at"), f"ledger[{index}]: delayed record was not delayed")
            elif operation == "cross_field_conflict":
                allowed = {
                    "enterprise_operation_monthly": {"main_business_revenue", "invoice_purchase_amount", "utility_invoice_amount", "travel_hospitality_invoice_amount"},
                    "enterprise_person": {"is_current", "end_date"},
                }
                _require(table in allowed and changed <= allowed[table], f"ledger[{index}]: undefined cross-field conflict")
                if table == "enterprise_operation_monthly":
                    main, revenue, purchase, utility, travel = (_money(after, field, field) for field in ("main_business_revenue", "operating_revenue", "invoice_purchase_amount", "utility_invoice_amount", "travel_hospitality_invoice_amount"))
                    _require((main is not None and revenue is not None and main > revenue) or (None not in (purchase, utility, travel) and utility + travel > purchase), f"ledger[{index}]: no defined revenue/invoice conflict")
                else:
                    if _reviewed_contract(config):
                        _require(after["is_current"] == 1 and after["end_date"] is not None and _day(after["end_date"], "end_date") < _day(config["reference_date"], "reference_date"), f"ledger[{index}]: appointment conflict is not independently observable at the business reference date")
                    else:
                        _require((after["is_current"] == 1 and after["end_date"] is not None) or (after["is_current"] == 0 and after["end_date"] is None), f"ledger[{index}]: no appointment flag/date conflict")
            replay[table][key] = deepcopy(after)
            if key in original[table]:
                touched.add((table, key))
            classes_by_row[(table, key)].add(operation)
        _require(all(len(classes) <= 2 for classes in classes_by_row.values()), "ledger: more than two classes on a record")
    for enterprise_id, assignment in assignments.items():
        _require(actual_company_classes.get(enterprise_id, set()) == set(assignment["classes"]), f": assigned classes not realized for {enterprise_id}")
    for enterprise_id, months in missing_months.items():
        maximum_run = int(config.get("pollution", {}).get("max_consecutive_missing_months", 2))
        ordered = sorted(_day(month, "missing month") for month in months)
        run = 0
        prior = None
        for month in ordered:
            ordinal = month.year * 12 + month.month
            run = run + 1 if prior is not None and ordinal == prior + 1 else 1
            _require(run <= maximum_run, f": too many consecutive omitted months for {enterprise_id}")
            prior = ordinal
        maximum_total = config.get("pollution", {}).get("max_missing_months_per_enterprise")
        if maximum_total is not None:
            _require(len(months) <= int(maximum_total), f": too many omitted months for {enterprise_id}")
    observed = _maps(tables, "observations")
    _require(observed == replay, ": observed tables differ from the exact ledger replay")
    for row in tables["enterprise_operation_monthly"]:
        _require(str(row["stat_month"])[:10] not in missing_months.get(row["enterprise_id"], set()), ": a claimed omitted month still has a source/version record")
    # Only modifiable report tables count toward this denominator; identities and
    # coverage cannot dilute the advertised contamination rate.
    report_tables = [table for table in TABLES if "report_revision" in TABLE_COLUMNS[table]]
    eligible_originals = sum(len(original[table]) for table in report_tables)
    affected_originals = sum(table in report_tables for table, _ in touched)
    maximum_fraction = Decimal(str(config.get("pollution", {}).get("max_modified_fraction", "0.15")))
    _require(Decimal(0) <= maximum_fraction <= Decimal("0.15"), ": configured contamination cap exceeds 15 percent")
    _require(eligible_originals > 0 and Decimal(affected_originals) <= Decimal(eligible_originals) * maximum_fraction, ": altered/omitted original observations exceed the configured cap")
    minimum_fields = int(config.get("pollution", {}).get("minimum_missing_field_enterprises", 12))
    minimum_months = int(config.get("pollution", {}).get("minimum_missing_month_enterprises", 12))
    _require(len(operation_enterprises["missing_field"]) >= minimum_fields and len(missing_months) >= minimum_months, ": insufficient enterprises with missing fields or omitted months")
    for enterprise_id, months in missing_months.items():
        coverage = [row for row in tables["enterprise_data_coverage"] if row["enterprise_id"] == enterprise_id and row["table_name"] == "enterprise_operation_monthly"]
        _require(coverage and all(row["coverage_status"] == "部分" for row in coverage), f"ledger: missing monthly coverage evidence for {enterprise_id}")
        for month in months:
            _require(any(_day(row["coverage_start"], "coverage_start") <= _day(month, "stat_month") <= _day(row["coverage_end"], "coverage_end") for row in coverage), f"ledger: omitted month outside documented coverage for {enterprise_id}")
    return {"operations": {operation: operation_counts[operation] for operation in sorted(OPERATIONS)}, "enterprises_per_operation": {name: len(operation_enterprises[name]) for name in sorted(OPERATIONS)}, "altered_or_omitted_originals": affected_originals, "eligible_original_observations": eligible_originals, "altered_or_omitted_ratio": str(Decimal(affected_originals) / Decimal(eligible_originals)), "added_duplicate_records": duplicates, "omitted_months": sum(map(len, missing_months.values())), "maximum_classes_per_record": max(map(len, classes_by_row.values()), default=0)}


def _context_checks(scenario: Mapping[str, Any], observed: Mapping[str, Any]) -> None:
    label = f"scenarios.{scenario.get('scenario_id')}.context_transition"
    context = scenario.get("context_transition")
    _require(isinstance(context, dict) and context.get("visibility") == "public_records_and_dictionary" and context.get("requires_hidden_truth") is False, f"{label}: missing publicly grounded scope transition")
    initial, followup = context.get("initial_scope"), context.get("followup_scope")
    _require(isinstance(initial, dict) and isinstance(followup, dict) and initial and followup, f"{label}: missing business scopes")
    expected_inherited = {field for field in set(initial) & set(followup) if initial[field] == followup[field]}
    expected_changed = {field for field in followup if field not in initial or initial[field] != followup[field]}
    expected_cancelled = set(initial) - set(followup)
    for field, expected in (("inherited_conditions", expected_inherited), ("changed_conditions", expected_changed), ("cancelled_conditions", expected_cancelled)):
        actual = context.get(field)
        _require(isinstance(actual, list) and len(actual) == len(set(actual)) and set(actual) == expected, f"{label}: {field} does not describe the actual scope change")
    public_refs = {(reference["table"], tuple(reference.get("key", reference.get("primary_key"))[field] for field in PRIMARY_KEYS[reference["table"]])) for reference in scenario["record_refs"]}
    witness_sets = []
    for scope, field in ((initial, "initial_record_refs"), (followup, "followup_record_refs")):
        refs = context.get(field)
        _require(isinstance(refs, list) and refs, f"{label}: {field} has no public-row witnesses")
        witnesses = set()
        for reference in refs:
            _require(isinstance(reference, dict) and reference.get("table") in TABLES, f"{label}: invalid context reference")
            table = reference["table"]
            primary_key = reference.get("key", reference.get("primary_key"))
            _require(isinstance(primary_key, dict) and set(primary_key) == set(PRIMARY_KEYS[table]), f"{label}: incomplete context key")
            key = tuple(primary_key[column] for column in PRIMARY_KEYS[table])
            _require((table, key) in public_refs and key in observed[table], f"{label}: context witness is not a declared public scenario reference")
            row = observed[table][key]
            _require(row["enterprise_id"] == scope.get("enterprise_id") == scenario["enterprise_id"], f"{label}: context witness crosses enterprise scope")
            if "period_start" in scope:
                _require(table == "enterprise_operation_monthly" and _day(row["period_start"], "period_start") >= _day(scope["period_start"], "scope.period_start") and _day(row["period_end"], "period_end") <= _day(scope["period_end"], "scope.period_end"), f"{label}: monthly witness is outside its declared period")
            if "event_type" in scope:
                _require(row.get("event_type") == scope["event_type"], f"{label}: state witness is outside its declared event type")
                candidates = [candidate for candidate in observed[table].values() if candidate["enterprise_id"] == row["enterprise_id"] and candidate.get("business_event_ref") == row.get("business_event_ref")]
                selection = scope.get("selection")
                if selection == "historical_occurrence":
                    _require(row["state_effective_at"] == min(candidate["state_effective_at"] for candidate in candidates), f"{label}: historical occurrence witness is not the initial business state")
                elif selection == "business_state_as_of":
                    business_end = datetime.combine(_day(scope.get("state_as_of"), "state_as_of"), datetime.max.time())
                    knowledge_end = _instant(scope.get("knowledge_cutoff"), "knowledge_cutoff")
                    candidates = [candidate for candidate in candidates if _instant(candidate["state_effective_at"], "state_effective_at") <= business_end and _instant(candidate["ingested_at"], "ingested_at") <= knowledge_end]
                    _require(candidates and row["state_effective_at"] == max(candidate["state_effective_at"] for candidate in candidates) and _instant(row["ingested_at"], "ingested_at") <= knowledge_end, f"{label}: current-state witness does not follow the declared business and knowledge times")
                else:
                    _require(selection == "state_history", f"{label}: undefined state selection")
                    if "state_effective_at" in scope:
                        _require(row["state_effective_at"] == scope["state_effective_at"], f"{label}: state witness does not meet the condition later cancelled")
            witnesses.add((table, key))
        witness_sets.append(witnesses)
    _require(witness_sets[0] != witness_sets[1], f"{label}: context change has no distinguishable public-row witnesses")


def _scenario_checks(scenarios: list[dict[str, Any]], truth: Mapping[str, Any], observed: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    _require(isinstance(scenarios, list), "scenarios: expected an array")
    counts = Counter()
    old, new = defaultdict(set), defaultdict(set)
    scenario_roles = Counter()
    category_roles = defaultdict(set)
    cohort_category_roles = defaultdict(set)
    genuine_enterprise_roles = defaultdict(set)
    response_counts = Counter()
    reviewed = _reviewed_contract(config)
    if reviewed:
        from .scenarios import _role_evidence, variant_for
        global_people = defaultdict(list)
        truth_by_enterprise = {table: defaultdict(list) for table in TABLES}
        for table, rows in truth.items():
            for row in rows.values():
                truth_by_enterprise[table][row["enterprise_id"]].append(row)
                if table == "enterprise_person":
                    global_people[row["person_entity_id"]].append(row)
    for scenario in scenarios:
        _require(isinstance(scenario, dict), "scenarios: expected objects")
        category, enterprise_id = scenario.get("category"), scenario.get("enterprise_id")
        _require(category in CATEGORIES and enterprise_id in EXPECTED_IDS, "scenarios: invalid category/enterprise")
        counts[category] += 1
        (old if int(enterprise_id[-6:]) <= 40 else new)[category].add(enterprise_id)
        roles = scenario.get("roles", [])
        if isinstance(roles, str):
            roles = [roles]
        if isinstance(roles, dict):
            roles = list(roles)
        _require(isinstance(roles, list), "scenarios: invalid roles")
        _require(roles and set(roles) <= {"typical", "transfer", "counterexample"}, "scenarios: undefined learning role")
        _require(isinstance(scenario.get("public_basis"), str) and bool(scenario["public_basis"].strip()), "scenarios: missing observable basis")
        _require(isinstance(scenario.get("expected_behavior"), str) and bool(scenario["expected_behavior"].strip()), "scenarios: missing expected response")
        response_type = scenario.get("response_type", scenario.get("expected_behavior") if not reviewed else None)
        if reviewed:
            _require(response_type in RESPONSE_TYPES, "scenarios: response_type must be answer, clarify or insufficient")
            _require(scenario["expected_behavior"] not in RESPONSE_TYPES, "scenarios: expected_behavior must explain the response rather than repeat its type")
            _require(isinstance(scenario.get("target_scope"), str) and bool(scenario["target_scope"].strip()), "scenarios: missing answerability scope")
            response_counts[response_type] += 1
        phases = scenario.get("learning_phases", [])
        _require(isinstance(phases, list) and phases and set(phases) <= {"L1", "L2", "L3"}, "scenarios: undefined learning phase")
        if reviewed:
            _require(set(phases) == {ROLE_PHASES[role] for role in roles}, "scenarios: learning role does not match its phase")
            if scenario.get("design_origin") == "genuine_business_fact":
                _require(scenario.get("role_assignment") == "family_category_rotation_v1" and len(roles) == 1, "scenarios: genuine fact must use the category-specific role contract")
                _require(roles[0] == variant_for(int(enterprise_id[-6:]), category), "scenarios: assigned role differs from the category-specific business variant")
                evidence = scenario.get("evidence")
                _require(isinstance(evidence, dict) and evidence.get("visibility") == "public_records_and_dictionary" and evidence.get("requires_hidden_truth") is False, "scenarios: genuine fact must have publicly observable evidence")
                _require(all(isinstance(evidence.get(field), list) and evidence[field] and all(isinstance(item, str) and bool(item.strip()) for item in evidence[field]) for field in ("observed_dimensions", "contrasting_scopes")), "scenarios: missing observable dimensions or contrasting business scopes")
                selected = {table: rows[enterprise_id] for table, rows in truth_by_enterprise.items()}
                _require(evidence.get("business_predicate") == _role_evidence(category, roles[0], enterprise_id, selected, global_people), "scenarios: claimed business predicate differs from its actual fact and role")
                cohort = "old" if int(enterprise_id[-6:]) <= 40 else "new"
                cohort_category_roles[(cohort, category)].update(roles)
                genuine_enterprise_roles[enterprise_id].update(roles)
        scenario_roles.update(set(roles))
        category_roles[category].update(roles)
        references = scenario.get("record_refs", [])
        _require(isinstance(references, list) and references, "scenarios: expected observable record references")
        related_enterprises = scenario.get("related_enterprise_ids", [])
        _require(isinstance(related_enterprises, list) and set(related_enterprises) <= EXPECTED_IDS, "scenarios: invalid related enterprise identity")
        permitted_enterprises = {enterprise_id, *related_enterprises}
        referenced_enterprises = set()
        for reference in references:
            _require(isinstance(reference, dict) and reference.get("table") in TABLES, "scenarios: invalid record reference")
            table = reference["table"]
            key_values = reference.get("key", reference.get("primary_key"))
            _require(isinstance(key_values, dict) and set(key_values) == set(PRIMARY_KEYS[table]), "scenarios: incomplete referenced key")
            key = tuple(key_values[field] for field in PRIMARY_KEYS[table])
            _require(key in observed[table] and observed[table][key].get("enterprise_id") in permitted_enterprises, f"scenarios: public reference missing or points to an undeclared enterprise: {table}{key}")
            referenced_enterprises.add(observed[table][key]["enterprise_id"])
            if scenario.get("design_origin") != "controlled_observation":
                _require(key in truth[table] and truth[table][key].get("enterprise_id") in permitted_enterprises, "scenarios: fact reference absent from truth")
            elif response_type == "answer":
                _require(scenario.get("operation") in {"duplicate_submission", "unit_representation", "delayed_report"}, "scenarios: hidden nonrecoverable truth cannot be required as an answer")
        _require(referenced_enterprises == permitted_enterprises, "scenarios: primary or related enterprise has no public reference evidence")
        if reviewed and category == "C6" and scenario.get("design_origin") == "genuine_business_fact":
            _context_checks(scenario, observed)
    required = config.get("coverage_requirements", {})
    for category in CATEGORIES:
        _require(len(old[category]) + len(new[category]) >= max(6, int(required.get("minimum_enterprises_per_category", 6))) and len(old[category]) >= max(2, int(required.get("minimum_old_per_category", 2))) and len(new[category]) >= max(2, int(required.get("minimum_new_per_category", 2))), f"scenarios: insufficient {category} coverage in existing/new enterprises")
        _require(category_roles[category] == {"typical", "transfer", "counterexample"}, f"scenarios: {category} lacks typical, transfer or counterexample behavior")
        if reviewed:
            for cohort in ("old", "new"):
                _require(cohort_category_roles[(cohort, category)] == set(ROLE_PHASES), f"scenarios: {category} genuine roles are not crossed within the {cohort} cohort")
    if reviewed:
        _require(genuine_enterprise_roles and all(roles == set(ROLE_PHASES) for roles in genuine_enterprise_roles.values()), "scenarios: participating enterprises must have category-specific typical, transfer and counterexample roles")
        _require(set(response_counts) == RESPONSE_TYPES, "scenarios: all three response types require observable support")
    return {"category_counts": dict(counts), "existing_enterprises_per_category": {key: len(value) for key, value in old.items()}, "new_enterprises_per_category": {key: len(value) for key, value in new.items()}, "role_counts": dict(scenario_roles), "roles_per_category": {category: sorted(category_roles[category]) for category in CATEGORIES}, "response_type_counts": dict(response_counts), "genuine_roles_by_cohort_category": {f"{cohort}:{category}": sorted(values) for (cohort, category), values in sorted(cohort_category_roles.items())}, "enterprises_with_category_specific_roles": len(genuine_enterprise_roles)}


def _public_edges(tables: Mapping[str, Any]) -> dict[str, set[str]]:
    zeros, equals, revisions, ratios, unknown = set(), set(), set(), set(), set()
    monthly_latest = {}
    for table, rows in tables.items():
        equal_values = defaultdict(lambda: defaultdict(set))
        revision_values = defaultdict(set)
        for row in rows:
            enterprise_id = row["enterprise_id"]
            for field in MONEY_FIELDS.get(table, ()):
                if _money(row, field, field) == 0:
                    zeros.add(enterprise_id)
            if "report_revision" in row:
                identity = (enterprise_id, row["source_system"], row["source_record_ref"])
                revision_values[identity].add(row["report_revision"])
            if table == "enterprise_operation_monthly":
                amount = _money(row, "operating_revenue", "operating_revenue")
                if amount is not None:
                    equal_values[enterprise_id][amount].add(row["stat_month"])
                if identity not in monthly_latest or row["report_revision"] > monthly_latest[identity]["report_revision"]:
                    monthly_latest[identity] = row
                if any(row[field] is None for field in ("employee_count", "social_security_count", *MONEY_FIELDS[table])):
                    unknown.add(enterprise_id)
            elif table == "enterprise_operation_event" and row["amount"] is not None:
                equal_values[enterprise_id][_money(row, "amount", "amount")].add(row["business_event_ref"])
            elif table == "enterprise_operation_profile":
                if any(row[field] is None for field in ("has_wechat_account", "total_assets", "total_liabilities")):
                    unknown.add(enterprise_id)
            elif table == "enterprise_person" and (row["age"] is None or row["household_region"] is None):
                unknown.add(enterprise_id)
        revisions.update(identity[0] for identity, values in revision_values.items() if len(values) >= 2)
        equals.update(enterprise_id for enterprise_id, amounts in equal_values.items() if any(len(identities) >= 2 for identities in amounts.values()))
    ratio_values = defaultdict(list)
    for row in monthly_latest.values():
        revenue, profit = _money(row, "operating_revenue", "operating_revenue"), _money(row, "net_profit", "net_profit")
        if row["report_status"] == "正式" and revenue is not None and revenue > 0 and profit is not None:
            ratio_values[row["enterprise_id"]].append((profit, revenue))
    for enterprise_id, amounts in ratio_values.items():
        if len(amounts) >= 2:
            mean = sum(profit / revenue for profit, revenue in amounts) / Decimal(len(amounts))
            combined = sum(profit for profit, _ in amounts) / sum(revenue for _, revenue in amounts)
            if abs(mean - combined) > Decimal("0.000000000001"):
                ratios.add(enterprise_id)
    return {"zero": zeros, "same_value": equals, "revision": revisions, "ratio": ratios, "unknown": unknown}


def _public_description_checks(tables: Mapping[str, Any], config: Mapping[str, Any]) -> int:
    """Reject target conclusions embedded in publicly queryable descriptions."""
    if not _reviewed_contract(config):
        return 0
    month_pair = r"\d{1,2}(?:月)?(?:[、，,及与和至到\-~]\d{1,2}(?:月)?)+"
    leaks = (
        re.compile(month_pair + r"(?:份)?[^。；;\n]{0,40}(?:收入|营业额)[^。；;\n]{0,15}(?:相同|相等|同额)"),
        re.compile(r"\d{1,2}(?:月|[、，,及与和至到\-~]\d{1,2}月)[^。；;\n]{0,35}(?:亏损|经营(?:恢复)?盈利|净利润(?:为负|转负))"),
        re.compile(r"(?:收入|营业额)[^。；;\n]{0,15}(?:相同|相等|同额)[^。；;\n]{0,30}" + month_pair),
    )
    checked = 0
    for table, rows in tables.items():
        text_fields = [field for field, column_type in COLUMN_TYPES[table].items() if column_type.upper() == "TEXT" or field in {"main_business_description", "business_scope"}]
        for row in rows:
            for field in text_fields:
                text = row.get(field)
                if text is not None:
                    _require(not any(pattern.search(text) for pattern in leaks), f"observations.{table}{_key(table, row)}.{field}: public description discloses a target monthly conclusion")
                    checked += 1
    return checked


def _management_profile_check(profile: Mapping[str, Any], truth: Mapping[str, Any], latest_months: Mapping[Any, Any]) -> None:
    enterprise_id = profile["enterprise_id"]
    label = f"profiles.{enterprise_id}"
    months = sorted((row for row in latest_months.values() if row["enterprise_id"] == enterprise_id), key=lambda row: row["stat_month"])
    by_month = {row["stat_month"]: row for row in months}
    metrics = profile.get("management_metrics")
    _require(isinstance(metrics, dict) and months, f"{label}: missing management measures or canonical operating process")
    snapshots = sorted((row for row in truth["enterprise_operation_profile"].values() if row["enterprise_id"] == enterprise_id), key=lambda row: (row["as_of_date"], row["report_revision"]))
    expected_money = {
        "latest_month_revenue": _money(months[-1], "operating_revenue", "operating_revenue"),
        "window_revenue": sum((_money(row, "operating_revenue", "operating_revenue") for row in months), Decimal(0)),
        "window_net_profit": sum((_money(row, "net_profit", "net_profit") for row in months), Decimal(0)),
        "current_assets": _money(snapshots[-1], "total_assets", "total_assets"),
        "current_liabilities": _money(snapshots[-1], "total_liabilities", "total_liabilities"),
    }
    _require(all(_decimal(metrics.get(field), f"{label}.{field}", 2) == expected for field, expected in expected_money.items()), f"{label}: management measures differ from final canonical facts")
    _require(metrics.get("latest_employee_count") == months[-1]["employee_count"] and metrics.get("loss_months") == [row["stat_month"] for row in months if _money(row, "net_profit", "net_profit") < 0], f"{label}: staffing or loss chronology differs")
    _require(profile.get("window_month_count") == len(months) and profile.get("partial_periods") == [row["stat_month"] for row in months if row["period_start"] != row["stat_month"]], f"{label}: management lifecycle/partial-period scope differs")
    basic = next(row for row in truth["enterprise_basic"].values() if row["enterprise_id"] == enterprise_id)
    _require(profile.get("current_business_description") == basic["main_business_description"], f"{label}: private management background differs from the public objective description")
    features = set(profile.get("business_features", []))
    if features & {"continuous_loss", "two_month_loss_then_recovery", "three_month_continuous_loss", "isolated_loss_without_continuity"}:
        signs = tuple(_money(by_month[month], "net_profit", "net_profit") < 0 for month in ("2026-04-01", "2026-05-01", "2026-06-01"))
        patterns = {"continuous_loss": (True, True, True), "two_month_loss_then_recovery": (True, True, False), "three_month_continuous_loss": (True, True, True), "isolated_loss_without_continuity": (True, False, True)}
        _require(all(signs == pattern for feature, pattern in patterns.items() if feature in features), f"{label}: loss feature differs from actual month-by-month business outcomes")
    if "equal_real_months" in features:
        first, second = (by_month[month] for month in ("2025-11-01", "2025-12-01"))
        _require(first["source_record_ref"] != second["source_record_ref"] and _money(first, "operating_revenue", "operating_revenue") == _money(second, "operating_revenue", "operating_revenue"), f"{label}: equal-value feature lacks two distinct real business months")
    for feature, predicate in (("confirmed_zero_tax", lambda value: value == 0), ("confirmed_positive_tax", lambda value: value > 0)):
        if feature in features:
            _require(predicate(_money(by_month["2026-02-01"], "tax_amount", "tax_amount")), f"{label}: confirmed-tax feature differs from the actual cash payment")
    if "revenue_contraction" in features:
        _require(all(_money(by_month[f"2026-{month:02d}-01"], "operating_revenue", "operating_revenue") < _money(by_month[f"2025-{month:02d}-01"], "operating_revenue", "operating_revenue") for month in (4, 5, 6)), f"{label}: delayed-acceptance contraction does not appear in comparable prior-year months")
    if "social_coverage_gap" in features:
        _require(all(by_month[f"2026-{month:02d}-01"]["social_security_count"] < by_month[f"2026-{month:02d}-01"]["employee_count"] for month in (4, 5, 6)), f"{label}: reporting-scope headcount difference is absent")
    initial_assets, final_assets = (_money(row, "total_assets", "total_assets") for row in (snapshots[0], snapshots[-1]))
    _require(initial_assets > 0 and final_assets > 0, f"{label}: leverage comparison has no positive asset base")
    initial_ratio = _money(snapshots[0], "total_liabilities", "total_liabilities") / initial_assets
    final_liabilities = _money(snapshots[-1], "total_liabilities", "total_liabilities")
    _require(("leverage_increase" in features) == (final_liabilities / final_assets > initial_ratio), f"{label}: leverage classification must describe the actual final-versus-initial stock result")
    _require(("negative_owners_equity" in features) == (final_liabilities > final_assets), f"{label}: negative-equity classification must describe the actual final stock result")


def _driver_process_check(entry: Mapping[str, Any], row: Mapping[str, Any], values: Mapping[str, Decimal], previous_revenue: Decimal, label: str, *, settlement_revenue: Decimal | None = None, settlement_process: Mapping[str, Any] | None = None) -> None:
    drivers = entry.get("drivers")
    _require(isinstance(drivers, dict), f"{label}: missing independent business drivers")
    driver_fields = (
        "delivered_units", "unit_price", "main_business_share", "material_units",
        "material_unit_cost", "staff", "monthly_salary", "coverage_fraction",
        "base_overhead", "variable_overhead_rate", "rework_units", "rework_unit_cost",
        "capacity_reservation_cost", "finance_rate", "invoice_current_weight",
        "invoice_previous_weight", "purchase_current_weight", "purchase_previous_weight",
        "sales_tax_payment_rate", "income_tax_rate", "utility_purchase_share",
        "travel_purchase_share", "social_count", "social_per_person",
        "asset_impairment_expense", "depreciation_expense",
        "salary_per_person", "employer_social_rate", "job_pay_factor",
        "staffing_reference_revenue", "lagged_capacity_reference_revenue", "lagged_capacity_factor",
        "loan_interest_expense", "bond_interest_expense",
        "prior_cumulative_income_tax_expense", "prior_cumulative_tax_paid", "scheduled_tax_payment",
        "base_material_rate", "reference_monthly_revenue_budget", "rework_quote_rate",
        "rework_quantity_fraction", "replacement_purchase_price_factor",
        "reverse_logistics_inspection_quote_rate", "reserved_remediation_capacity_rate",
    )
    parsed = {}
    for field in driver_fields:
        value = drivers.get(field)
        if isinstance(value, int) and not isinstance(value, bool):
            value = str(value)
        parsed[field] = _decimal(value, f"{label}.drivers.{field}", 28)
        _require(parsed[field] is not None and parsed[field] >= 0, f"{label}: missing/negative driver {field}")
    _require(all(parsed[field] <= 1 for field in ("main_business_share", "coverage_fraction", "finance_rate", "income_tax_rate", "sales_tax_payment_rate", "utility_purchase_share", "travel_purchase_share")), f"{label}: invalid bounded business rate")
    _require(parsed["staff"] == row["employee_count"] and parsed["social_count"] == row["social_security_count"], f"{label}: staffing drivers differ from the month-end fact")
    _require(Decimal(0) < parsed["employer_social_rate"] <= Decimal(1) and parsed["monthly_salary"] == parsed["salary_per_person"] == (parsed["social_per_person"] / parsed["employer_social_rate"] * parsed["job_pay_factor"]).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP), f"{label}: comprehensive staffing cost is not derived from the declared contribution base and job-pay parameters")
    _require(drivers.get("social_rate_semantics") == "synthetic_payroll_reporting_budget_coefficient_not_statutory_rate" and drivers.get("finance_expense_kind") == "settlement_processing_fee_and_contract_interest" and drivers.get("finance_billing_base") == "previous_finalized_business_month_revenue", f"{label}: contribution-budget and finance-cost parameters have undefined business meaning")
    stage = "before_revenue_and_profit" if settlement_revenue is None else "acceptance_document_estimate_before_reported_profit"
    _require(entry.get("driver_selection_stage") == stage, f"{label}: business inputs must be selected before financial results")
    _require(drivers.get("staffing_lag_months") == 3 and parsed["staffing_reference_revenue"] > 0 and parsed["lagged_capacity_factor"] == Decimal(".92") + Decimal(".08") * parsed["lagged_capacity_reference_revenue"] / parsed["staffing_reference_revenue"], f"{label}: staffing does not follow its declared lagged-capacity response")
    _require(parsed["unit_price"] > 0 and parsed["reference_monthly_revenue_budget"] == parsed["staffing_reference_revenue"], f"{label}: remediation must use the preselected capacity-reference budget and contract price")
    rework_kind = drivers.get("rework_mechanism_kind")
    if rework_kind == "affected_goods_batch_repurchase_reverse_logistics_and_inspection":
        _require(Decimal(".25") <= parsed["rework_quantity_fraction"] <= Decimal(".45") and Decimal("1.01") <= parsed["replacement_purchase_price_factor"] <= Decimal("1.08") and Decimal(".025") <= parsed["reverse_logistics_inspection_quote_rate"] <= Decimal(".045") and Decimal(".035") <= parsed["reserved_remediation_capacity_rate"] <= Decimal(".055"), f"{label}: trade replacement quantity, repurchase or handling quote is outside its preselected business range")
        _require(parsed["rework_quote_rate"] == parsed["base_material_rate"] * parsed["replacement_purchase_price_factor"] + parsed["reverse_logistics_inspection_quote_rate"], f"{label}: trade replacement price is not the ordinary purchase quote plus independent reverse-logistics and inspection work")
    else:
        _require(rework_kind == "external_specialist_remediation_or_urgent_redelivery" and parsed["rework_quantity_fraction"] == Decimal(".95") and Decimal(".95") <= parsed["rework_quote_rate"] <= Decimal("1.15") and parsed["replacement_purchase_price_factor"] == parsed["reverse_logistics_inspection_quote_rate"] == 0 and parsed["reserved_remediation_capacity_rate"] == Decimal(".12"), f"{label}: specialist remediation has an undefined independent quantity or emergency quote")
    _require(parsed["rework_unit_cost"] == _cents(parsed["unit_price"] * parsed["rework_quote_rate"]), f"{label}: remediation unit cost differs from its independent quote")
    month = _day(entry["stat_month"], "stat_month")
    feature_months = {"two_month_loss_then_recovery": {4, 5}, "three_month_continuous_loss": {4, 5, 6}, "continuous_loss": {4, 5, 6}, "isolated_loss_without_continuity": {4, 6}}
    features = set(entry.get("business_features", []))
    shock = month.year == 2026 and any(month.month in months for feature, months in feature_months.items() if feature in features)
    planned_rework = (parsed["reference_monthly_revenue_budget"] / parsed["unit_price"] * parsed["rework_quantity_fraction"] * parsed["coverage_fraction"]).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP) if shock else Decimal(0)
    planned_capacity = _cents(parsed["reference_monthly_revenue_budget"] * parsed["reserved_remediation_capacity_rate"] * parsed["coverage_fraction"]) if shock else Decimal(0)
    _require(parsed["rework_units"] == planned_rework and parsed["capacity_reservation_cost"] >= planned_capacity and ("business_exit_and_asset_review" in features or parsed["capacity_reservation_cost"] == planned_capacity), f"{label}: remediation quantity or reserved capacity does not follow the preselected business plan")
    start, end, month = (_day(row[field], field) for field in ("period_start", "period_end", "stat_month"))
    expected_fraction = Decimal((end - start).days + 1) / Decimal((end - month).days + 1)
    _require(parsed["coverage_fraction"] == expected_fraction, f"{label}: flow coverage fraction differs from the actual business period")
    revenue = values["revenue"]
    _require(values["other_income"] == 0, f"{label}: this process contract has no independently sourced other-income mechanism")
    expected_costs = {
        "revenue": _cents(parsed["delivered_units"] * parsed["unit_price"]),
        "direct_cost": _cents(parsed["material_units"] * parsed["material_unit_cost"] + parsed["rework_units"] * parsed["rework_unit_cost"]),
        "payroll_expense": _cents(parsed["staff"] * parsed["monthly_salary"] * parsed["coverage_fraction"]),
        "operating_expense": _cents(parsed["base_overhead"] * parsed["coverage_fraction"] + revenue * parsed["variable_overhead_rate"] + parsed["capacity_reservation_cost"] + parsed["asset_impairment_expense"] + parsed["depreciation_expense"]),
        "finance_expense": _cents(previous_revenue * parsed["finance_rate"] * parsed["coverage_fraction"]) + parsed["loan_interest_expense"] + parsed["bond_interest_expense"],
        "income_tax_expense": _cents(values["total_profit"] * parsed["income_tax_rate"]) if values["total_profit"] > 0 else Decimal(0),
    }
    for field, expected in expected_costs.items():
        _require(values[field] == expected, f"{label}: {field} is not derived directly from its declared business drivers")
    exempt = drivers.get("tax_payment_exempt")
    _require(isinstance(exempt, bool), f"{label}: tax_payment_exempt must be an explicit boolean")
    due_tax = parsed["prior_cumulative_income_tax_expense"] - parsed["prior_cumulative_tax_paid"]
    _require(drivers.get("tax_payment_basis") == "prior_known_income_tax_accrual_after_paid_credits" and parsed["sales_tax_payment_rate"] == 0 and due_tax >= 0 and parsed["scheduled_tax_payment"] == due_tax, f"{label}: tax notification does not reflect prior accrued and still unpaid income tax")
    cash_revenue = revenue if settlement_revenue is None else settlement_revenue
    _require(settlement_revenue is None or settlement_process is not None, f"{label}: original report estimate has no independent actual procurement process")
    actual_process = entry if settlement_process is None else settlement_process
    invoice_fields = (
        "sales_invoice_opening_pending", "sales_invoice_current_issued", "sales_invoice_closing_pending",
        "procurement_consumed_direct_cost", "procurement_purchased_period_expense", "capital_expenditure", "procurement_basis",
        "purchase_invoice_opening_pending", "purchase_invoice_current_issued", "purchase_invoice_closing_pending",
        "opening_inventory", "previous_inventory", "closing_inventory",
        "opening_procurement_basis_estimate", "normal_monthly_input_budget", "inventory_buffer_months",
        "inventory_target_factor", "opening_sales_pending_rate", "opening_purchase_pending_rate",
        "utility_expense_base", "travel_hospitality_expense_base", "other_purchased_period_expense_base",
        "utility_invoice_opening_pending", "utility_invoice_current_issued", "utility_invoice_closing_pending",
        "travel_hospitality_invoice_opening_pending", "travel_hospitality_invoice_current_issued", "travel_hospitality_invoice_closing_pending",
        "utility_opening_prototype_base", "travel_opening_prototype_base",
    )
    invoice = {}
    for field in invoice_fields:
        scale = 28 if field == "inventory_target_factor" else 6 if field in {"normal_monthly_input_budget", "inventory_buffer_months", "opening_sales_pending_rate", "opening_purchase_pending_rate"} else 2
        invoice[field] = _decimal(drivers.get(field), f"{label}.drivers.{field}", scale)
        _require(invoice[field] is not None and invoice[field] >= 0, f"{label}: missing/negative invoice or inventory driver {field}")
    change = _decimal(drivers.get("inventory_change"), "inventory_change", 2)
    _require(change is not None and change == invoice["closing_inventory"] - invoice["previous_inventory"], f"{label}: inventory purchase movement differs from its actual opening and closing stock")
    _require(invoice["closing_inventory"] == _cents(invoice["opening_inventory"] + invoice["normal_monthly_input_budget"] * invoice["inventory_buffer_months"] * invoice["inventory_target_factor"]), f"{label}: inventory does not follow its preselected capacity-buffer plan")
    _require(drivers.get("invoice_process_mode") == "conserved_current_and_prior_pending_queues" and parsed["invoice_previous_weight"] == parsed["purchase_previous_weight"] == 0 and Decimal(0) <= parsed["invoice_current_weight"] <= 1 and Decimal(0) <= parsed["purchase_current_weight"] <= 1, f"{label}: invoice reporting must conserve current and prior pending amounts")
    actual_direct, actual_overhead = (_decimal(actual_process[field], field, 2) for field in ("direct_cost", "operating_expense"))
    _require(invoice["procurement_consumed_direct_cost"] == actual_direct and invoice["procurement_purchased_period_expense"] == actual_overhead - parsed["depreciation_expense"] - parsed["asset_impairment_expense"], f"{label}: procurement improperly includes own payroll, noncash costs or estimated rather than actual consumption")
    expected_procurement = actual_direct + invoice["procurement_purchased_period_expense"] + invoice["capital_expenditure"] + change
    _require(expected_procurement >= 0 and invoice["procurement_basis"] == expected_procurement, f"{label}: purchase basis is not derived from actual external consumption, purchased period expenses, inventory movements and independent capital purchases")
    issued_sales, issued_purchase = _cents(cash_revenue * parsed["invoice_current_weight"]), _cents(expected_procurement * parsed["purchase_current_weight"])
    _require(invoice["sales_invoice_current_issued"] == issued_sales and invoice["sales_invoice_closing_pending"] == cash_revenue - issued_sales, f"{label}: current sales invoices and uninvoiced revenue do not preserve the actual revenue")
    _require(invoice["purchase_invoice_current_issued"] == issued_purchase and invoice["purchase_invoice_closing_pending"] == expected_procurement - issued_purchase, f"{label}: current purchase invoices and pending purchases do not preserve actual procurement")
    utility_basis = _cents(parsed["base_overhead"] * parsed["coverage_fraction"] * parsed["utility_purchase_share"])
    travel_basis = _cents(cash_revenue * parsed["variable_overhead_rate"] * parsed["travel_purchase_share"])
    _require(invoice["utility_expense_base"] == utility_basis and invoice["travel_hospitality_expense_base"] == travel_basis and invoice["other_purchased_period_expense_base"] + utility_basis + travel_basis == invoice["procurement_purchased_period_expense"], f"{label}: energy and travel do not follow the separate facility/activity budgets within actual purchased period expenses")
    issued_utility, issued_travel = _cents(utility_basis * parsed["purchase_current_weight"]), _cents(travel_basis * parsed["purchase_current_weight"])
    _require(invoice["utility_invoice_current_issued"] == issued_utility and invoice["utility_invoice_closing_pending"] == utility_basis - issued_utility and invoice["travel_hospitality_invoice_current_issued"] == issued_travel and invoice["travel_hospitality_invoice_closing_pending"] == travel_basis - issued_travel, f"{label}: energy or travel invoices do not preserve their actual expense bases")
    _require(drivers.get("vat_settlement_assumption") == "same_period_pass_through_net_zero_no_vat_timing_assets_or_liabilities", f"{label}: invoice cash amounts have no documented VAT settlement scope")
    expected_purchase = issued_purchase + invoice["purchase_invoice_opening_pending"]
    expected_flows = {
        "main_business_revenue": _cents(revenue * parsed["main_business_share"]),
        "invoice_sales_amount": issued_sales + invoice["sales_invoice_opening_pending"],
        "invoice_purchase_amount": expected_purchase,
        "utility_invoice_amount": issued_utility + invoice["utility_invoice_opening_pending"],
        "travel_hospitality_invoice_amount": issued_travel + invoice["travel_hospitality_invoice_opening_pending"],
        "social_security_amount": _cents(parsed["social_count"] * parsed["social_per_person"] * parsed["coverage_fraction"]),
        "tax_amount": Decimal(0) if exempt else parsed["scheduled_tax_payment"],
    }
    for field, expected in expected_flows.items():
        _require(_money(row, field, field) == expected, f"{label}: {field} differs from its final-revenue, staffing or settlement driver")
    _require(drivers.get("tax_payment_semantics") == "synthetic_income_tax_prepayment_schedule" and _decimal(entry.get("tax_payment"), f"{label}.tax_payment", 2) == expected_flows["tax_amount"], f"{label}: cash-tax process is not the declared income-tax prepayment schedule")
    social_cost = _decimal(entry.get("reported_employer_social_cost"), f"{label}.reported_employer_social_cost", 2)
    other_labor = _decimal(entry.get("other_labor_cost"), f"{label}.other_labor_cost", 2)
    _require(drivers.get("payroll_cost_basis") == "comprehensive_employer_labor_cost_including_reported_social_contribution" and social_cost == expected_flows["social_security_amount"] and other_labor is not None and other_labor >= 0 and social_cost + other_labor == values["payroll_expense"], f"{label}: comprehensive payroll does not include its reported social contribution exactly once")


def _interest_contract_maps(debt: list[dict[str, Any]], events: list[dict[str, Any]]) -> tuple[dict[Any, Any], dict[Any, Any], dict[Any, Decimal]]:
    loans, bonds, rates = {}, {}, {}
    for ledger in debt:
        identity = (ledger["enterprise_id"], ledger["business_object_ref"])
        _require(identity not in loans, f"interest: duplicate loan ledger {identity}")
        loans[identity] = ledger
    for event in events:
        kind = event["event_type"]
        if kind not in {"银行借款", "债券发行"}:
            continue
        identity = (event["enterprise_id"], kind, event["business_object_ref"])
        match = re.search(r"合同约定年利率(\d+(?:\.\d+)?)%", event.get("description", ""))
        _require(match is not None and "实际天数/365" in event["description"] and "月末现金结息" in event["description"], f"interest: missing public contract-rate or payment basis for {identity}")
        rate = Decimal(match.group(1)) / Decimal(100)
        _require(Decimal(0) < rate <= Decimal(1) and (identity not in rates or rates[identity] == rate), f"interest: inconsistent public contract rate for {identity}")
        rates[identity] = rate
        if kind == "债券发行":
            bond_identity = (event["enterprise_id"], event["business_object_ref"])
            facts = (event["event_date"], _money(event, "amount", "bond principal"))
            _require(bond_identity not in bonds or bonds[bond_identity] == facts, f"interest: bond object identifies more than one issuance for {bond_identity}")
            bonds[bond_identity] = facts
    return loans, bonds, rates


def _interest_process_check(entry: Mapping[str, Any], loans: Mapping[Any, Any], bonds: Mapping[Any, Any], rates: Mapping[Any, Decimal], label: str) -> None:
    enterprise_id, drivers = entry["enterprise_id"], entry["drivers"]
    start, end = (_day(entry[field], field) for field in ("period_start", "period_end"))
    for kind, terms_field, amount_field in (("银行借款", "loan_interest_terms", "loan_interest_expense"), ("债券发行", "bond_interest_terms", "bond_interest_expense")):
        expected = {}
        if kind == "银行借款":
            for (eid, object_ref), ledger in loans.items():
                if eid != enterprise_id:
                    continue
                daily = {}
                day = start
                while day <= end:
                    applicable = [point for point in ledger["checkpoints"] if _day(point["balance_as_of_date"], "balance_as_of_date") <= day]
                    checkpoint = max(applicable, key=lambda point: point["balance_as_of_date"]) if applicable else None
                    daily[day] = Decimal(0) if checkpoint is None else _decimal(checkpoint["outstanding_principal"], "outstanding_principal", 2)
                    day += timedelta(days=1)
                if sum(daily.values(), Decimal(0)) > 0:
                    expected[object_ref] = daily
        else:
            for (eid, object_ref), (issued, principal) in bonds.items():
                if eid != enterprise_id or _day(issued, "bond.event_date") > end:
                    continue
                day, daily = max(start, _day(issued, "bond.event_date")), {}
                while day <= end:
                    daily[day] = principal
                    day += timedelta(days=1)
                expected[object_ref] = daily
        terms = drivers.get(terms_field)
        _require(isinstance(terms, list), f"{label}: missing per-contract {kind} interest process")
        seen, interest_total = set(), Decimal(0)
        for term in terms:
            _require(isinstance(term, dict), f"{label}: invalid interest term")
            object_ref = term.get("business_object_ref")
            _require(object_ref in expected and object_ref not in seen, f"{label}: missing/duplicate/nonaccruing {kind} interest contract")
            _require(term.get("day_count_basis") == 365 and term.get("payment_status") == "settled_at_period_end", f"{label}: actual/365 and month-end cash settlement must be explicit")
            rate = rates.get((enterprise_id, kind, object_ref))
            _require(rate is not None and _decimal(term.get("annual_interest_rate"), "annual_interest_rate", 6) == rate, f"{label}: private interest rate differs from the public contract")
            segments = term.get("daily_principal_days")
            _require(isinstance(segments, list) and segments, f"{label}: missing dated principal segments")
            actual_daily = {}
            for segment in segments:
                _require(isinstance(segment, dict), f"{label}: invalid principal segment")
                first, last = (_day(segment.get(field), field) for field in ("period_start", "period_end"))
                days, principal = segment.get("days"), _decimal(segment.get("principal"), "segment.principal", 2)
                _require(start <= first <= last <= end and isinstance(days, int) and not isinstance(days, bool) and days == (last - first).days + 1 and principal is not None and principal >= 0, f"{label}: invalid principal-period coverage")
                day = first
                while day <= last:
                    _require(day not in actual_daily, f"{label}: overlapping principal segments")
                    actual_daily[day] = principal
                    day += timedelta(days=1)
            _require(actual_daily == expected[object_ref], f"{label}: daily interest principal differs from the dated contract ledger")
            interest = _cents(sum(expected[object_ref].values(), Decimal(0)) * rate / Decimal(365))
            _require(_decimal(term.get("interest_amount"), "interest_amount", 2) == interest, f"{label}: actual/365 contract interest differs")
            interest_total += interest
            seen.add(object_ref)
        _require(seen == set(expected) and _decimal(drivers.get(amount_field), amount_field, 2) == interest_total, f"{label}: finance expense omits or duplicates actual {kind} contract interest")


def _reporting_estimate_checks(processes: list[dict[str, Any]], monthly_rows: list[dict[str, Any]], latest: Mapping[Any, Any], loans: Mapping[Any, Any], bonds: Mapping[Any, Any], rates: Mapping[Any, Decimal]) -> int:
    all_rows = {row["monthly_record_id"]: row for row in monthly_rows}
    canonical_ids = {row["monthly_record_id"] for row in latest.values()}
    earlier_ids = set(all_rows) - canonical_ids
    seen = set()
    for final_process in processes:
        identity = tuple(final_process[field] for field in ("enterprise_id", "stat_month", "source_system", "source_record_ref"))
        final_row = latest[identity]
        estimates = final_process.get("lawful_reporting_estimates", [])
        _require(isinstance(estimates, list), f"reporting_estimates.{identity}: expected an array")
        for estimate in estimates:
            _require(isinstance(estimate, dict), f"reporting_estimates.{identity}: expected an object")
            record_id = estimate.get("monthly_record_id")
            label = f"reporting_estimates.{record_id}"
            _require(record_id in earlier_ids and record_id not in seen, f"{label}: missing/duplicate/nonhistorical report process")
            row = all_rows[record_id]
            _require(estimate.get("estimate_kind") == "lawful_initial_acceptance_estimate" and tuple(estimate.get(field) for field in ("enterprise_id", "stat_month", "source_system", "source_record_ref")) == identity, f"{label}: original report estimate has an undefined or mismatched business scope")
            reference = estimate.get("record_ref", {})
            _require(reference.get("table") == "enterprise_operation_monthly" and reference.get("primary_key", reference.get("key")) == {"monthly_record_id": record_id}, f"{label}: estimate has no exact original public report reference")
            _require(estimate.get("report_revision") == row["report_revision"] < final_row["report_revision"] and row["period_start"] == estimate.get("period_start") == final_row["period_start"] and row["period_end"] == estimate.get("period_end") == final_row["period_end"], f"{label}: report estimate revision or covered period differs")
            altered_fields = {"monthly_record_id", "operating_revenue", "main_business_revenue", "total_profit", "net_profit", "report_revision", "reported_at", "ingested_at"}
            _require(all(row[field] == final_row[field] for field in set(TABLE_COLUMNS["enterprise_operation_monthly"]) - altered_fields), f"{label}: acceptance estimate changed settled cash, invoices, workforce or unrelated source facts")
            values = {field: _decimal(estimate.get(field), f"{label}.{field}", 2) for field in ("revenue", "direct_cost", "payroll_expense", "operating_expense", "finance_expense", "other_income", "total_profit", "income_tax_expense", "net_profit")}
            _require(None not in values.values() and all(values[field] >= 0 for field in ("revenue", "direct_cost", "payroll_expense", "operating_expense", "finance_expense", "other_income", "income_tax_expense")), f"{label}: incomplete/negative original-report process")
            computed = values["revenue"] - sum(values[field] for field in ("direct_cost", "payroll_expense", "operating_expense", "finance_expense")) + values["other_income"]
            _require(computed == values["total_profit"] and values["total_profit"] - values["income_tax_expense"] == values["net_profit"], f"{label}: original-report profit does not follow its estimated business process")
            _require(all(_money(row, field, field) == values[source] for field, source in (("operating_revenue", "revenue"), ("total_profit", "total_profit"), ("net_profit", "net_profit"))), f"{label}: estimate differs from the public original report")
            drivers, final_drivers = estimate.get("drivers"), final_process["drivers"]
            _require(isinstance(drivers, dict) and set(drivers) == set(final_drivers) | {"initial_acceptance_coverage_rate"}, f"{label}: initial-report driver contract differs")
            changed_inputs = {"delivered_units", "material_units"}
            _require(all(drivers[field] == value for field, value in final_drivers.items() if field not in changed_inputs), f"{label}: initial acceptance estimate altered independent fixed inputs or prices")
            fraction = _decimal(drivers.get("initial_acceptance_coverage_rate"), "initial_acceptance_coverage_rate", 6)
            _require(fraction is not None and Decimal(0) < fraction < Decimal(1), f"{label}: undefined original acceptance-document coverage")
            quantity = (_decimal(final_drivers["delivered_units"], "final delivered_units", 6) * fraction).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
            _require(_decimal(drivers["delivered_units"], "initial delivered_units", 6) == _decimal(drivers["material_units"], "initial material_units", 6) == quantity, f"{label}: original quantity does not follow the independently chosen document-coverage estimate")
            previous_revenue = _decimal(estimate.get("invoice_and_payment_timing_reference"), "invoice_and_payment_timing_reference", 2)
            _require(previous_revenue == _decimal(final_process["invoice_and_payment_timing_reference"], "final invoice_and_payment_timing_reference", 2), f"{label}: estimate changed the settled preceding-month reference")
            _driver_process_check(estimate, row, values, previous_revenue, label, settlement_revenue=_decimal(final_process["revenue"], "final revenue", 2), settlement_process=final_process)
            _interest_process_check(estimate, loans, bonds, rates, label)
            _require(isinstance(estimate.get("mechanism"), str) and bool(estimate["mechanism"].strip()), f"{label}: original report has no business estimation mechanism")
            seen.add(record_id)
    _require(seen == earlier_ids, "facts: every legitimate earlier monthly report must have a verifiable reporting estimate process")
    return len(seen)


def _stock_driver_check(entry: Mapping[str, Any], values: Mapping[str, Decimal], process_by_month: Mapping[Any, Any], events: list[dict[str, Any]], config: Mapping[str, Any], prior_owner_totals: Mapping[str, Decimal] | None, label: str, *, established: date) -> dict[str, Decimal]:
    day, enterprise_id = _day(entry["as_of_date"], "as_of_date"), entry["enterprise_id"]
    opening_fields = ("opening_cash", "opening_trade_receivables", "opening_inventory", "opening_fixed_assets", "opening_trade_payables", "opening_other_liabilities", "opening_owners_equity", "opening_total_assets", "opening_total_liabilities", "prototype_assets_before_distribution", "prototype_liabilities", "pre_window_owner_distribution", "opening_owner_contributed_equity_budget", "opening_retained_earnings_budget", "opening_retained_earnings_after_distribution")
    opening = {field: _decimal(entry.get(field), f"{label}.{field}", 2) for field in opening_fields}
    _require(None not in opening.values() and all(value >= 0 for value in opening.values()), f"{label}: incomplete/negative opening stock")
    opening_day = _day(entry.get("opening_stock_date"), "opening_stock_date")
    _require(opening_day <= day and entry.get("opening_stock_kind") in {"registration_opening_funding_estimate", "window_opening_prototype_estimate"}, f"{label}: invalid opening stock scope")
    _require(sum(opening[field] for field in ("opening_cash", "opening_trade_receivables", "opening_inventory", "opening_fixed_assets")) == opening["opening_total_assets"], f"{label}: opening asset components differ")
    _require(opening["opening_trade_payables"] + opening["opening_other_liabilities"] == opening["opening_total_liabilities"] == opening["prototype_liabilities"], f"{label}: opening liability components differ")
    _require(opening["opening_total_assets"] - opening["opening_total_liabilities"] == opening["opening_owners_equity"] and opening["opening_total_assets"] + opening["pre_window_owner_distribution"] == opening["prototype_assets_before_distribution"], f"{label}: initial owner distribution is not explicit in the opening stock")
    age_days = (opening_day - established).days
    _require(age_days >= 0 and isinstance(entry.get("opening_business_age_days"), int) and not isinstance(entry["opening_business_age_days"], bool) and entry["opening_business_age_days"] == age_days and entry.get("pre_window_distribution_age_eligible") is (age_days >= 365), f"{label}: pre-window distribution maturity does not follow the actual establishment and opening-stock dates")
    retained_fraction = _decimal(entry.get("opening_retained_earnings_budget_fraction"), "opening_retained_earnings_budget_fraction", 6)
    distribution_ratio = _decimal(entry.get("pre_window_distribution_ratio"), "pre_window_distribution_ratio", 6)
    _require(retained_fraction is not None and distribution_ratio is not None and Decimal(0) <= retained_fraction <= 1 and Decimal(0) <= distribution_ratio <= 1 and entry.get("opening_equity_composition_scope") == "preselected_budget_estimate_not_observed_history", f"{label}: initial owner capital and retained earnings need an explicit independent budget scope")
    prototype_equity = opening["prototype_assets_before_distribution"] - opening["prototype_liabilities"]
    _require(opening["opening_retained_earnings_budget"] == _cents(prototype_equity * retained_fraction) and opening["opening_owner_contributed_equity_budget"] + opening["opening_retained_earnings_budget"] == prototype_equity, f"{label}: owner-funded equity and retained-earnings budgets do not preserve the initial equity composition")
    _require(opening["pre_window_owner_distribution"] == _cents(opening["opening_retained_earnings_budget"] * distribution_ratio) and opening["opening_retained_earnings_after_distribution"] == opening["opening_retained_earnings_budget"] - opening["pre_window_owner_distribution"], f"{label}: pre-window distribution is not funded solely by its preselected retained-earnings budget")
    _require(age_days >= 365 or (distribution_ratio == 0 and opening["pre_window_owner_distribution"] == 0), f"{label}: an enterprise operating for less than one year has a pre-window owner distribution")
    distribution_date = entry.get("pre_window_owner_distribution_date")
    _require((distribution_date is not None) == (opening["pre_window_owner_distribution"] > 0) and (distribution_date is None or _day(distribution_date, "pre_window_owner_distribution_date") == opening_day), f"{label}: pre-window distribution execution/date mismatch")
    _require(isinstance(entry.get("pre_window_distribution_reason"), str) and bool(entry["pre_window_distribution_reason"].strip()), f"{label}: missing opening distribution rationale")
    plan = entry.get("funding_plan")
    _require(isinstance(plan, dict) and plan.get("schedule_scope") == "preselected_business_funding_plan" and plan.get("selection_stage") == "before_snapshot_totals" and plan.get("enterprise_id") == enterprise_id, f"{label}: funding must follow a named plan selected before snapshot totals")
    opening_metadata_fields = ("opening_stock_date", "opening_stock_kind", "pre_window_distribution_reason", "opening_business_age_days", "pre_window_distribution_age_eligible", "opening_equity_composition_scope", "opening_retained_earnings_budget_fraction", "pre_window_distribution_ratio", "pre_window_owner_distribution_date")
    _require(isinstance(plan.get("opening"), dict) and all(plan["opening"].get(field) == entry.get(field) for field in opening_fields + opening_metadata_fields), f"{label}: funding plan opening differs from the stock process")
    flows = plan.get("flows")
    _require(isinstance(flows, list), f"{label}: missing independent owner/capital schedule")
    scheduled = {kind: Decimal(0) for kind in ("capital_expenditure", "owner_contribution", "owner_distribution")}
    capital_flows = {}
    owner_distribution_flows = {}
    flow_ids = set()
    for flow in flows:
        _require(isinstance(flow, dict) and flow.get("flow_type") in scheduled, f"{label}: undefined funding flow")
        flow_day = _day(flow.get("flow_date"), "flow_date")
        amount = _decimal(flow.get("amount"), "scheduled amount", 2)
        _require(amount is not None and amount >= 0 and opening_day <= flow_day <= _day(config["window_end"], "window_end"), f"{label}: invalid scheduled owner/capital flow")
        _require(isinstance(flow.get("reason"), str) and bool(flow["reason"].strip()), f"{label}: scheduled funding has no commercial reason")
        flow_id = (flow_day, flow["flow_type"])
        _require(flow_id not in flow_ids, f"{label}: duplicate scheduled funding identity")
        flow_ids.add(flow_id)
        if flow["flow_type"] == "capital_expenditure":
            capital_flows[flow_day.isoformat()] = amount
        elif flow["flow_type"] == "owner_distribution":
            owner_distribution_flows[flow_day.isoformat()] = amount
        if flow_day <= day:
            scheduled[flow["flow_type"]] += amount
    applicable = sorted((process for (eid, _), process in process_by_month.items() if eid == enterprise_id and _day(process["period_end"], "period_end") <= day), key=lambda process: process["stat_month"])
    _require(bool(applicable), f"{label}: stock date has no applicable operating process")
    decisions = plan.get("owner_distribution_decisions")
    _require(isinstance(decisions, list), f"{label}: missing retained-earnings distribution decisions")
    decided_payments = {}
    for decision in decisions:
        _require(isinstance(decision, dict), f"{label}: invalid owner distribution decision")
        planned_day, decision_day, knowledge_day, payment_day = (_day(decision.get(field), field) for field in ("planned_at", "decision_at", "knowledge_period_end", "planned_payment_date"))
        _require(planned_day == opening_day and knowledge_day == date(2026, 5, 31) and decision_day == date(2026, 6, 15) and payment_day == date(2026, 6, 30) and planned_day <= knowledge_day < decision_day <= payment_day, f"{label}: distribution decision uses undefined or future business information")
        payout_fraction = _decimal(decision.get("planned_payout_fraction"), "planned_payout_fraction", 6)
        planned_amount, recorded_profit, recorded_available, actual_amount = (_decimal(decision.get(field), field, 2) for field in ("planned_amount", "known_cumulative_net_profit", "available_retained_earnings", "actual_amount"))
        _require(payout_fraction is not None and Decimal(0) <= payout_fraction <= 1 and None not in (planned_amount, recorded_profit, recorded_available, actual_amount) and planned_amount >= 0 and actual_amount >= 0 and decision.get("planned_budget_basis") == "opening_remaining_retained_earnings_budget", f"{label}: undefined independent owner distribution budget")
        known_profit = sum((_decimal(process["net_profit"], "known net_profit", 2) for (eid, _), process in process_by_month.items() if eid == enterprise_id and _day(process["period_end"], "period_end") <= knowledge_day), Decimal(0))
        available = opening["opening_retained_earnings_after_distribution"] + known_profit
        expected_planned = _cents(opening["opening_retained_earnings_after_distribution"] * payout_fraction)
        expected_actual = expected_planned if expected_planned > 0 and available >= expected_planned else Decimal(0)
        expected_status = "executed" if expected_actual > 0 else "deferred" if expected_planned > 0 else "not_planned"
        _require(planned_amount == expected_planned and recorded_profit == known_profit and recorded_available == available and actual_amount == expected_actual and decision.get("status") == expected_status, f"{label}: distribution decision does not use the initial retained budget and profit available through May; future or residual equity cannot fund the decision")
        actual_payment_date = decision.get("actual_payment_date")
        _require((actual_payment_date is not None) == (actual_amount > 0) and (actual_payment_date is None or _day(actual_payment_date, "actual_payment_date") == payment_day), f"{label}: distribution actual payment/date mismatch")
        _require(isinstance(decision.get("reason"), str) and bool(decision["reason"].strip()), f"{label}: owner distribution has no business decision basis")
        if actual_amount > 0:
            _require(payment_day.isoformat() not in decided_payments, f"{label}: duplicate actual distribution decision")
            decided_payments[payment_day.isoformat()] = actual_amount
    _require(owner_distribution_flows == decided_payments, f"{label}: owner cash distributions do not match independently reviewed executed decisions")
    _require(all(_decimal(process["drivers"].get("capital_expenditure"), "capital_expenditure", 2) == capital_flows.get(process["period_end"]) for process in applicable), f"{label}: monthly procurement capital purchases differ from the independent dated funding plan")
    cumulative = {
        "cumulative_net_profit": sum((_decimal(process["net_profit"], "net_profit", 2) for process in applicable), Decimal(0)),
        "cumulative_income_tax_expense": sum((_decimal(process["income_tax_expense"], "income_tax_expense", 2) for process in applicable), Decimal(0)),
        "cumulative_tax_paid": sum((_decimal(process["tax_payment"], "tax_payment", 2) for process in applicable), Decimal(0)),
        "cumulative_depreciation": sum((_decimal(process["drivers"]["depreciation_expense"], "depreciation_expense", 2) for process in applicable), Decimal(0)),
        "cumulative_asset_impairment": sum((_decimal(process["drivers"]["asset_impairment_expense"], "asset_impairment_expense", 2) for process in applicable), Decimal(0)),
        "cumulative_capital_expenditure": scheduled["capital_expenditure"],
        "cumulative_owner_contributions": scheduled["owner_contribution"],
        "cumulative_owner_distributions": scheduled["owner_distribution"],
    }
    real_funding = {}
    for event in events:
        if event["enterprise_id"] == enterprise_id and event["event_type"] in {"融资", "债券发行"} and _day(event["event_date"], "event_date") <= day:
            event_id = event["business_event_ref"]
            amount = _money(event, "amount", "funding amount")
            _require(event_id not in real_funding or real_funding[event_id] == (event["event_type"], amount), f"{label}: funding revisions change the real-event amount without a supported funding process")
            real_funding[event_id] = (event["event_type"], amount)
    cumulative["cumulative_equity_financing"] = sum((amount for kind, amount in real_funding.values() if kind == "融资"), Decimal(0))
    bond_principal = sum((amount for kind, amount in real_funding.values() if kind == "债券发行"), Decimal(0))
    for field, expected in cumulative.items():
        _require(_decimal(entry.get(field), f"{label}.{field}", 2) == expected, f"{label}: {field} does not follow the operating, independent funding or real-event process")
    _require(_decimal(entry.get("bond_principal"), "bond_principal", 2) == bond_principal, f"{label}: bond stock differs from distinct issuance events")
    working = entry.get("working_capital_drivers")
    _require(isinstance(working, dict), f"{label}: missing working-capital mechanisms")
    rates = {field: _decimal(working.get(field), f"{label}.working_capital_drivers.{field}", 28) for field in ("receivable_revenue_weight", "inventory_cost_weight", "payable_cost_weight", "customer_advance_revenue_weight", "supplier_credit_cost_rate")}
    _require(None not in rates.values() and all(Decimal(0) <= value <= Decimal(1) for value in rates.values()), f"{label}: invalid working-capital rate")
    revenue, direct = (_decimal(applicable[-1][field], field, 2) for field in ("revenue", "direct_cost"))
    _require(_decimal(working.get("latest_revenue"), "latest_revenue", 2) == revenue and _decimal(working.get("latest_direct_cost"), "latest_direct_cost", 2) == direct, f"{label}: working-capital driver does not reference the final applicable month")
    inventory_drivers = applicable[-1]["drivers"]
    capacity_budget = _decimal(working.get("inventory_capacity_budget"), "inventory_capacity_budget", 6)
    inventory_target_factor = _decimal(working.get("inventory_target_factor"), "inventory_target_factor", 28)
    _require(working.get("inventory_process_mode") == "monthly_preselected_buffer_rollforward" and working.get("inventory_period_end") == applicable[-1]["period_end"] and capacity_budget == _decimal(inventory_drivers.get("normal_monthly_input_budget"), "normal_monthly_input_budget", 6) and inventory_target_factor == _decimal(inventory_drivers.get("inventory_target_factor"), "inventory_target_factor", 28) and rates["inventory_cost_weight"] == _decimal(inventory_drivers.get("inventory_buffer_months"), "inventory_buffer_months", 6), f"{label}: snapshot inventory plan differs from its actual monthly procurement process")
    _require(opening["opening_inventory"] == _decimal(inventory_drivers.get("opening_inventory"), "opening_inventory", 2) and values["inventory"] == _decimal(inventory_drivers.get("closing_inventory"), "closing_inventory", 2), f"{label}: inventory stock is not the latest completed canonical monthly closing stock")
    supplier_credit = sum((_decimal(process["direct_cost"], "direct_cost", 2) * rates["supplier_credit_cost_rate"] for process in applicable), Decimal(0))
    _require(_decimal(working.get("supplier_credit_extension"), "supplier_credit_extension", 2) == _cents(supplier_credit), f"{label}: supplier credit is not derived from the agreed cost-based term")
    operating_other = _decimal(entry.get("operating_other_liabilities"), "operating_other_liabilities", 2)
    tax_balance = cumulative["cumulative_income_tax_expense"] - cumulative["cumulative_tax_paid"]
    tax_payable = tax_balance if tax_balance > 0 else Decimal(0)
    tax_prepaid = -tax_balance if tax_balance < 0 else Decimal(0)
    _require(_decimal(entry.get("tax_payable"), "tax_payable", 2) == tax_payable and _decimal(entry.get("tax_prepaid"), "tax_prepaid", 2) == tax_prepaid, f"{label}: tax payable/prepaid classifications do not preserve the signed accrual-payment difference")
    fixed_assets = opening["opening_fixed_assets"] + cumulative["cumulative_capital_expenditure"] - cumulative["cumulative_depreciation"] - cumulative["cumulative_asset_impairment"]
    _require(fixed_assets >= 0 and _decimal(entry.get("fixed_assets"), "fixed_assets", 2) == fixed_assets, f"{label}: fixed assets do not follow a nonnegative independent capital and noncash expense plan")
    _require(operating_other is not None and operating_other >= 0, f"{label}: invalid operating other-liability stock")
    expected_components = {
        "trade_receivables": _cents(opening["opening_trade_receivables"] + revenue * rates["receivable_revenue_weight"]),
        "inventory": _cents(opening["opening_inventory"] + capacity_budget * rates["inventory_cost_weight"] * inventory_target_factor),
        "trade_payables": _cents(opening["opening_trade_payables"] + direct * rates["payable_cost_weight"] + supplier_credit),
        "fixed_and_other_assets": fixed_assets + tax_prepaid,
        "owners_equity": opening["opening_owners_equity"] + cumulative["cumulative_net_profit"] + cumulative["cumulative_owner_contributions"] + cumulative["cumulative_equity_financing"] - cumulative["cumulative_owner_distributions"],
    }
    for field, expected in expected_components.items():
        _require(values[field] == expected, f"{label}: {field} is not derived directly from its declared stock mechanism")
    _require(operating_other == _cents(opening["opening_other_liabilities"] + revenue * rates["customer_advance_revenue_weight"]) and values["other_liabilities"] == operating_other + bond_principal + tax_payable, f"{label}: other liabilities do not reflect customer advances, bond principal and accrued unpaid tax")
    expected_cash = (opening["opening_cash"] + cumulative["cumulative_net_profit"] + cumulative["cumulative_income_tax_expense"] - cumulative["cumulative_tax_paid"] + cumulative["cumulative_depreciation"] + cumulative["cumulative_asset_impairment"] - (values["trade_receivables"] - opening["opening_trade_receivables"]) - (values["inventory"] - opening["opening_inventory"]) - cumulative["cumulative_capital_expenditure"] + (values["trade_payables"] - opening["opening_trade_payables"]) + (operating_other - opening["opening_other_liabilities"]) + values["bank_loan_principal"] + bond_principal + cumulative["cumulative_owner_contributions"] + cumulative["cumulative_equity_financing"] - cumulative["cumulative_owner_distributions"])
    _require(values["cash"] == expected_cash, f"{label}: cash does not follow the named operating, funding and noncash movements")
    changes = {"owner_contributions_change": "cumulative_owner_contributions", "equity_financing_change": "cumulative_equity_financing", "owner_distributions_change": "cumulative_owner_distributions"}
    owner_changes = {}
    for field, source in changes.items():
        owner_changes[field] = _decimal(entry.get(field), field, 2)
        expected = Decimal(0) if prior_owner_totals is None else cumulative[source] - prior_owner_totals[source]
        _require(owner_changes[field] == expected, f"{label}: {field} must follow the preselected schedule or real funding event, not an equity residual")
    _require(values["owner_and_other_equity_change"] == owner_changes["owner_contributions_change"] + owner_changes["equity_financing_change"] - owner_changes["owner_distributions_change"], f"{label}: owner-equity change has no named source")
    return cumulative


def _process_checks(private: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, int]:
    metadata = private.get("fact_metadata", {})
    _require(isinstance(metadata, dict), "fact_metadata: expected an object")
    facts = metadata.get("facts", metadata)
    _require(isinstance(facts, dict), "fact_metadata.facts: expected an object")
    processes = metadata.get("economic_processes", facts.get("economic_processes"))
    debt = metadata.get("debt_ledgers", facts.get("debt_ledgers"))
    balances = metadata.get("balance_processes", facts.get("balance_processes"))
    _require(isinstance(processes, list) and processes, "facts: missing income/cost/profit process ledger")
    _require(isinstance(debt, list), "facts: missing principal/repayment process ledger")
    _require(isinstance(balances, list) and balances, "facts: missing assets/liabilities stock process ledger")
    latest = {}
    for row in private["truth_tables"]["enterprise_operation_monthly"]:
        identity = (row["enterprise_id"], row["stat_month"], row["source_system"], row["source_record_ref"])
        if identity not in latest or row["report_revision"] > latest[identity]["report_revision"]:
            latest[identity] = row
    reviewed = _reviewed_contract(config)
    if reviewed:
        priced_loans, priced_bonds, public_interest_rates = _interest_contract_maps(debt, private["truth_tables"]["enterprise_operation_event"])
    canonical_months = {(row["enterprise_id"], row["stat_month"]): row for row in latest.values()}
    if reviewed:
        _require(len(canonical_months) == len(latest) == 2875, "facts: reviewed economic ledger must have one canonical process per establishment-conditioned business month")
    process_by_month = {(entry.get("enterprise_id"), entry.get("stat_month")): entry for entry in processes}
    seen = set()
    for index, entry in enumerate(processes):
        _require(isinstance(entry, dict), f"economic_processes[{index}]: expected an object")
        identity = (entry.get("enterprise_id"), entry.get("stat_month"), entry.get("source_system"), entry.get("source_record_ref"))
        _require(identity in latest and identity not in seen, f"economic_processes[{index}]: missing/duplicate business month")
        values = {field: _decimal(entry.get(field), f"economic_processes[{index}].{field}", 2) for field in ("revenue", "direct_cost", "payroll_expense", "operating_expense", "finance_expense", "other_income", "total_profit", "income_tax_expense", "net_profit")}
        _require(None not in values.values(), f"economic_processes[{index}]: incomplete financial process")
        _require(all(values[field] >= 0 for field in ("revenue", "direct_cost", "payroll_expense", "operating_expense", "finance_expense", "other_income", "income_tax_expense")), f"economic_processes[{index}]: negative expense/revenue")
        computed_profit = values["revenue"] - sum(values[field] for field in ("direct_cost", "payroll_expense", "operating_expense", "finance_expense")) + values["other_income"]
        _require(computed_profit == values["total_profit"] and values["total_profit"] - values["income_tax_expense"] == values["net_profit"], f"economic_processes[{index}]: income/cost/profit identity failed for {identity}")
        row = latest[identity]
        _require(all(_money(row, field, field) == values[source] for field, source in (("operating_revenue", "revenue"), ("total_profit", "total_profit"), ("net_profit", "net_profit"))), f"economic_processes[{index}]: financial process differs from the latest business fact")
        _require(row["period_start"] == entry.get("period_start") and row["period_end"] == entry.get("period_end"), f"economic_processes[{index}]: coverage differs")
        if reviewed and ("monthly_record_id" in entry or "report_revision" in entry):
            _require(entry.get("monthly_record_id") == row["monthly_record_id"] and entry.get("report_revision") == row["report_revision"], f"economic_processes[{index}]: canonical reporting identity differs from its final public revision")
        previous_revenue = _decimal(entry.get("invoice_and_payment_timing_reference"), f"economic_processes[{index}].invoice_and_payment_timing_reference", 2)
        _require(previous_revenue is not None and previous_revenue >= 0, f"economic_processes[{index}]: missing/negative invoice and payment timing reference")
        enterprise = next(item for item in private["truth_tables"]["enterprise_basic"] if item["enterprise_id"] == entry["enterprise_id"])
        established = _day(enterprise["established_date"], "established_date")
        if established >= _day(config["window_start"], "window_start") and _day(entry["stat_month"], "stat_month") == established.replace(day=1):
            _require(previous_revenue == 0, f"economic_processes[{index}]: first business month refers to revenue before enterprise establishment")
            _require(entry.get("timing_reference_kind") == "no_pre_establishment_activity" and entry.get("opening_prototype") is False, f"economic_processes[{index}]: establishment-month timing reference is incorrectly described")
        if reviewed:
            month = _day(entry["stat_month"], "stat_month")
            preceding_month = date(month.year - (month.month == 1), 12 if month.month == 1 else month.month - 1, 1).isoformat()
            prior_fact = canonical_months.get((entry["enterprise_id"], preceding_month))
            if prior_fact is not None:
                _require(previous_revenue == _money(prior_fact, "operating_revenue", "operating_revenue"), f"economic_processes[{index}]: prior-month reference differs from the final canonical revenue of {entry['enterprise_id']} {preceding_month}")
                _require(entry.get("timing_reference_kind") == "previous_canonical_month" and entry.get("opening_prototype") is False, f"economic_processes[{index}]: in-window timing reference is incorrectly classified")
            elif established < _day(config["window_start"], "window_start"):
                _require(entry["stat_month"] == config["window_start"] and entry.get("timing_reference_kind") == "opening_prototype" and entry.get("opening_prototype") is True, f"economic_processes[{index}]: unexplained missing preceding month")
            _driver_process_check(entry, row, values, previous_revenue, f"economic_processes[{index}] {identity}")
            _interest_process_check(entry, priced_loans, priced_bonds, public_interest_rates, f"economic_processes[{index}] {identity}")
            invoice_drivers = entry["drivers"]
            _require(invoice_drivers.get("invoice_opening_reference_kind") == entry.get("timing_reference_kind"), f"economic_processes[{index}]: invoice carry-forward scope differs from the documented business chronology")
            prior_process = process_by_month.get((entry["enterprise_id"], preceding_month))
            if prior_process is not None:
                prior_drivers = prior_process["drivers"]
                continuity_fields = (("sales_invoice_opening_pending", "sales_invoice_closing_pending"), ("purchase_invoice_opening_pending", "purchase_invoice_closing_pending"), ("utility_invoice_opening_pending", "utility_invoice_closing_pending"), ("travel_hospitality_invoice_opening_pending", "travel_hospitality_invoice_closing_pending"), ("previous_inventory", "closing_inventory"))
                _require(all(_decimal(invoice_drivers.get(current), current, 2) == _decimal(prior_drivers.get(previous), previous, 2) for current, previous in continuity_fields), f"economic_processes[{index}]: invoice pending queues or inventory stock do not continue from the final preceding business month")
                _require(all(invoice_drivers.get(field) == prior_drivers.get(field) for field in ("opening_inventory", "normal_monthly_input_budget", "inventory_buffer_months", "opening_sales_pending_rate", "opening_purchase_pending_rate", "opening_procurement_basis_estimate", "utility_opening_prototype_base", "travel_opening_prototype_base", "utility_purchase_share", "travel_purchase_share")), f"economic_processes[{index}]: preselected opening/capacity invoice and inventory inputs changed between months")
            else:
                _require(_decimal(invoice_drivers.get("previous_inventory"), "previous_inventory", 2) == _decimal(invoice_drivers.get("opening_inventory"), "opening_inventory", 2), f"economic_processes[{index}]: first business month does not start from its documented opening inventory")
                if established >= _day(config["window_start"], "window_start"):
                    _require(all(_decimal(invoice_drivers.get(field), field, 2) == 0 for field in ("sales_invoice_opening_pending", "purchase_invoice_opening_pending", "opening_procurement_basis_estimate", "utility_invoice_opening_pending", "travel_hospitality_invoice_opening_pending", "utility_opening_prototype_base", "travel_opening_prototype_base")), f"economic_processes[{index}]: a newly established enterprise has pre-establishment sales or purchase invoices pending")
                else:
                    sales_rate = _decimal(invoice_drivers.get("opening_sales_pending_rate"), "opening_sales_pending_rate", 6)
                    purchase_rate = _decimal(invoice_drivers.get("opening_purchase_pending_rate"), "opening_purchase_pending_rate", 6)
                    _require(Decimal(0) <= sales_rate <= 1 and Decimal(0) <= purchase_rate <= 1 and _decimal(invoice_drivers.get("sales_invoice_opening_pending"), "sales_invoice_opening_pending", 2) == _cents(previous_revenue * sales_rate) and _decimal(invoice_drivers.get("purchase_invoice_opening_pending"), "purchase_invoice_opening_pending", 2) == _cents(_decimal(invoice_drivers.get("opening_procurement_basis_estimate"), "opening_procurement_basis_estimate", 2) * purchase_rate), f"economic_processes[{index}]: window-opening pending invoices do not follow the explicit prototype estimate")
                    prototype_utility = _cents(_decimal(invoice_drivers["base_overhead"], "base_overhead", 6) * _decimal(invoice_drivers["utility_purchase_share"], "utility_purchase_share", 6))
                    prototype_travel = _cents(previous_revenue * _decimal(invoice_drivers["variable_overhead_rate"], "variable_overhead_rate", 6) * _decimal(invoice_drivers["travel_purchase_share"], "travel_purchase_share", 6))
                    _require(_decimal(invoice_drivers.get("utility_opening_prototype_base"), "utility_opening_prototype_base", 2) == prototype_utility and _decimal(invoice_drivers.get("travel_opening_prototype_base"), "travel_opening_prototype_base", 2) == prototype_travel and _decimal(invoice_drivers.get("utility_invoice_opening_pending"), "utility_invoice_opening_pending", 2) == _cents(prototype_utility * purchase_rate) and _decimal(invoice_drivers.get("travel_hospitality_invoice_opening_pending"), "travel_hospitality_invoice_opening_pending", 2) == _cents(prototype_travel * purchase_rate), f"economic_processes[{index}]: window-opening energy or travel invoice components do not follow their explicit expense prototype")
            preceding_processes = [process for (eid, stat_month), process in process_by_month.items() if eid == entry["enterprise_id"] and stat_month < entry["stat_month"]]
            prior_accrual = sum((_decimal(process["income_tax_expense"], "income_tax_expense", 2) for process in preceding_processes), Decimal(0))
            prior_paid = sum((_decimal(process["tax_payment"], "tax_payment", 2) for process in preceding_processes), Decimal(0))
            _require(_decimal(entry["drivers"].get("prior_cumulative_income_tax_expense"), "prior_cumulative_income_tax_expense", 2) == prior_accrual and _decimal(entry["drivers"].get("prior_cumulative_tax_paid"), "prior_cumulative_tax_paid", 2) == prior_paid, f"economic_processes[{index}]: tax notification refers to prior values that differ from the final canonical accrual and cash process")
            lag_ordinal = month.year * 12 + month.month - 1 - 3
            lag_month = date(lag_ordinal // 12, lag_ordinal % 12 + 1, 1).isoformat()
            lag_fact = canonical_months.get((entry["enterprise_id"], lag_month))
            drivers = entry["drivers"]
            lag_reference = _decimal(drivers.get("lagged_capacity_reference_revenue"), "lagged_capacity_reference_revenue", 2)
            if lag_fact is not None:
                _require(drivers.get("staffing_reference_kind") == "three_month_lagged_canonical_revenue" and drivers.get("staffing_reference_month") == lag_month and lag_reference == _money(lag_fact, "operating_revenue", "operating_revenue"), f"economic_processes[{index}]: staffing reference does not match the final canonical month three months earlier")
            else:
                _require(drivers.get("staffing_reference_kind") == "opening_staffing_plan_prototype" and drivers.get("staffing_reference_month") is None and lag_reference == _cents(_decimal(drivers.get("staffing_reference_revenue"), "staffing_reference_revenue", 6)), f"economic_processes[{index}]: initial staffing estimate is incorrectly represented as a historical operating fact")
        _require(isinstance(entry.get("mechanism"), str) and bool(entry["mechanism"].strip()), f"economic_processes[{index}]: missing business mechanism")
        seen.add(identity)
    _require(seen == set(latest), "facts: economic process ledger does not cover all business months")
    original_report_process_count = (_reporting_estimate_checks(processes, private["truth_tables"]["enterprise_operation_monthly"], latest, priced_loans, priced_bonds, public_interest_rates) if reviewed else 0)
    contracts = defaultdict(list)
    for row in private["truth_tables"]["enterprise_operation_event"]:
        if row["event_type"] in {"银行借款", "借款逾期"}:
            contracts[(row["enterprise_id"], row["business_object_ref"])].append(row)
    debt_seen = set()
    for index, entry in enumerate(debt):
        _require(isinstance(entry, dict), f"debt_ledgers[{index}]: expected an object")
        identity = (entry.get("enterprise_id"), entry.get("business_object_ref"))
        _require(identity in contracts and identity not in debt_seen, f"debt_ledgers[{index}]: missing/duplicate contract")
        principal = _decimal(entry.get("original_principal"), "original_principal", 2)
        _require(principal is not None and principal >= 0, f"debt_ledgers[{index}]: invalid original principal")
        repayments, checkpoints = entry.get("repayments"), entry.get("checkpoints")
        _require(isinstance(repayments, list) and isinstance(checkpoints, list) and checkpoints, f"debt_ledgers[{index}]: incomplete repayment history")
        repayment_values = []
        for payment in repayments:
            day, amount = _day(payment["payment_date"], "payment_date"), _decimal(payment["principal_amount"], "principal_amount", 2)
            _require(amount is not None and amount >= 0, f"debt_ledgers[{index}]: negative/missing principal repayment")
            repayment_values.append((day, amount))
        snapshots = {}
        for checkpoint in checkpoints:
            day = _day(checkpoint["balance_as_of_date"], "balance_as_of_date")
            paid, outstanding, overdue = (_decimal(checkpoint[field], field, 2) for field in ("cumulative_principal_repaid", "outstanding_principal", "overdue_principal"))
            _require(None not in (paid, outstanding, overdue) and paid >= 0 and outstanding >= 0 and 0 <= overdue <= outstanding, f"debt_ledgers[{index}]: invalid principal checkpoint")
            _require(paid + outstanding == principal and paid == sum(amount for when, amount in repayment_values if when <= day), f"debt_ledgers[{index}]: principal/repayment ledger identity failed")
            _require(day not in snapshots, f"debt_ledgers[{index}]: duplicate balance date")
            snapshots[day] = (outstanding, overdue)
        for row in contracts[identity]:
            if row["event_type"] == "银行借款":
                _require(_money(row, "amount", "amount") == principal, f"debt_ledgers[{index}]: contract principal differs")
            if row["balance_as_of_date"] is not None:
                day = _day(row["balance_as_of_date"], "balance_as_of_date")
                _require(day in snapshots and snapshots[day] == (_money(row, "outstanding_principal", "outstanding_principal"), _money(row, "overdue_principal", "overdue_principal")), f"debt_ledgers[{index}]: reported principal snapshot differs from the ledger")
        debt_seen.add(identity)
    _require(debt_seen == set(contracts), "facts: debt ledger does not cover all loan contracts")
    profile_rows = {_key("enterprise_operation_profile", row): row for row in private["truth_tables"]["enterprise_operation_profile"]}
    enterprise_basics = {row["enterprise_id"]: row for row in private["truth_tables"]["enterprise_basic"]}
    balance_seen = set()
    prior_by_enterprise = {}
    prior_owner_totals = {}
    independent_plans = {}
    for index, entry in enumerate(sorted(balances, key=lambda item: (item["enterprise_id"], item["as_of_date"], item["report_revision"]))):
        reference = entry.get("record_ref", {})
        _require(reference.get("table") == "enterprise_operation_profile", f"balance_processes[{index}]: missing public snapshot reference")
        primary_key = reference.get("key", reference.get("primary_key"))
        _require(isinstance(primary_key, dict) and set(primary_key) == set(PRIMARY_KEYS["enterprise_operation_profile"]), f"balance_processes[{index}]: incomplete snapshot key")
        key = tuple(primary_key[field] for field in PRIMARY_KEYS["enterprise_operation_profile"])
        _require(key in profile_rows and key not in balance_seen, f"balance_processes[{index}]: missing/duplicate stock snapshot")
        row = profile_rows[key]
        _require((entry["enterprise_id"], entry["as_of_date"], entry["report_revision"]) == (row["enterprise_id"], row["as_of_date"], row["report_revision"]), f"balance_processes[{index}]: stock date/revision identity differs")
        fields = ("cash", "trade_receivables", "inventory", "fixed_and_other_assets", "total_assets", "bank_loan_principal", "trade_payables", "other_liabilities", "total_liabilities", "owners_equity", "retained_profit_change", "owner_and_other_equity_change")
        values = {field: _decimal(entry.get(field), f"balance_processes[{index}].{field}", 2) for field in fields}
        _require(None not in values.values(), f"balance_processes[{index}]: incomplete stock components")
        _require(all(values[field] >= 0 for field in fields[:9]), f"balance_processes[{index}]: negative assets/liability components")
        _require(sum(values[field] for field in ("cash", "trade_receivables", "inventory", "fixed_and_other_assets")) == values["total_assets"] and sum(values[field] for field in ("bank_loan_principal", "trade_payables", "other_liabilities")) == values["total_liabilities"], f"balance_processes[{index}]: asset/liability component identity failed")
        _require(values["total_assets"] - values["total_liabilities"] == values["owners_equity"], f"balance_processes[{index}]: equity stock identity failed")
        _require(_money(row, "total_assets", "total_assets") == values["total_assets"] and _money(row, "total_liabilities", "total_liabilities") == values["total_liabilities"], f"balance_processes[{index}]: stock ledger differs from the public fact")
        day = _day(entry["as_of_date"], "as_of_date")
        expected_loan_balance = Decimal(0)
        for contract in debt:
            if contract["enterprise_id"] != entry["enterprise_id"]:
                continue
            applicable = [point for point in contract["checkpoints"] if _day(point["balance_as_of_date"], "balance_as_of_date") <= day]
            if applicable:
                point = max(applicable, key=lambda item: _day(item["balance_as_of_date"], "balance_as_of_date"))
                expected_loan_balance += _decimal(point["outstanding_principal"], "outstanding_principal", 2)
        _require(values["bank_loan_principal"] == expected_loan_balance, f"balance_processes[{index}]: stock loan balance must use the latest contract checkpoint, not add periods")
        if reviewed:
            enterprise_id = entry["enterprise_id"]
            established = _day(enterprise_basics[enterprise_id]["established_date"], "established_date")
            opening_day = max(_day(config["window_start"], "window_start") - timedelta(days=1), established)
            opening_kind = "registration_opening_funding_estimate" if established >= _day(config["window_start"], "window_start") else "window_opening_prototype_estimate"
            _require(_day(entry.get("opening_stock_date"), "opening_stock_date") == opening_day and entry.get("opening_stock_kind") == opening_kind, f"balance_processes[{index}]: initial stock does not follow enterprise establishment and observation-window scope")
            plan = entry.get("funding_plan")
            _require(enterprise_id not in independent_plans or independent_plans[enterprise_id] == plan, f"balance_processes[{index}]: the funding plan was changed between snapshots or revisions")
            totals = _stock_driver_check(entry, values, process_by_month, private["truth_tables"]["enterprise_operation_event"], config, prior_owner_totals.get(enterprise_id), f"balance_processes[{index}] {enterprise_id} {day}", established=established)
            independent_plans[enterprise_id] = plan
            prior_owner_totals[enterprise_id] = totals
        retained = sum((_money(month, "net_profit", "net_profit") for month in latest.values() if month["enterprise_id"] == entry["enterprise_id"] and _day(month["period_end"], "period_end") <= day), Decimal(0))
        prior = prior_by_enterprise.get(entry["enterprise_id"])
        if prior is None:
            _require(values["retained_profit_change"] == 0 and values["owner_and_other_equity_change"] == 0, f"balance_processes[{index}]: initial snapshot must be marked as initial stock")
        else:
            equity, previous_retained = prior
            _require(values["retained_profit_change"] == retained - previous_retained and values["owners_equity"] - equity == values["retained_profit_change"] + values["owner_and_other_equity_change"], f"balance_processes[{index}]: equity/profit bridge failed")
        prior_by_enterprise[entry["enterprise_id"]] = (values["owners_equity"], retained)
        balance_seen.add(key)
    _require(balance_seen == set(profile_rows), "facts: balance process ledger does not cover every stock snapshot/version")
    return {"economic_business_months": len(seen), "loan_contracts": len(debt_seen), "stock_snapshots": len(balance_seen), **({"lawful_original_report_processes": original_report_process_count, "independent_funding_plans": len(independent_plans), "stock_driver_snapshots": len(balance_seen), "publicly_priced_financing_contracts": len(public_interest_rates)} if reviewed else {})}


def _sampling_checks(statistics: Mapping[str, Any], ledger: list[dict[str, Any]], assignments: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    sampling = statistics.get("sampling")
    _require(isinstance(sampling, dict) and set(sampling) == OPERATIONS, "pollution statistics: incomplete operation sampling denominators")
    for operation in sorted(OPERATIONS):
        values = sampling[operation]
        _require(isinstance(values, dict), f"sampling.{operation}: expected an object")
        for field in ("eligible", "guaranteed", "sampled_candidates", "sampled_selected", "applied_original_records"):
            value = values.get(field)
            _require(isinstance(value, int) and not isinstance(value, bool) and value >= 0, f"sampling.{operation}.{field}: expected a nonnegative integer")
        _require(values["guaranteed"] == sum(operation in assignment["classes"] for assignment in assignments.values()), f"sampling.{operation}: guarantee count differs from assigned companies")
        _require(values["sampled_selected"] <= values["sampled_candidates"] and values["guaranteed"] + values["sampled_candidates"] <= values["eligible"], f"sampling.{operation}: sampled/eligible denominators are inconsistent")
        entries = [entry for entry in ledger if entry["operation"] == operation]
        if operation == "missing_month":
            monthly_entries = [entry for entry in entries if entry["table"] == "enterprise_operation_monthly" and entry["after"] is None]
            selected = len({(entry["before"]["enterprise_id"], entry["before"]["stat_month"]) for entry in monthly_entries})
            # Missing-month sampling uses enterprise-months. Removing a month
            # removes all its physical versions; coverage updates are evidence,
            # rather than extra selected months or original monthly records.
            applied_records = len(monthly_entries)
        else:
            selected = applied_records = len(entries)
        _require(values["guaranteed"] + values["sampled_selected"] == selected and values["applied_original_records"] == applied_records, f"sampling.{operation}: actual applied samples differ from the ledger")
        configured = values.get("configured_probability")
        _require(isinstance(configured, str) and Decimal(configured).is_finite() and Decimal(0) <= Decimal(configured) <= Decimal(1) and Decimal(configured) == Decimal(str(config["pollution"]["rates"][operation])), f"sampling.{operation}: configured probability differs")
        observed = values.get("observed_selected_fraction")
        if values["sampled_candidates"]:
            expected = Decimal(values["sampled_selected"]) / Decimal(values["sampled_candidates"])
            _require(isinstance(observed, str) and Decimal(observed).is_finite() and Decimal(observed) == expected, f"sampling.{operation}: actual sampled fraction differs from its denominator")
        else:
            _require(observed is None and values["sampled_selected"] == 0, f"sampling.{operation}: zero-denominator fraction must be unknown")
    return deepcopy(sampling)


def _validate_dataset(tables: dict[str, Any], private: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Validate the dataset or raise ValueError; never silently repair records."""
    _require(isinstance(tables, dict) and isinstance(private, dict) and isinstance(config, dict), "dataset/config/private must be objects")
    fixed_config = {"schema_version": "3.0", "enterprise_count": 120, "seed": 20261012, "reference_date": "2026-08-31", "knowledge_cutoff": "2026-09-15 23:59:59", "window_start": "2024-09-01", "window_end": "2026-08-31", "profile_dates": ["2025-08-31", "2025-12-31", "2026-03-31", "2026-08-31"]}
    _require(all(config.get(name) == value for name, value in fixed_config.items()), "config: fixed baseline/date/seed/lifecycle contract differs")
    _require(isinstance(config.get("baseline_id"), str) and re.fullmatch(r"synthetic_enterprise_v30_120_v[1-9]\d*\.\d+\.\d+", config["baseline_id"]) is not None, "config: invalid independent V3.0 baseline version")
    _require(config.get("pollution", {}).get("max_operations_per_record") == 2 and config.get("pollution", {}).get("max_consecutive_missing_months") == 2, "config: max two operations per row and max two consecutive omitted months are required")
    _require(isinstance(private.get("truth_tables"), dict) and isinstance(private.get("base_observations"), dict), "private: missing facts or original observations")
    _require(isinstance(private.get("assignments"), dict) and isinstance(private.get("pollution_ledger"), list), "private: missing assignments or pollution ledger")
    truth = _maps(private["truth_tables"], "truth")
    observed = _maps(tables, "observations")
    _identity_and_relations(private["truth_tables"], "truth")
    _identity_and_relations(tables, "observations")
    _normal_facts(private["truth_tables"], config)
    groups = _assignment_checks(private["assignments"])
    _identity_and_relations(private["base_observations"], "base_observations")
    _normal_facts(private["base_observations"], config)
    _require(truth == _maps(private["base_observations"], "base_observations"), "private: unexplained differences between true facts and lawful original observations")
    pollution = _ledger_checks(tables, private["base_observations"], private["pollution_ledger"], private["assignments"], config)
    scenarios = _scenario_checks(private.get("scenarios", []), truth, observed, config)
    processes = _process_checks(private, config)
    public_description_count = _public_description_checks(tables, config)
    statistics = private.get("pollution_statistics")
    _require(isinstance(statistics, dict), "private: missing actual pollution statistics")
    expected_statistics = {"base_observation_count": pollution["eligible_original_observations"], "modified_or_missing_original_records": pollution["altered_or_omitted_originals"], "modified_fraction": pollution["altered_or_omitted_ratio"], "added_duplicate_records": pollution["added_duplicate_records"], "operations": pollution["operations"]}
    _require(all(statistics.get(name) == value for name, value in expected_statistics.items()), "private: actual pollution statistics differ from exact ledger reconstruction")
    pollution["sampling"] = _sampling_checks(statistics, private["pollution_ledger"], private["assignments"], config)
    # Explicitly observable edge cases, not hidden scenario labels alone.
    zero_enterprises = set()
    same_value_enterprises = set()
    revision_enterprises = set()
    ratio_comparison_enterprises = set()
    for table in TABLES:
        amounts_by_enterprise = defaultdict(list)
        revisions_by_identity = defaultdict(set)
        for row in private["truth_tables"][table]:
            enterprise_id = row.get("enterprise_id")
            if "report_revision" in row:
                basis_fields = ["source_system", "source_record_ref"]
                identity = (enterprise_id,) + tuple(row[field] for field in basis_fields)
                revisions_by_identity[identity].add(row["report_revision"])
            for field in MONEY_FIELDS.get(table, ()):
                value = _money(row, field, field)
                if value == 0:
                    zero_enterprises.add(enterprise_id)
            if table == "enterprise_operation_event" and row.get("amount") is not None:
                amounts_by_enterprise[enterprise_id].append((_money(row, "amount", "amount"), row.get("business_event_ref")))
            elif table == "enterprise_operation_monthly":
                amounts_by_enterprise[enterprise_id].append((_money(row, "operating_revenue", "operating_revenue"), row["stat_month"]))
        for identity, versions in revisions_by_identity.items():
            if len(versions) > 1:
                revision_enterprises.add(identity[0])
        for enterprise_id, values in amounts_by_enterprise.items():
            identities_by_amount = defaultdict(set)
            for amount, event_ref in values:
                identities_by_amount[amount].add(event_ref)
            if any(len(refs) >= 2 for refs in identities_by_amount.values()):
                same_value_enterprises.add(enterprise_id)
    latest_months = {}
    for row in private["truth_tables"]["enterprise_operation_monthly"]:
        identity = (row["enterprise_id"], row["source_system"], row["source_record_ref"])
        if identity not in latest_months or row["report_revision"] > latest_months[identity]["report_revision"]:
            latest_months[identity] = row
    ratios_by_enterprise = defaultdict(list)
    for row in latest_months.values():
        if row["report_status"] != "正式":
            continue
        revenue = _money(row, "operating_revenue", "operating_revenue")
        profit = _money(row, "net_profit", "net_profit")
        if revenue is not None and revenue > 0 and profit is not None:
            ratios_by_enterprise[row["enterprise_id"]].append((profit, revenue))
    for enterprise_id, amounts in ratios_by_enterprise.items():
        if len(amounts) < 2:
            continue
        average_ratio = sum(profit / revenue for profit, revenue in amounts) / Decimal(len(amounts))
        aggregate_ratio = sum(profit for profit, _ in amounts) / sum(revenue for _, revenue in amounts)
        if abs(average_ratio - aggregate_ratio) > Decimal("0.000000000001"):
            ratio_comparison_enterprises.add(enterprise_id)
    _require(len(zero_enterprises) >= 12, "facts: fewer than 12 enterprises with genuine zero monetary values")
    _require(len(same_value_enterprises) >= 12, "facts: fewer than 12 enterprises with different real months/events having equal amounts")
    _require(len(revision_enterprises) >= 12, "facts: fewer than 12 enterprises with legitimate revisions")
    _require(len(ratio_comparison_enterprises) >= 12, "facts: fewer than 12 observable average/aggregate ratio counterexamples")
    public_edges = _public_edges(tables)
    requirements = config.get("coverage_requirements", {})
    duplicate_enterprises = {entry["before"]["enterprise_id"] for entry in private["pollution_ledger"] if entry["operation"] == "duplicate_submission"}
    paired_enterprises = same_value_enterprises & public_edges["same_value"] & duplicate_enterprises
    _require(statistics.get("equal_real_and_duplicate_enterprises") == len(paired_enterprises), "pollution statistics: same-value/duplicate enterprise count differs from the public evidence intersection")
    visible_zero_enterprises = zero_enterprises & public_edges["zero"]
    visible_revision_enterprises = revision_enterprises & public_edges["revision"]
    visible_ratio_enterprises = ratio_comparison_enterprises & public_edges["ratio"]
    _require(len(paired_enterprises) >= max(12, int(requirements.get("minimum_equal_real_and_duplicate_enterprises", 12))), "observations: fewer than 12 enterprises visibly carry both distinct real equal-value facts and duplicate reports")
    _require(len(visible_zero_enterprises) >= max(12, int(requirements.get("minimum_zero_enterprises", 12))), "observations: insufficient publicly observable genuine zero values")
    _require(len(visible_revision_enterprises) >= max(12, int(requirements.get("minimum_revision_enterprises", 12))), "observations: insufficient publicly observable lawful revisions")
    _require(len(visible_ratio_enterprises) >= max(12, int(requirements.get("minimum_ratio_comparison_enterprises", 12))), "observations: insufficient publicly observable ratio-formula differences")
    _require(len(public_edges["unknown"]) >= 12, "observations: fewer than 12 enterprises have explicitly unknown applicable fields")
    feature_counts = Counter()
    profiles = private.get("profiles", [])
    _require(isinstance(profiles, list) and len(profiles) == 120 and {profile.get("enterprise_id") for profile in profiles} == EXPECTED_IDS, "profiles: expected one management profile per enterprise")
    profile_groups = defaultdict(set)
    for profile in profiles:
        feature_counts.update(set(profile.get("business_features", [])))
        enterprise_id = profile["enterprise_id"]
        number = int(enterprise_id[-6:])
        _require(profile.get("anchor_enterprise_id") == f"SYN-E{(number - 1) % 40 + 1:06d}", f"profiles: wrong original anchor for {enterprise_id}")
        _require(profile.get("profile_type_id") == f"P{(number - 1) % 40 + 1:03d}", f"profiles: wrong profile family for {enterprise_id}")
        profile_groups[profile["profile_type_id"]].add(enterprise_id)
        if _reviewed_contract(config):
            _management_profile_check(profile, truth, latest_months)
    _require(len(profile_groups) == 40 and all(len(enterprise_ids) == 3 for enterprise_ids in profile_groups.values()), "profiles: expected 40 profile types with three enterprises each")
    scales = {profile["enterprise_id"]: profile["scale"] for profile in profiles}
    strata = defaultdict(lambda: {"none": 0, "single": 0, "double": 0})
    for enterprise in tables["enterprise_basic"]:
        enterprise_id = enterprise["enterprise_id"]
        cohort = "old" if int(enterprise_id[-6:]) <= 40 else "new"
        strata[(cohort, enterprise["industry"], scales[enterprise_id])][private["assignments"][enterprise_id]["group"]] += 1
    actual_strata = [{"cohort": identity[0], "industry": identity[1], "scale": identity[2], "groups": groups} for identity, groups in sorted(strata.items())]
    _require(statistics.get("strata") == actual_strata, "pollution statistics: cohort/industry/scale group counts differ from enterprise assignments")
    pollution["strata"] = actual_strata
    canonical_months = {(row["enterprise_id"], row["stat_month"]) for row in private["truth_tables"]["enterprise_operation_monthly"]}
    canonical_snapshots = {(row["enterprise_id"], row["as_of_date"]) for row in private["truth_tables"]["enterprise_operation_profile"]}
    _require(len(canonical_months) <= 2875, "facts: monthly lifecycle grid exceeds the establishment-conditioned maximum 2875")
    basic_rows = {row["enterprise_id"]: row for row in tables["enterprise_basic"]}
    for enterprise_id, row in basic_rows.items():
        if int(enterprise_id[-6:]) > 40:
            _require(_day(row["established_date"], "established_date") < _day(config["window_start"], "window_start"), f"facts: new enterprise {enterprise_id} was established inside the observation window")
    for row in private["truth_tables"]["enterprise_operation_profile"]:
        _require(str(row["as_of_date"]) in config["profile_dates"] and _day(row["as_of_date"], "as_of_date") >= _day(basic_rows[row["enterprise_id"]]["established_date"], "established_date"), "facts: invalid snapshot date/lifecycle")
    _require(canonical_snapshots == {(enterprise_id, day) for enterprise_id in EXPECTED_IDS for day in config["profile_dates"] if _day(day, "profile_date") >= _day(basic_rows[enterprise_id]["established_date"], "established_date")}, "facts: missing configured lifecycle snapshots")
    cutoff = _instant(config["knowledge_cutoff"], "knowledge_cutoff")
    for table, rows in tables.items():
        for row in rows:
            if "ingested_at" in row:
                _require(_instant(row["ingested_at"], "ingested_at") <= cutoff, f"observations.{table}: ingestion exceeds the collection cutoff")
            elif "source_updated_at" in row:
                _require(_instant(row["source_updated_at"], "source_updated_at") <= cutoff, f"observations.{table}: source update exceeds the collection cutoff")
            elif "last_synced_at" in row:
                _require(_instant(row["last_synced_at"], "last_synced_at") <= cutoff, f"observations.{table}: coverage sync exceeds the collection cutoff")
    table_counts = {table: len(tables[table]) for table in TABLES}
    reviewed_checks = {name: "PASS" for name in (
        "final_previous_month_dependency", "independent_business_driver_formulas",
        "lawful_original_report_estimate_process", "public_rate_daily_principal_interest",
        "staffing_three_month_fact_reference", "comprehensive_payroll_includes_social_cost",
        "named_funding_and_stock_drivers", "cash_tax_accrual_balance",
        "retained_earnings_distribution_budget", "pre_window_distribution_maturity",
        "may_known_profit_distribution_decision", "actual_post_stock_risk_classification",
        "conserved_invoice_pending_queues", "actual_procurement_cost_stock_capital_basis",
        "energy_travel_expense_invoice_components", "monthly_inventory_snapshot_alignment",
        "management_profile_fact_alignment", "genuine_risk_feature_predicates",
        "response_type_contract", "category_specific_learning_roles",
        "observable_scenario_predicates", "context_scope_transitions",
        "public_description_target_conclusions",
    )} if _reviewed_contract(config) else {}
    return {
        "status": "PASS", "schema_version": "3.0", "enterprise_count": 120,
        "industry_counts": dict(Counter(row["industry"] for row in tables["enterprise_basic"])),
        "record_counts": table_counts, "table_counts": table_counts,
        "truth_record_counts": {table: len(private["truth_tables"][table]) for table in TABLES},
        "pollution_groups": groups, "assignment_counts": groups,
        "assignment_breakdown": {"old": dict(OLD_GROUP_COUNTS), "new": dict(NEW_GROUP_COUNTS)},
        "pollution": pollution, "scenarios": scenarios,
        "base_facts": {"feature_counts": dict(feature_counts), "canonical_month_count": len(canonical_months), "canonical_snapshot_count": len(canonical_snapshots), "maximum_lifecycle_months": 2875, "financial_processes": processes, "public_descriptions_reviewed": public_description_count},
        "edge_cases": {"genuine_zero_enterprises": len(visible_zero_enterprises), "equal_value_distinct_fact_enterprises": len(same_value_enterprises & public_edges["same_value"]), "equal_real_and_duplicate_enterprises": len(paired_enterprises), "legitimate_revision_enterprises": len(visible_revision_enterprises), "ratio_comparison_enterprises": len(visible_ratio_enterprises), "explicit_unknown_enterprises": len(public_edges["unknown"])},
        "checks": {"technical_identity_and_relations": "PASS", "truth_business_invariants": "PASS", "income_cost_profit_process": "PASS", "no_pre_establishment_operating_reference": "PASS", "loan_principal_repayment_process": "PASS", "asset_liability_stock_process": "PASS", "exact_pollution_ledger_replay": "PASS", "pollution_sampling_statistics": "PASS", "scenario_coverage": "PASS", "public_edge_evidence": "PASS", "monetary_precision": "PASS", **reviewed_checks},
    }


def validate_dataset(tables: dict[str, Any], private: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Return a report or raise ValueError for any malformed dataset contract."""
    try:
        return _validate_dataset(tables, private, config)
    except ValueError:
        raise
    except (KeyError, TypeError, ArithmeticError, AttributeError, IndexError) as error:
        raise ValueError(f"invalid V3.0 dataset contract: {error}") from error
