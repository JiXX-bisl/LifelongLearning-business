"""确定性构建和十进制文件工具，不连接数据库或模型。"""
from __future__ import annotations

import hashlib
import json
import random
import re
from datetime import date, datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OPERATIONS = (
    "duplicate_submission", "missing_field", "missing_month", "delayed_report",
    "unit_representation", "cross_field_conflict",
)
TARGET_CATEGORIES = ("A1", "A2", "A4", "A5", "B2", "B3", "C1", "C2", "C4", "C6")


def resolve(path):
    value = Path(path).expanduser()
    return (value if value.is_absolute() else ROOT / value).resolve()


def decimal(value):
    if isinstance(value, (bool, float)):
        raise ValueError("金额及生成参数不能通过布尔值或浮点数传递")
    result = value if isinstance(value, Decimal) else Decimal(str(value))
    if not result.is_finite():
        raise ValueError("金额及生成参数必须是有限十进制数")
    return result


def money(value):
    return format(decimal(value).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP), ".2f")


def observed_money(value, unit="元"):
    if unit not in {"元", "万元"}:
        raise ValueError(f"未知金额单位：{unit}")
    divisor = Decimal(10000) if unit == "万元" else Decimal(1)
    places = Decimal("0.000001") if unit == "万元" else Decimal("0.01")
    return format((decimal(value) / divisor).quantize(places, rounding=ROUND_HALF_UP), "f")


def rng(seed, *parts):
    payload = json.dumps([str(seed), *[str(part) for part in parts]],
                         ensure_ascii=False, separators=(",", ":"))
    return random.Random(int(hashlib.sha256(payload.encode("utf-8")).hexdigest(), 16))


def month_end(value):
    value = date.fromisoformat(str(value)[:10]) if not isinstance(value, date) else value
    following = date(value.year + int(value.month == 12), value.month % 12 + 1, 1)
    return following - timedelta(days=1)


def dt(value, days=0):
    text = str(value)
    parsed = datetime.fromisoformat(text) if len(text) > 10 else datetime.combine(
        date.fromisoformat(text), datetime.min.time())
    return (parsed + timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"JSON键重复：{key}")
            result[key] = value
        return result
    return json.loads(resolve(path).read_text(encoding="utf-8"), parse_float=Decimal,
                      object_pairs_hook=unique)


def _json_default(value):
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, (date, datetime)):
        return value.isoformat(sep=" ") if isinstance(value, datetime) else value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"不能序列化：{type(value).__name__}")


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       default=_json_default, allow_nan=False) + "\n").encode("utf-8")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(value))


def file_hash(path):
    with Path(path).open("rb") as stream:
        result = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def digest(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def load_config(path):
    config = read_json(path)
    if config.get("enterprise_count") != 120 or config.get("schema_version") != "3.0":
        raise ValueError("本构建配置必须为120家、Schema 3.0")
    if config.get("mysql", {}).get("database") != "synthetic_enterprise_v30_120":
        raise ValueError("本构建只生成独立的新数据库 synthetic_enterprise_v30_120")
    required = {
        "seed": 20261012,
        "reference_date": "2026-08-31", "knowledge_cutoff": "2026-09-15 23:59:59",
        "window_start": "2024-09-01", "window_end": "2026-08-31",
        "profile_dates": ["2025-08-31", "2025-12-31", "2026-03-31", "2026-08-31"],
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f"固定构建契约不符：{key} 应为 {expected!r}")
    if not re.fullmatch(r"synthetic_enterprise_v30_120_v[1-9]\d*\.\d+\.\d+", str(config.get("baseline_id", ""))):
        raise ValueError("基线身份必须使用 synthetic_enterprise_v30_120_v<版本> 格式")
    for name in ("source_data", "source_profiles", "scenario_definitions", "public_output", "private_output"):
        config[name] = str(resolve(config[name]))
    for name in ("source_data", "source_profiles", "scenario_definitions"):
        if not Path(config[name]).is_file():
            raise FileNotFoundError(config[name])
    public, private = Path(config["public_output"]), Path(config["private_output"])
    if (not public.is_relative_to(ROOT / "data/sql")
            or not private.is_relative_to(ROOT / "data/evaluation")
            or public == private or public.is_relative_to(private) or private.is_relative_to(public)):
        raise ValueError("公共与私有产物必须分别位于 data/sql 和 data/evaluation 的独立目录")
    if public.name != config["baseline_id"] or private.name != config["baseline_id"]:
        raise ValueError("公共与私有目录名必须与独立基线身份一致")
    if set(config["pollution"]["rates"]) != set(OPERATIONS):
        raise ValueError("污染配置必须包含六类已确认操作")
    for value in config["pollution"]["rates"].values():
        if not Decimal(0) <= decimal(value) <= Decimal(1):
            raise ValueError("污染概率必须在0至1之间")
    pollution = config["pollution"]
    if (pollution["old_distribution"] != {"none": 10, "single": 20, "double": 10}
            or pollution["new_distribution"] != {"none": 20, "single": 40, "double": 20}):
        raise ValueError("企业污染分布必须为原40家10/20/10，新增80家20/40/20")
    if not 0 < decimal(pollution["max_modified_fraction"]) <= Decimal("0.15"):
        raise ValueError("原始观测记录修改与缺报比例上限必须不超过15%")
    if pollution["max_operations_per_record"] != 2 or pollution["max_consecutive_missing_months"] != 2:
        raise ValueError("单行最多两类操作且连续缺报最多两个月")
    if tuple(config["coverage_requirements"]["categories"]) != TARGET_CATEGORIES:
        raise ValueError("场景配置必须完整覆盖十类优先问题")
    retained = config.get("retained_candidates", [])
    if (not isinstance(retained, list) or len(retained) != len(set(retained))
            or any(not isinstance(name, str) or not re.fullmatch(
                r"synthetic_enterprise_v30_120_v[1-9]\d*\.\d+\.\d+", name)
                   or name == config["baseline_id"] for name in retained)):
        raise ValueError("保留候选清单必须使用独立且有效的旧基线身份")
    return config
