"""本地来源适配：缺失仅作标注，明确冲突会阻断使用。"""
from __future__ import annotations

import hashlib
from pathlib import Path


def original_version(record):
    provenance = record.get("provenance", {})
    if "original_data_version" in provenance:
        return provenance["original_data_version"]
    return record["task"]["data_version"]  # 兼容第一版已入库记录。


def inspect_source(record, expected_sha, import_state, resolve_path, locations=()):
    task = record["task"]
    version = original_version(record)
    warnings, conflicts, verified = [], [], []
    if version is None:
        warnings.append("original_version_unknown")
    elif version != task["data_version"]:
        conflicts.append("data_version_mismatch")
    if import_state == "conflict":
        conflicts.append("import_content_conflict")
    references = [task.get("source_ref"), record.get("original_source_ref"), *locations]
    paths = set()
    for ref in references:
        if ref:
            try:
                paths.add(Path(resolve_path(ref)).resolve())
            except (OSError, ValueError):
                warnings.append("source_path_unresolvable")
    for path in sorted(paths):
        try:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            continue
        if actual == expected_sha:
            verified.append(str(path))
        else:
            conflicts.append("source_content_changed")
    if not verified:
        warnings.append("source_unverifiable")
    return {
        "status": "conflict" if conflicts else ("verified" if verified else "unverifiable"),
        "original_data_version": version,
        "target_data_version": task["data_version"],
        "warnings": sorted(set(warnings)),
        "conflicts": sorted(set(conflicts)),
    }


def source_aliases(record, locations, resolve_path):
    aliases = set()
    for ref in [record["task"].get("source_ref"), record.get("original_source_ref"), *locations]:
        if isinstance(ref, str) and ref:
            aliases.add(ref)
            try:
                aliases.add(str(Path(resolve_path(ref)).resolve()))
            except (OSError, ValueError):
                pass
    return aliases
