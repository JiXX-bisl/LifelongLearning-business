#!/usr/bin/env python3
"""Package frozen reproduction inputs using only file reads and Git object reads.

This command never imports project modules, opens a database, calls a model, or
runs independent tests. The checked-in recipe is an exact, hash-pinned allowlist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECIPE = "configs/validation/reproduction_bundle_v1.json"
ZIP_TIME = (2026, 10, 20, 0, 0, 0)


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or ":" in value
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or path.as_posix() != value
    ):
        raise ValueError(f"Unsafe relative path: {value!r}")
    return value


def local_path(value: str) -> Path:
    value = relative_path(value)
    path = ROOT
    for part in PurePosixPath(value).parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"Symbolic links are not supported: {value}")
    return path


def read_local(value: str) -> bytes:
    path = local_path(value)
    if not path.is_file():
        raise ValueError(f"Missing regular file: {value}")
    if value.endswith(("-wal", "-shm", "-journal")):
        raise ValueError(f"SQLite sidecar is not a release input: {value}")
    if value.endswith(".sqlite3"):
        for suffix in ("-wal", "-shm", "-journal"):
            if Path(str(path) + suffix).exists():
                raise ValueError(f"SQLite sidecar must be resolved by the user: {value}{suffix}")
    return path.read_bytes()


def read_git(commit: str, source: str) -> bytes:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Git provenance must use a full commit SHA")
    relative_path(source)
    mode = subprocess.run(
        ["git", "ls-tree", commit, "--", source],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.decode("utf-8")
    if not mode.startswith(("100644 blob ", "100755 blob ")):
        raise ValueError(f"Git provenance is not a regular tracked file: {source}")
    return subprocess.run(
        ["git", "show", f"{commit}:{source}"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def verify_exact_directories(recipe: dict) -> None:
    local_sources = {
        item["source"]
        for item in recipe["files"]
        if item.get("source_type", "local") == "local"
    }
    for directory in recipe["exact_directories"]:
        root = local_path(directory)
        if not root.is_dir():
            raise ValueError(f"Missing input directory: {directory}")
        actual = set()
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"Symbolic link inside input directory: {path.relative_to(ROOT)}")
            if path.is_file():
                actual.add(path.relative_to(ROOT).as_posix())
            elif not path.is_dir():
                raise ValueError(f"Unsupported input type: {path.relative_to(ROOT)}")
        expected = {path for path in local_sources if path.startswith(directory + "/")}
        if actual != expected:
            raise ValueError(
                f"Input allowlist mismatch in {directory}: "
                f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
            )


def verify_bindings(recipe: dict, payload: dict[str, bytes]) -> None:
    """Check recorded relationships without interpreting or executing SQL."""
    def document(path: str) -> dict:
        return json.loads(payload[path])

    def check_hash(path: str, expected: str) -> None:
        if path not in payload or digest(payload[path]) != expected:
            raise ValueError(f"Recorded source hash mismatch: {path}")

    public = recipe["bindings"]["public_manifest"]
    private = recipe["bindings"]["private_manifest"]
    for path in (public, private):
        parent = PurePosixPath(path).parent
        for name, expected in document(path)["artifacts"].items():
            item_path = (parent / relative_path(name)).as_posix()
            check_hash(item_path, expected["sha256"])
            if len(payload[item_path]) != expected["bytes"]:
                raise ValueError(f"Recorded source byte count mismatch: {item_path}")
    check_hash(private, document(public)["private_manifest_sha256"])
    config = (PurePosixPath(private).parent / "generation_config.json").as_posix()
    check_hash(config, document(public)["build_config_sha256"])

    bundle_path = recipe["bindings"]["authored_bundle"]
    bundle = document(bundle_path)
    formation = document(recipe["bindings"]["formation_summary"])
    split = document(recipe["bindings"]["split"])
    seeds = document(recipe["bindings"]["seeds"])
    for doc in (bundle, split, seeds):
        for source in doc["sources"].values():
            original_path = source["path"]
            destination = recipe["source_path_map"].get(original_path, original_path)
            check_hash(destination, source["sha256"])
    for name, source in bundle["sources"].items():
        if formation["source_sha256"][name] != source["sha256"]:
            raise ValueError(f"Formation source binding mismatch: {name}")

    case_ids = [case["case_id"] for case in bundle["cases"]]
    if len(case_ids) != 60 or len(set(case_ids)) != 60:
        raise ValueError("Expected exactly 60 unique authored formation cases")
    if set(case_ids) != set(split["formation_case_ids"]) or len(split["test_case_ids"]) != 72:
        raise ValueError("Formation/test partition differs from the frozen 60/72 split")
    if set(case_ids).intersection(split["test_case_ids"]):
        raise ValueError("Formation and test partitions overlap")
    for case in bundle["cases"]:
        sql_path = (PurePosixPath(bundle_path).parent / relative_path(case["sql_file"])).as_posix()
        check_hash(sql_path, case["sql_sha256"])
        result_path = (
            PurePosixPath(recipe["bindings"]["formation_summary"]).parent
            / "results"
            / f'{case["case_id"]}.json'
        ).as_posix()
        result = document(result_path)
        if result["case_id"] != case["case_id"]:
            raise ValueError(f"Formation result case ID mismatch: {result_path}")
    if len(seeds["entries"]) != 18:
        raise ValueError("Expected exactly 18 common seeds in the selection manifest")
    for entry in seeds["entries"]:
        check_hash(entry["source"]["path"], entry["source"]["sha256"])

    initial = document(recipe["bindings"]["initial_summary"])
    reviewed = document(recipe["bindings"]["reviewed_summary"])
    original_root = recipe["original_project_root"].rstrip("/") + "/"
    for summary in (initial, reviewed):
        if summary["member_count"] != 18 or not summary["ready"]:
            raise ValueError("Seed summary does not declare 18 ready common seeds")
        for source, expected in summary["source_sha256"].items():
            if not source.startswith(original_root):
                raise ValueError(f"Unexpected original source root: {source}")
            relative = source[len(original_root):]
            check_hash(recipe["source_path_map"].get(relative, relative), expected)
    review = reviewed["manual_review"]
    for key, expected_key in (("summary", "summary_sha256"), ("seed", "seed_sha256")):
        source = review["source_run"][key]
        if not source.startswith(original_root):
            raise ValueError("Unexpected manual review source root")
        check_hash(source[len(original_root):], review["source_run"][expected_key])
    review_path = review["review_file"]["path"][len(original_root):]
    check_hash(recipe["source_path_map"][review_path], review["review_file"]["sha256"])
    application_path = recipe["source_path_map"]["src/psu_memory/validation/seed_pool.py"]
    check_hash(application_path, review["application_sha256"])


def create_bundle(recipe_path: Path, output_dir: Path) -> dict:
    recipe_bytes = recipe_path.read_bytes()
    recipe = json.loads(recipe_bytes)
    bundle_id = recipe["bundle_id"]
    if recipe["schema_version"] != "1.0" or not re.fullmatch(r"[a-z0-9][a-z0-9.-]+", bundle_id):
        raise ValueError("Unsupported recipe schema or unsafe bundle ID")
    archive_path = output_dir / f"{bundle_id}.zip"
    manifest_path = output_dir / f"{bundle_id}.manifest.json"
    checksum_path = output_dir / f"{bundle_id}.zip.sha256"
    for path in (archive_path, manifest_path, checksum_path):
        if path.exists() or path.is_symlink():
            raise ValueError(f"Refusing to overwrite an existing release artifact: {path}")
    if output_dir.is_symlink():
        raise ValueError("Release output directory must not be a symbolic link")
    verify_exact_directories(recipe)
    payload = {}
    entries = []
    for item in sorted(recipe["files"], key=lambda row: row["path"]):
        path = relative_path(item["path"])
        if not path.startswith(("data/", "provenance/")) or path in payload:
            raise ValueError(f"Duplicate or unsupported payload path: {path}")
        kind = item.get("source_type", "local")
        if kind == "local":
            content = read_local(item["source"])
        elif kind == "git":
            content = read_git(item["commit"], item["source"])
        else:
            raise ValueError(f"Unknown input source type: {kind}")
        if len(content) != item["bytes"] or digest(content) != item["sha256"]:
            raise ValueError(f"Frozen allowlist hash or length mismatch: {item['source']}")
        payload[path] = content
        entry = {key: item[key] for key in ("path", "bytes", "sha256", "category")}
        entry["source"] = {"type": kind, "path": item["source"]}
        if kind == "git":
            entry["source"]["commit"] = item["commit"]
        entries.append(entry)
    verify_bindings(recipe, payload)
    manifest = {
        key: recipe[key]
        for key in (
            "schema_version", "bundle_id", "source_commit", "data_version",
            "question_bank", "comparison_version", "counts", "embedding",
            "purpose", "known_limitations",
        )
    }
    manifest["recipe_sha256"] = digest(recipe_bytes)
    manifest["source_path_map"] = recipe["source_path_map"]
    manifest["verification_scope"] = (
        "File hashes, byte lengths, original manifest bindings and JSON metadata only. "
        "No database, embedding, model request or independent test was executed."
    )
    manifest["file_count"] = len(entries)
    manifest["payload_bytes"] = sum(entry["bytes"] for entry in entries)
    manifest["files"] = entries
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    members = {**payload, "bundle-manifest.json": manifest_bytes}
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, mode="x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, content in sorted(members.items()):
            info = zipfile.ZipInfo(name, date_time=ZIP_TIME)
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    with manifest_path.open("xb") as output:
        output.write(manifest_bytes)
    archive_sha256 = digest(archive_path.read_bytes())
    with checksum_path.open("x", encoding="utf-8", newline="\n") as output:
        output.write(f"{archive_sha256}  {archive_path.name}\n")
    return {
        "archive": str(archive_path),
        "manifest": str(manifest_path),
        "checksum": str(checksum_path),
        "sha256": archive_sha256,
        "archive_bytes": archive_path.stat().st_size,
        "payload_files": len(entries),
        "payload_bytes": manifest["payload_bytes"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=ROOT / DEFAULT_RECIPE)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist")
    args = parser.parse_args()
    try:
        result = create_bundle(args.recipe, args.output_dir)
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        print(f"Packaging refused: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
