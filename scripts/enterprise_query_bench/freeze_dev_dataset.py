from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATASET_DIR = (
    PROJECT_ROOT / 'data/archive/questions/enterprise_query_bench/enterprise_query_bench_dev_v0.1.0'
)

DEFAULT_BASELINE_DIR = (
    PROJECT_ROOT / 'data/sql/synthetic_enterprise_v21_40_v1.0.0'
)

REFERENCE_ENTRY_FILE = "entry_dict.json"
REFERENCE_ORACLE_FILE = "oracle_entry_dict.json"
ACTIVE_ENTRY_FILE = "entry_dict_colloquial_instructions.json"
ACTIVE_ORACLE_FILE = "oracle_entry_dict_colloquial_instructions.json"

TASK_FILE_NAMES = [
    REFERENCE_ENTRY_FILE,
    REFERENCE_ORACLE_FILE,
    ACTIVE_ENTRY_FILE,
    ACTIVE_ORACLE_FILE,
]


class FreezeValidationError(ValueError):
    """任务集不能被冻结。"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and freeze the Enterprise Query Bench "
            "development dataset."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=DEFAULT_BASELINE_DIR,
    )
    parser.add_argument(
        "--dataset-id",
        default="enterprise_query_bench_dev_v0.1.0",
    )
    parser.add_argument(
        "--version",
        default="0.1.0",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify an existing frozen dataset without rewriting files.",
    )
    return parser.parse_args()


def reject_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}

    for key, value in pairs:
        if key in result:
            raise FreezeValidationError(
                f"Duplicate JSON key: {key!r}"
            )
        result[key] = value

    return result


def load_json_strict(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(
            file,
            object_pairs_hook=reject_duplicate_keys,
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    temporary_path = path.with_name(
        f".{path.name}.tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            value,
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")

    temporary_path.replace(path)


def write_text_atomic(path: Path, value: str) -> None:
    temporary_path = path.with_name(
        f".{path.name}.tmp"
    )
    temporary_path.write_text(
        value,
        encoding="utf-8",
    )
    temporary_path.replace(path)


def validate_sample_indices(
    dataset: dict[str, Any],
    expected_count: int = 40,
) -> list[str]:
    try:
        sample_indices = sorted(
            dataset.keys(),
            key=int,
        )
    except (TypeError, ValueError) as error:
        raise FreezeValidationError(
            "Top-level sample indices must be numeric strings."
        ) from error

    expected_indices = [
        str(index)
        for index in range(expected_count)
    ]

    if sample_indices != expected_indices:
        raise FreezeValidationError(
            "Sample indices must be 0 through 39."
        )

    return sample_indices


def validate_task_ids(
    entry_dict: dict[str, Any],
    sample_indices: list[str],
) -> None:
    task_ids: list[str] = []

    for sample_index in sample_indices:
        task_id = entry_dict[sample_index].get(
            "task_id"
        )

        if not isinstance(task_id, str):
            raise FreezeValidationError(
                f"Task {sample_index} has no valid task_id."
            )

        if not re.fullmatch(
            r"ENT-D-\d{6}",
            task_id,
        ):
            raise FreezeValidationError(
                f"Invalid development task_id: {task_id}"
            )

        expected_task_id = (
            f"ENT-D-{int(sample_index) + 1:06d}"
        )

        if task_id != expected_task_id:
            raise FreezeValidationError(
                f"Task {sample_index} has task_id "
                f"{task_id}, expected {expected_task_id}."
            )

        task_ids.append(task_id)

    if len(task_ids) != len(set(task_ids)):
        raise FreezeValidationError(
            "Duplicate task_id detected."
        )


def validate_entry_oracle_pair(
    entry_dict: dict[str, Any],
    oracle_dict: dict[str, Any],
    sample_indices: list[str],
    pair_name: str,
) -> None:
    if set(entry_dict) != set(oracle_dict):
        raise FreezeValidationError(
            f"{pair_name}: Entry and Oracle indices differ."
        )

    for sample_index in sample_indices:
        entry = entry_dict[sample_index]
        oracle = oracle_dict[sample_index]
        answer_info = entry["answer_info"]

        checks = {
            "task_id": (
                entry["task_id"],
                oracle["task_id"],
            ),
            "instruction": (
                entry["instruction"],
                oracle["instruction"],
            ),
            "sql": (
                answer_info["sql"],
                oracle["sql"],
            ),
            "answer_direct": (
                answer_info["direct"],
                oracle["answer_direct"],
            ),
            "comparison_policy": (
                answer_info["comparison_policy"],
                oracle["comparison_policy"],
            ),
        }

        for field_name, (
            entry_value,
            oracle_value,
        ) in checks.items():
            if entry_value != oracle_value:
                raise FreezeValidationError(
                    f"{pair_name}: task {sample_index} "
                    f"differs in {field_name}."
                )


def without_instruction(
    entry: dict[str, Any],
) -> dict[str, Any]:
    return {
        key: value
        for key, value in entry.items()
        if key != "instruction"
    }


def validate_colloquial_variant(
    reference_dict: dict[str, Any],
    colloquial_dict: dict[str, Any],
    sample_indices: list[str],
    file_type: str,
) -> None:
    if set(reference_dict) != set(colloquial_dict):
        raise FreezeValidationError(
            f"{file_type}: reference and colloquial indices differ."
        )

    instruction_list: list[str] = []

    for sample_index in sample_indices:
        reference_entry = reference_dict[
            sample_index
        ]
        colloquial_entry = colloquial_dict[
            sample_index
        ]

        if (
            without_instruction(reference_entry)
            != without_instruction(colloquial_entry)
        ):
            raise FreezeValidationError(
                f"{file_type}: task {sample_index} changed "
                "fields other than instruction."
            )

        reference_instruction = (
            reference_entry["instruction"].strip()
        )
        colloquial_instruction = (
            colloquial_entry["instruction"].strip()
        )

        if not colloquial_instruction:
            raise FreezeValidationError(
                f"{file_type}: task {sample_index} "
                "has an empty instruction."
            )

        if reference_instruction == colloquial_instruction:
            raise FreezeValidationError(
                f"{file_type}: task {sample_index} "
                "was not colloquialized."
            )

        instruction_list.append(
            colloquial_instruction
        )

    if len(instruction_list) != len(
        set(instruction_list)
    ):
        raise FreezeValidationError(
            f"{file_type}: duplicate colloquial "
            "instructions detected."
        )


def validate_review_status(
    entry_dict: dict[str, Any],
    sample_indices: list[str],
) -> None:
    invalid_tasks: list[str] = []

    for sample_index in sample_indices:
        review_status = (
            entry_dict[sample_index]
            .get("metadata", {})
            .get("review_status")
        )

        if review_status != "reviewed":
            invalid_tasks.append(sample_index)

    if invalid_tasks:
        raise FreezeValidationError(
            "All active tasks must be reviewed before "
            f"freezing. Invalid samples: {invalid_tasks}"
        )


def collect_distributions(
    entry_dict: dict[str, Any],
    sample_indices: list[str],
) -> dict[str, dict[str, int]]:
    fields = {
        "difficulty": lambda entry: (
            entry["metadata"]["difficulty"]
        ),
        "business_query_type": lambda entry: (
            entry["metadata"]["business_query_type"]
        ),
        "task_family_id": lambda entry: (
            entry["metadata"]["task_family_id"]
        ),
        "result_type": lambda entry: (
            entry["metadata"]["result_type"]
        ),
        "row_order": lambda entry: (
            entry["answer_info"]
            ["comparison_policy"]["row_order"]
        ),
    }

    distributions: dict[str, dict[str, int]] = {}

    for field_name, getter in fields.items():
        counter = Counter(
            getter(entry_dict[sample_index])
            for sample_index in sample_indices
        )
        distributions[field_name] = dict(
            sorted(counter.items())
        )

    return distributions


def validate_and_collect(
    dataset_dir: Path,
    baseline_dir: Path,
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, int]],
]:
    for file_name in TASK_FILE_NAMES:
        file_path = dataset_dir / file_name

        if not file_path.is_file():
            raise FileNotFoundError(
                f"Missing task file: {file_path}"
            )

    baseline_manifest_path = (
        baseline_dir / "manifest.json"
    )

    if not baseline_manifest_path.is_file():
        raise FileNotFoundError(
            "Database baseline manifest is missing: "
            f"{baseline_manifest_path}"
        )

    reference_entry = load_json_strict(
        dataset_dir / REFERENCE_ENTRY_FILE
    )
    reference_oracle = load_json_strict(
        dataset_dir / REFERENCE_ORACLE_FILE
    )
    active_entry = load_json_strict(
        dataset_dir / ACTIVE_ENTRY_FILE
    )
    active_oracle = load_json_strict(
        dataset_dir / ACTIVE_ORACLE_FILE
    )
    baseline_manifest = load_json_strict(
        baseline_manifest_path
    )

    for name, value in {
        REFERENCE_ENTRY_FILE: reference_entry,
        REFERENCE_ORACLE_FILE: reference_oracle,
        ACTIVE_ENTRY_FILE: active_entry,
        ACTIVE_ORACLE_FILE: active_oracle,
    }.items():
        if not isinstance(value, dict):
            raise FreezeValidationError(
                f"{name} root must be a JSON object."
            )

    sample_indices = validate_sample_indices(
        active_entry
    )

    validate_sample_indices(reference_entry)
    validate_sample_indices(reference_oracle)
    validate_sample_indices(active_oracle)

    validate_task_ids(
        active_entry,
        sample_indices,
    )

    validate_entry_oracle_pair(
        reference_entry,
        reference_oracle,
        sample_indices,
        "reference",
    )
    validate_entry_oracle_pair(
        active_entry,
        active_oracle,
        sample_indices,
        "colloquial",
    )

    validate_colloquial_variant(
        reference_entry,
        active_entry,
        sample_indices,
        "Entry",
    )
    validate_colloquial_variant(
        reference_oracle,
        active_oracle,
        sample_indices,
        "Oracle",
    )

    validate_review_status(
        active_entry,
        sample_indices,
    )

    schema_signatures = {
        (
            entry["schema_info"]["baseline_id"],
            str(entry["schema_info"]["schema_version"]),
            entry["schema_info"]["schema_ref"],
        )
        for entry in active_entry.values()
    }

    if len(schema_signatures) != 1:
        raise FreezeValidationError(
            "Active tasks reference different Schemas."
        )

    baseline_id, schema_version, schema_ref = next(
        iter(schema_signatures)
    )

    if baseline_manifest.get("status") != "FROZEN":
        raise FreezeValidationError(
            "The database baseline is not frozen."
        )

    if (
        baseline_id
        != baseline_manifest.get("baseline_id")
    ):
        raise FreezeValidationError(
            "Task baseline_id differs from the database manifest."
        )

    if (
        schema_version
        != str(
            baseline_manifest.get("schema_version")
        )
    ):
        raise FreezeValidationError(
            "Task schema_version differs from the "
            "database manifest."
        )

    database_reference = {
        "baseline_id": baseline_id,
        "schema_version": schema_version,
        "schema_ref": schema_ref,
        "database_name": (
            baseline_manifest["mysql"]["database"]
        ),
        "manifest_sha256": sha256_file(
            baseline_manifest_path
        ),
    }

    distributions = collect_distributions(
        active_entry,
        sample_indices,
    )

    return database_reference, distributions


def build_artifact_records(
    dataset_dir: Path,
) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}

    for file_name in TASK_FILE_NAMES:
        path = dataset_dir / file_name
        artifacts[file_name] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }

    return artifacts


def write_checksums(
    dataset_dir: Path,
) -> None:
    checksum_names = [
        *TASK_FILE_NAMES,
        "manifest.json",
    ]

    lines = [
        f"{sha256_file(dataset_dir / file_name)}  "
        f"{file_name}"
        for file_name in sorted(checksum_names)
    ]

    write_text_atomic(
        dataset_dir / "checksums.sha256",
        "\n".join(lines) + "\n",
    )


def verify_checksums(
    dataset_dir: Path,
) -> None:
    checksum_path = (
        dataset_dir / "checksums.sha256"
    )

    if not checksum_path.is_file():
        raise FileNotFoundError(
            f"Checksum file is missing: {checksum_path}"
        )

    checked_count = 0

    for line in checksum_path.read_text(
        encoding="utf-8"
    ).splitlines():
        if not line.strip():
            continue

        try:
            expected_hash, file_name = (
                line.split(maxsplit=1)
            )
        except ValueError as error:
            raise FreezeValidationError(
                f"Invalid checksum line: {line!r}"
            ) from error

        file_name = file_name.strip()
        target_path = dataset_dir / file_name

        if not target_path.is_file():
            raise FreezeValidationError(
                f"Checksum target is missing: {file_name}"
            )

        actual_hash = sha256_file(target_path)

        if actual_hash != expected_hash:
            raise FreezeValidationError(
                f"Checksum mismatch: {file_name}"
            )

        checked_count += 1

    if checked_count != 5:
        raise FreezeValidationError(
            f"Expected 5 checksum records, "
            f"found {checked_count}."
        )


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    baseline_dir = args.baseline_dir.resolve()

    database_reference, distributions = (
        validate_and_collect(
            dataset_dir=dataset_dir,
            baseline_dir=baseline_dir,
        )
    )

    manifest_path = (
        dataset_dir / "manifest.json"
    )
    checksum_path = (
        dataset_dir / "checksums.sha256"
    )

    if args.verify:
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Manifest is missing: {manifest_path}"
            )

        manifest = load_json_strict(
            manifest_path
        )

        if manifest.get("status") != "FROZEN":
            raise FreezeValidationError(
                "Task dataset manifest is not frozen."
            )

        if (
            manifest.get("dataset_id")
            != args.dataset_id
        ):
            raise FreezeValidationError(
                "Dataset ID differs from the expected value."
            )

        verify_checksums(dataset_dir)

        print("Enterprise query dataset verification passed.")
        print(
            f"dataset_id: {manifest['dataset_id']}"
        )
        print(
            f"task_count: {manifest['task_count']}"
        )
        return

    if manifest_path.exists() or checksum_path.exists():
        raise FileExistsError(
            "Dataset freeze artifacts already exist. "
            "Use --verify to validate them. "
            "Do not overwrite a frozen version."
        )

    artifacts = build_artifact_records(
        dataset_dir
    )

    manifest = {
        "dataset_id": args.dataset_id,
        "version": args.version,
        "split": "dev",
        "status": "FROZEN",
        "task_type": "enterprise_query_bench",
        "language": "zh-CN",
        "task_count": 40,
        "active_agent_entry": ACTIVE_ENTRY_FILE,
        "active_oracle_entry": ACTIVE_ORACLE_FILE,
        "design_reference_entry": REFERENCE_ENTRY_FILE,
        "design_reference_oracle": REFERENCE_ORACLE_FILE,
        "active_instruction_style": "colloquial",
        "review_status": "reviewed",
        "sample_index_range": [
            "0",
            "39",
        ],
        "task_id_range": [
            "ENT-D-000001",
            "ENT-D-000040",
        ],
        "database_baseline": database_reference,
        "evaluation": {
            "policy": "B1",
            "evaluation_target": (
                "last_successful_read_only_query_result"
            ),
            "natural_language_answer_scored": False,
            "maximum_result_rows": 50,
        },
        "distributions": distributions,
        "frozen_at": datetime.now(
            ZoneInfo("Asia/Shanghai")
        ).isoformat(timespec="seconds"),
        "hash_algorithm": "SHA-256",
        "artifacts": artifacts,
        "immutability_note": (
            "Do not overwrite this frozen directory. "
            "Any task, instruction, SQL, Ground Truth or "
            "comparison-policy change requires a new version."
        ),
    }

    write_json_atomic(
        manifest_path,
        manifest,
    )
    write_checksums(dataset_dir)

    print("Enterprise query dataset frozen successfully.")
    print(f"dataset_id: {args.dataset_id}")
    print("task_count: 40")
    print(f"manifest: {manifest_path}")
    print(f"checksums: {checksum_path}")


if __name__ == "__main__":
    main()