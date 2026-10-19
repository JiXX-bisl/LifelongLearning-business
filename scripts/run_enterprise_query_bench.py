from __future__ import annotations

import argparse
import json
import subprocess   
import sys
from pathlib import Path
from typing import Any
import hashlib


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.lifelong_learning.agents.language_model_agent import LanguageModelAgent
from src.project_paths import resolve_project_path as resolve_catalog_path
from src.lifelong_learning.configuration.yaml_loader import load_yaml_config
from src.lifelong_learning.environments.fixed_mysql_environment import FixedMySQLReadOnlyEnvironment
from src.lifelong_learning.experiments.enterprise_query_metrics import calculate_enterprise_query_metric
from src.lifelong_learning.experiments.runner import DBBenchExperimentRunner, save_json
from src.lifelong_learning.language_models.openai_compatible import OpenAICompatibleLanguageModel
from src.lifelong_learning.memory.group_self_consistency import GroupSelfConsistencyMemory
from src.lifelong_learning.memory.previous_sample_utilization import PreviousSampleUtilizationMemory
from src.lifelong_learning.tasks.enterprise_query_bench.task import EnterpriseQueryTask
from src.lifelong_learning.prompts.enterprise_query_prompt_builder import (
    GUIDED_INTERFACE_PROFILE,
    get_enterprise_query_protocol_version,
)


SUPPORTED_METHODS = {
    "standard", 
    "previous_sample_utilization", 
    "group_self_consistency"
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Standard, PSU or GSC on "
            "Enterprise Query Bench."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to an Enterprise Query Bench YAML config.",
    )
    return parser.parse_args()


def resolve_project_path(
    path_value: str | Path,
) -> Path:
    return resolve_catalog_path(path_value, project_root=PROJECT_ROOT)


def load_json(path: Path) -> Any:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)

def verify_frozen_task_dataset(
    data_file_path: Path,
    baseline_dir: Path,
) -> dict[str, Any]:
    """
    每次正式实验前验证任务集Manifest和SHA-256。

    同时确认配置使用的是Manifest指定的口语化Entry。
    """
    dataset_dir = data_file_path.parent
    manifest_path = dataset_dir / "manifest.json"
    freeze_script_path = (
        PROJECT_ROOT
        / "scripts"
        / "enterprise_query_bench"
        / "freeze_dev_dataset.py"
    )

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Task dataset manifest is missing: {manifest_path}"
        )

    if not freeze_script_path.is_file():
        raise FileNotFoundError(
            f"Freeze verification script is missing: "
            f"{freeze_script_path}"
        )

    manifest = load_json(manifest_path)

    if manifest.get("status") != "FROZEN":
        raise ValueError(
            "Enterprise query task dataset is not frozen."
        )

    dataset_id = manifest.get("dataset_id")

    if not isinstance(dataset_id, str):
        raise ValueError(
            "Task dataset manifest has no valid dataset_id."
        )

    subprocess.run(
        [
            sys.executable,
            str(freeze_script_path),
            "--dataset-dir",
            str(dataset_dir),
            "--baseline-dir",
            str(baseline_dir),
            "--dataset-id",
            dataset_id,
            "--verify",
        ],
        check=True,
    )

    active_entry_name = manifest.get(
        "active_agent_entry"
    )
    expected_entry_path = (
        dataset_dir / str(active_entry_name)
    ).resolve()

    if data_file_path != expected_entry_path:
        raise ValueError(
            "Configured data_file_path is not the frozen "
            "active colloquial Agent Entry file."
        )

    return manifest


def validate_sample_order(
    sample_order: Any,
    available_sample_indices: set[str],
) -> list[str]:
    if not isinstance(sample_order, list):
        raise TypeError(
            "experiment.sample_order must be a list."
        )

    normalized_order = [
        str(sample_index)
        for sample_index in sample_order
    ]

    if not normalized_order:
        raise ValueError(
            "experiment.sample_order must not be empty."
        )

    if len(normalized_order) != len(
        set(normalized_order)
    ):
        raise ValueError(
            "experiment.sample_order contains duplicates."
        )

    unknown_indices = (
        set(normalized_order)
        - available_sample_indices
    )

    if unknown_indices:
        raise ValueError(
            "experiment.sample_order contains unknown "
            f"indices: {sorted(unknown_indices)}"
        )

    return normalized_order

def ensure_clean_output_directory(
    output_dir: Path,
) -> None:
    if output_dir.exists() and any(
        output_dir.iterdir()
    ):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Use a new output directory for each experiment."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


def build_memory(method: str, experiment_config: dict[str, Any]) -> tuple[
    PreviousSampleUtilizationMemory | None, GroupSelfConsistencyMemory: None
]:
    psu_memory = None
    gsc_memory = None

    if method == "previous_sample_utilization":
        utilized_sample_count = int(
            experiment_config["utilized_sample_count"]
        )
    
        psu_memory = PreviousSampleUtilizationMemory(
            utilized_sample_count=utilized_sample_count
        )
    elif method == "group_self_consistency":
        group_count = int(
            experiment_config["group_count"]
        )
        sample_count_per_group = int(
            experiment_config[
                "sample_count_per_group"
            ]
        )
        gsc_memory = GroupSelfConsistencyMemory(
            group_count=group_count,
            sample_count_per_group=sample_count_per_group
        )

    return psu_memory, gsc_memory


def main() -> None:
    args = parse_args()
    config_path = resolve_project_path(args.config)
    config = load_yaml_config(config_path)

    task_config = config["task"]
    language_model_config = config["language_model"]
    agent_config = config["agent"]
    experiment_config = config["experiment"]

    method = experiment_config["method"]

    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported experiment method: {method}"
        )

    if (language_model_config["type"] != "openai_compatible"):
        raise ValueError(
            "Only OpenAI compatible language models are supported."
        )
    if agent_config["type"] != "language_model":
        raise ValueError(
            "Only language_model agents are supported."
        )

    data_file_path = resolve_project_path(
        task_config["data_file_path"]
    )
    baseline_dir = resolve_project_path(
        task_config["baseline_dir"]
    )
    prompt_file_path = resolve_project_path(
        task_config["prompt_file_path"]
    )
    output_dir = resolve_project_path(
        experiment_config["output_dir"]
    )
    interface_profile = str(
        task_config.get(
            "interface_profile",
            GUIDED_INTERFACE_PROFILE,
        )
    )

    expected_protocol_version = (
        get_enterprise_query_protocol_version(
            interface_profile
        )
    )

    prompt_artifact = load_json(
        prompt_file_path
    )
    prompt_metadata = prompt_artifact.get(
        "metadata"
    )

    if not isinstance(prompt_metadata, dict):
        raise ValueError(
            "Prompt artifact has no metadata. "
            "Regenerate prompts before running."
        )

    if (
        prompt_metadata.get("interface_profile")
        != interface_profile
    ):
        raise ValueError(
            "Configured interface_profile differs "
            "from prompt artifact metadata."
        )

    if (
        prompt_metadata.get(
            "interaction_protocol_version"
        )
        != expected_protocol_version
    ):
        raise ValueError(
            "Prompt protocol version mismatch."
        )
    
    task_manifest = verify_frozen_task_dataset(data_file_path=data_file_path, baseline_dir=baseline_dir)
    baseline_manifest_path = baseline_dir / "manifest.json"
    baseline_manifest = load_json(baseline_manifest_path)

    manifest_baseline_id = task_manifest["database_baseline"]["baseline_id"]

    if manifest_baseline_id != baseline_manifest["baseline_id"]:
        raise ValueError(
            "Task dataset and database baseline differ."
        )
    mysql_config = task_config["mysql"]

    if mysql_config.get("password") is not None:
        raise ValueError(
            "Do not store the MySQL password in YAML. "
            "Use the configured password environment variable."
        )

    ensure_clean_output_directory(output_dir)

    language_model = (
        OpenAICompatibleLanguageModel(
            model_name=language_model_config.get(
                "model_name"
            ),
            temperature=language_model_config.get(
                "temperature"
            ),
            max_tokens=language_model_config.get(
                "max_tokens"
            ),
            api_key=language_model_config.get(
                "api_key"
            ),
            base_url=language_model_config.get(
                "base_url"
            ),
            request_interval_seconds=(
                language_model_config.get(
                    "request_interval_seconds"
                )
            ),
        )
    )

    language_model_config["model_name"] = (
        language_model.model_name
    )

    agent = LanguageModelAgent(
        language_model=language_model,
        system_prompt=agent_config[
            "system_prompt"
        ],
    )

    psu_memory, gsc_memory = build_memory(
        method=method,
        experiment_config=experiment_config,
    )

    database_name = baseline_manifest["mysql"]["database"]

    environment: FixedMySQLReadOnlyEnvironment | None = None

    try:
        environment = FixedMySQLReadOnlyEnvironment(
            database=database_name, 
            host=mysql_config.get("host", "127.0.0.1"),
            port = int(mysql_config.get("port", 3306)),
            user=mysql_config.get("user", "synthetic"),
            password=None,
            password_env=mysql_config.get("password_env", "ENTERPRISE_MYSQL_PASSWORD"),
            connection_timeout_seconds=int(mysql_config.get("connection_timeout_seconds", 10)),
            statement_timeout_ms=int(mysql_config.get("statement_timeout_ms", 10_000)),
            max_result_rows=int(mysql_config.get("max_result_rows", 50))
        )
        task = EnterpriseQueryTask(
            data_file_path=data_file_path,
            baseline_dir=baseline_dir,
            prompt_file_path=prompt_file_path,
            environment=environment,
            interface_profile=interface_profile,
            max_round=int(
                task_config.get("max_round", 3)
            ),
            max_schema_requests=int(
                task_config.get(
                    "max_schema_requests",
                    6,
                )
            ),
            verify_live_baseline=True,
        )

        sample_order = validate_sample_order(
            experiment_config["sample_order"],
            available_sample_indices=set(task.get_sample_index_list())
        )

        resolved_config = {
            **config,
            "runtime": {
                "task_dataset_id": (
                    task_manifest["dataset_id"]
                ),
                "database_baseline_id": (
                    baseline_manifest[
                        "baseline_id"
                    ]
                ),
                "interface_profile": interface_profile,
                "schema_prompt_version": (
                    task.dataset_bundle.schema_prompt_version
                ),
                "max_schema_requests": task.max_schema_requests,
                "business_round_budget": task.max_round,
                "schema_catalog_sha256": hashlib.sha256(
                    json.dumps(
                        task.dataset_bundle.table_schema_map,
                        ensure_ascii=False,
                        sort_keys=True,
                    ).encode("utf-8")
                ).hexdigest(),
                "initial_prompt_sha256": hashlib.sha256(
                    prompt_file_path.read_bytes()
                ).hexdigest(),
                "interaction_protocol_version": (
                    expected_protocol_version
                ),
            },
        }

        save_json(
            output_dir / "config.json",
            resolved_config,
        )

        runner = DBBenchExperimentRunner(
            task=task,
            agent=agent,
            sample_order=sample_order,
            output_dir=output_dir,
            previous_sample_utilization_memory=psu_memory,
            group_self_consistency_memory=gsc_memory,
            metric_calculator=calculate_enterprise_query_metric
        )

        print(
            "Starting Enterprise Query Bench: "
            f"dataset={task_manifest['dataset_id']}, "
            f"model={language_model.model_name}, "
            f"method={method}, "
            f"sample_count={len(sample_order)}"
        )

        runner.run()
    
    except Exception:
        if environment is not None:
            environment.close()
        raise

if __name__ == "__main__":
    main()
