from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


ConfigDict = dict[str, Any]


def deep_merge(
    base_config: ConfigDict,
    override_config: ConfigDict,
) -> ConfigDict:
    merged_config = deepcopy(base_config)

    for key, override_value in override_config.items():
        base_value = merged_config.get(key)

        if (
            isinstance(base_value, dict)
            and isinstance(override_value, dict)
        ):
            merged_config[key] = deep_merge(
                base_config=base_value,
                override_config=override_value,
            )
        else:
            merged_config[key] = deepcopy(
                override_value
            )

    return merged_config


def load_yaml_config(
    config_file_path: str | Path,
) -> ConfigDict:
    config_file_path = Path(
        config_file_path
    ).resolve()
    with config_file_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        raw_config = yaml.safe_load(file) or {}

    if not isinstance(raw_config, dict):
        raise TypeError(
            "The YAML root must be a mapping."
        )

    import_path = raw_config.pop(
        "import",
        None,
    )

    if import_path is None:
        return raw_config

    if not isinstance(import_path, str):
        raise TypeError(
            "The import field must be a string."
        )

    base_config_path = (
        config_file_path.parent
        / import_path
    ).resolve()

    base_config = load_yaml_config(
        base_config_path
    )

    return deep_merge(
        base_config=base_config,
        override_config=raw_config,
    )