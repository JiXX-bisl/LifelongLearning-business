"""Resolve project data paths while preserving historical provenance strings."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=8)
def _read_map(path: Path, mtime_ns: int, size: int) -> dict:
    # File metadata participates in the cache key so map updates are visible.
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_project_path(value: str | Path, *, project_root: Path | None = None) -> Path:
    """Map known old file/directory paths to their current project locations.

    Exact file aliases precede directory aliases, including consolidated copies
    and early unscoped logs. Unknown external absolute paths are unchanged.
    This function does not modify a loaded record or require its target to exist,
    so new output paths use the same directory rules as historical reads.
    """
    root = (project_root or PROJECT_ROOT).resolve()
    path = Path(value).expanduser()
    absolute = (path if path.is_absolute() else root / path).resolve()
    map_file = root / "data/catalog/path_map.json"
    try:
        stat = map_file.stat()
    except FileNotFoundError:
        return absolute
    mapping = _read_map(map_file, stat.st_mtime_ns, stat.st_size)
    relative = None
    for base in [root, Path(mapping["original_project_root"])]:
        try:
            relative = absolute.relative_to(base).as_posix()
            break
        except ValueError:
            continue
    if relative is None:
        return absolute
    target = mapping["files"].get(relative)
    if target is None:
        for old, new in sorted(mapping["directories"].items(), key=lambda item: (-len(item[0]), item[0])):
            if relative == old or relative.startswith(old + "/"):
                target = new + relative[len(old):]
                break
    resolved = (root / (target or relative)).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("A catalog path mapping points outside the project")
    return resolved
