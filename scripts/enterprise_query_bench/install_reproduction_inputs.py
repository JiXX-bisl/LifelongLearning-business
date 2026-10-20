#!/usr/bin/env python3
"""Install a checksummed experiment bundle without replacing existing files.

Uses only the Python standard library. Treats SQLite files as opaque bytes;
does not import project modules, connect to databases, or load models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[2]
MAX_TOTAL_BYTES = 1024 * 1024 * 1024
MANIFEST_NAME = "bundle-manifest.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, f"Duplicate manifest key: {key}")
        value[key] = item
    return value


def relative_path(value):
    require(isinstance(value, str) and value, "Empty bundle path")
    path = PurePosixPath(value)
    require(
        not path.is_absolute()
        and value == path.as_posix()
        and "\\" not in value
        and ":" not in value
        and all(part not in ("", ".", "..") for part in value.split("/"))
        and path.parts[0] in ("data", "provenance"),
        f"Invalid bundle path: {value}",
    )
    return path


def target_path(root, relative):
    target = root.joinpath(*relative_path(relative).parts)
    current = root
    for part in target.relative_to(root).parts:
        current = current / part
        require(not current.is_symlink(), f"Refusing symlink: {current}")
    require(target.resolve().is_relative_to(root), f"Path escapes project: {relative}")
    return target


def install(archive, checksum_file, root):
    require(root.is_dir(), "Destination must be an existing project directory")
    root = root.resolve()
    require((root / "src/psu_memory/validation/runner.py").is_file(),
            "Destination must contain the cloned project")
    checksum_lines = checksum_file.read_text(encoding="utf-8").splitlines()
    require(len(checksum_lines) == 1, "Expected one archive checksum")
    checksum_parts = checksum_lines[0].split(maxsplit=1)
    require(len(checksum_parts) == 2
            and re.fullmatch(r"[0-9a-f]{64}", checksum_parts[0]) is not None
            and checksum_parts[1].lstrip("*") == archive.name,
            "Checksum file must name this archive")
    require(sha256_file(archive) == checksum_parts[0], "Archive SHA-256 mismatch")

    with zipfile.ZipFile(archive) as bundle:
        infos = bundle.infolist()
        names = [info.filename for info in infos]
        require(len(names) == len(set(names)), "Duplicate ZIP member")
        require(len(names) == len({name.casefold() for name in names}),
                "ZIP paths collide on a case-insensitive filesystem")
        require(MANIFEST_NAME in names, "Missing bundle-manifest.json")
        require(bundle.getinfo(MANIFEST_NAME).file_size <= 4 * 1024 * 1024,
                "Manifest is too large")
        manifest_bytes = bundle.read(MANIFEST_NAME)
        manifest = json.loads(manifest_bytes, object_pairs_hook=unique_object)
        require(isinstance(manifest, dict) and manifest.get("schema_version") == "1.0",
                "Unsupported bundle manifest")
        bundle_id = manifest.get("bundle_id")
        require(isinstance(bundle_id, str)
                and re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,127}", bundle_id) is not None,
                "Invalid bundle ID")
        entries = manifest.get("files")
        require(isinstance(entries, list) and entries, "Empty bundle manifest")
        expected = {}
        total = 0
        for entry in entries:
            require(isinstance(entry, dict), "Invalid file entry")
            path = entry.get("path")
            relative_path(path)
            require(path not in expected, f"Duplicate manifest path: {path}")
            require(type(entry.get("bytes")) is int and entry["bytes"] >= 0,
                    f"Invalid file size: {path}")
            require(isinstance(entry.get("sha256"), str)
                    and re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is not None,
                    f"Invalid file checksum: {path}")
            total += entry["bytes"]
            expected[path] = entry
        require(total <= MAX_TOTAL_BYTES, "Bundle exceeds the 1 GiB input limit")
        require(set(names) == set(expected) | {MANIFEST_NAME},
                "Archive contents differ from the manifest")
        for info in infos:
            file_type = stat.S_IFMT(info.external_attr >> 16)
            require(not info.is_dir() and file_type in (0, stat.S_IFREG)
                    and not (info.flag_bits & 1), f"Unsupported ZIP entry: {info.filename}")
            if info.filename in expected:
                require(info.file_size == expected[info.filename]["bytes"],
                        f"ZIP size mismatch: {info.filename}")

        record = f"data/catalog/reproduction-bundles/{bundle_id}.json"
        require(record not in expected, "Manifest collides with its installation record")
        all_paths = (*expected, record)
        folded = {name.casefold() for name in all_paths}
        require(len(folded) == len(all_paths), "Installation record has a case-colliding path")
        for name in all_paths:
            require(not any(parent.as_posix().casefold() in folded
                            for parent in PurePosixPath(name).parents),
                    f"Bundle file is also used as a directory: {name}")
        targets = {name: target_path(root, name) for name in (*expected, record)}
        for name, target in targets.items():
            for parent in target.parents:
                if parent == root:
                    break
                require(not parent.exists() or parent.is_dir(),
                        f"Parent is not a directory: {parent}")
            if target.exists():
                require(target.is_file(), f"Destination is not a file: {target}")
                wanted = (hashlib.sha256(manifest_bytes).hexdigest()
                          if name == record else expected[name]["sha256"])
                require(sha256_file(target) == wanted,
                        f"Existing file differs; nothing installed: {name}")

        # Verify every payload before publishing any file into data/provenance.
        with tempfile.TemporaryDirectory(prefix=".reproduction-stage-", dir=root) as staging:
            stage = Path(staging)
            for number, (name, entry) in enumerate(expected.items()):
                digest = hashlib.sha256()
                count = 0
                with bundle.open(name) as source, (stage / str(number)).open("xb") as out:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        count += len(block)
                        require(count <= entry["bytes"], f"Expanded size exceeds manifest: {name}")
                        digest.update(block)
                        out.write(block)
                require(count == entry["bytes"] and digest.hexdigest() == entry["sha256"],
                        f"Payload checksum mismatch: {name}")
            (stage / str(len(expected))).write_bytes(manifest_bytes)
            installed, unchanged = 0, 0
            for number, name in enumerate((*expected, record)):
                target = target_path(root, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                target_path(root, name)
                staged = stage / str(number)
                if target.exists():
                    require(target.is_file() and sha256_file(target) == sha256_file(staged),
                            f"Destination changed during installation: {name}")
                    unchanged += 1
                    continue
                # Same-filesystem hard link is atomic and fails if a file appears.
                os.link(staged, target)
                installed += 1
    return {
        "bundle_id": bundle_id, "destination": str(root),
        "installed_files": installed, "identical_existing_files": unchanged,
        "manifest": record,
        "next_step": "Download BGE-M3, configure the device, and perform local database acceptance.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--checksum-file", type=Path)
    parser.add_argument("--destination", type=Path, default=ROOT)
    args = parser.parse_args()
    checksum = args.checksum_file or args.archive.with_name(args.archive.name + ".sha256")
    try:
        result = install(args.archive, checksum, args.destination)
    except (OSError, ValueError, zipfile.BadZipFile, RuntimeError) as error:
        parser.exit(1, f"Installation stopped: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
