"""Pinned external SQL dataset adapters; public inputs never contain gold SQL.

Geography v4 is from Finegan-Dollak et al. (ACL 2018), distributed by
jkkummerfeld/text2sql-data. This adapter uses our execution comparison baseline,
not the publication's SQL matching score. Source splits remain explicit.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sqlite3
from urllib.request import urlopen
from .comparison import COMPARISON_VERSION

GEOGRAPHY_COMMIT = "5050bad2309a93636ea4788d3660c9202bf11e0e"
GEOGRAPHY_REPOSITORY = "https://github.com/jkkummerfeld/text2sql-data"
GEOGRAPHY_FILES = {
    "geography.json": "39bf45466efa21bec9e3df8a7da7a7e7fcc7a190",
    "geography-db.sql": "61793693baf330895eb12feb5d07a1a944dbc37c",
    "geography-schema.csv": "ea1eac66923eede9a160ccabec79bcbd33846920",
}


def _git_blob_sha(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def verify_geography_sources(source_dir):
    source_dir = Path(source_dir)
    verified = {}
    for name, expected in GEOGRAPHY_FILES.items():
        content = (source_dir / name).read_bytes()
        if _git_blob_sha(content) != expected:
            raise ValueError(f"Source does not match pinned Geography v4 blob: {name}")
        verified[name] = {"git_blob": expected, "sha256": hashlib.sha256(content).hexdigest(),
                          "bytes": len(content)}
    return verified


def download_geography_sources(source_dir):
    """Fetch official immutable files; cached files must match source identities."""
    source_dir = Path(source_dir)
    source_dir.mkdir(parents=True, exist_ok=True)
    for name, expected in GEOGRAPHY_FILES.items():
        path = source_dir / name
        if path.exists():
            if _git_blob_sha(path.read_bytes()) != expected:
                raise ValueError(f"Refusing to overwrite mismatched source: {path}")
            continue
        url = ("https://raw.githubusercontent.com/jkkummerfeld/text2sql-data/"
               + GEOGRAPHY_COMMIT + "/data/" + name)
        with urlopen(url, timeout=30) as response:
            content = response.read()
        if _git_blob_sha(content) != expected:
            raise ValueError(f"Downloaded source hash mismatch: {name}")
        path.write_bytes(content)
    return verify_geography_sources(source_dir)


def geography_cases(source_dir, *, split="dev", split_by="query"):
    """Load all questions for a declared split, in original file/sentence order.

    Mirrors the source converter's variable substitution and first-SQL choice.
    It deliberately does not de-duplicate questions or drop failed gold queries.
    Gold is returned separately from each public input, for evaluator use only.
    """
    if split not in {"train", "dev", "test"} or split_by not in {"query", "question"}:
        raise ValueError("Expected train/dev/test and query/question split")
    verify_geography_sources(source_dir)
    examples = json.loads((Path(source_dir) / "geography.json").read_text())
    cases = []
    for query_index, entry in enumerate(examples):
        for sentence_index, sentence in enumerate(entry["sentences"]):
            assignment = (entry["query-split"] if split_by == "query"
                          else sentence["question-split"])
            if assignment != split:
                continue
            question, sql = sentence["text"], entry["sql"][0]
            defaults = {variable["name"]: variable["example"] for variable in entry["variables"]}
            for name in sorted(sentence["variables"], key=len, reverse=True):
                value = sentence["variables"][name] or defaults[name]
                # Replacements are the published dataset values, not model input.
                question, sql = question.replace(name, value), sql.replace(name, value)
            cases.append({"public": {"id": f"geo-{query_index:04d}-{sentence_index:03d}",
                                      "question": question},
                          "gold_sql": sql, "query_split": entry["query-split"],
                          "question_split": sentence["question-split"]})
    return cases


def _create_geography_database(source_dir, database_path):
    """Convert only the pinned dump's CREATE/INSERT statements into SQLite.

    Storage types/values remain unchanged. MySQL engine/lock/session commands do
    not describe data and are omitted. This is specific to the verified dump.
    """
    dump = (Path(source_dir) / "geography-db.sql").read_text()
    statements = re.findall(r"^CREATE TABLE .*?;|^INSERT INTO .*?;", dump, re.MULTILINE | re.DOTALL)
    if len(statements) != 14:
        raise ValueError("Unexpected Geography dump structure")
    with sqlite3.connect(database_path) as connection:
        for statement in statements:
            statement = re.sub(r"\) ENGINE=InnoDB DEFAULT CHARSET=latin1;$", ");", statement)
            statement = statement.replace("\\'", "''")
            connection.execute(statement)
        tables = [row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        counts = {table: connection.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0]
                  for table in tables}
        if len(tables) != 7 or connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            raise ValueError("Geography database integrity failure")
        schema = "\n".join(row[0] + ";" for row in connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' ORDER BY name"))
    return counts, schema


def prepare_geography(source_dir, output_dir, *, split="dev", split_by="query", limit=20):
    """Write a fresh independent database/public/evaluator bundle and manifest.

    The default 20-item subset is an interface smoke baseline, not a leaderboard
    result. Set limit=None for the complete selected split. Gold execution errors
    are recorded and retained as UNKNOWN entries rather than silently excluded.
    """
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("limit must be positive or None")
    hashes = verify_geography_sources(source_dir)
    all_cases = geography_cases(source_dir, split=split, split_by=split_by)
    selected = all_cases if limit is None else all_cases[:limit]
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    database = output_dir / "database.sqlite3"
    counts, schema = _create_geography_database(source_dir, database)
    private, private_sql, errors = {}, {}, {}
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
        for case in selected:
            case_id = case["public"]["id"]
            private_sql[case_id] = case["gold_sql"]
            try:
                cursor = connection.execute(case["gold_sql"])
                private[case_id] = {"columns": [column[0] for column in cursor.description],
                                    "rows": cursor.fetchall()}
            except sqlite3.Error as error:
                private[case_id] = None
                errors[case_id] = str(error)
    public = {"namespace": "geography", "data_version": "text2sql-v4-5050bad",
              "schema": schema,
              "default_rules": "Geography benchmark uses historical US geography data. "
                               "Names are lowercase in the database. Use SQLite SELECT queries. "
                               "Compare full rows, preserve duplicates; unordered result comparison.",
              "cases": [case["public"] for case in selected]}
    manifest = {"dataset": "Geography", "version": 4, "commit": GEOGRAPHY_COMMIT,
                "repository": GEOGRAPHY_REPOSITORY, "source_files": hashes,
                "split": split, "split_by": split_by, "available_cases": len(all_cases),
                "selected_ids": [case["public"]["id"] for case in selected],
                "selection": "source_order_prefix" if limit else "complete_split",
                "table_rows": counts, "reference_errors": errors,
                "scoring": {"version": COMPARISON_VERSION, "ordered": False,
                            "column_correspondence": "explicit_or_unique_name_else_equal_width_position_or_global_projection",
                            "extra_columns": "ignored_after_projection", "duplicates": "preserved",
                            "numeric_tolerance": 0,
                            "purpose": "new execution-result baseline, not published SQL matching accuracy"},
                "split_assignments": {case["public"]["id"]:
                                      {"query": case["query_split"], "question": case["question_split"]}
                                      for case in selected}}
    for name, value in {"public.json": public, "private.json": private,
                        "private_sql.json": private_sql, "manifest.json": manifest,
                        "comparison_policy.json": {"ordered": False}}.items():
        (output_dir / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    return {"output_dir": str(output_dir), "case_count": len(selected),
            "reference_error_count": len(errors), "table_rows": counts}
