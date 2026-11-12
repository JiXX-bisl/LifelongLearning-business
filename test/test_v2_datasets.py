import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

from src.sql_learning import datasets


@pytest.fixture
def source(tmp_path, monkeypatch):
    folder = tmp_path / 'source'
    folder.mkdir()
    dump = '\n'.join(f'CREATE TABLE `t{i}` (\n  `name` text\n) ENGINE=InnoDB DEFAULT CHARSET=latin1;\n'
                     f"INSERT INTO `t{i}` VALUES ('alpha');" for i in range(7))
    examples = [{"query-split": "dev", "sentences": [
        {"question-split": "train", "text": "find name0", "variables": {"name0": "alpha"}},
        {"question-split": "test", "text": "show name0", "variables": {"name0": ""}}],
        "sql": ['SELECT name FROM t0 WHERE name = "name0"'],
        "variables": [{"name": "name0", "example": "alpha", "location": "both", "type": "name"}]}]
    contents = {'geography.json': json.dumps(examples), 'geography-db.sql': dump,
                'geography-schema.csv': 'Table Name, Field Name\nt0, name\n'}
    hashes = {}
    for name, content in contents.items():
        (folder / name).write_text(content)
        value = content.encode()
        hashes[name] = hashlib.sha1(b'blob ' + str(len(value)).encode() + b'\0' + value).hexdigest()
    monkeypatch.setattr(datasets, 'GEOGRAPHY_FILES', hashes)
    return folder


def test_declared_split_and_variable_fallback(source):
    cases = datasets.geography_cases(source)
    assert [case['public']['question'] for case in cases] == ['find alpha', 'show alpha']
    assert len(datasets.geography_cases(source, split='train', split_by='question')) == 1
    assert datasets.geography_cases(source, split='train', split_by='query') == []


def test_bundle_keeps_gold_private_and_refuses_overwrite(source, tmp_path):
    output = tmp_path / 'bundle'
    report = datasets.prepare_geography(source, output)
    assert report['reference_error_count'] == 0
    public = json.loads((output / 'public.json').read_text())
    assert public['cases'] == [{'id': 'geo-0000-000', 'question': 'find alpha'},
                               {'id': 'geo-0000-001', 'question': 'show alpha'}]
    assert 'gold_sql' not in json.dumps(public)
    assert 'SELECT name FROM' not in json.dumps(public)
    assert json.loads((output / 'private.json').read_text())['geo-0000-000']['rows'] == [['alpha']]
    with pytest.raises(FileExistsError):
        datasets.prepare_geography(source, output)


def test_source_mutation_rejected_before_output_creation(source, tmp_path):
    (source / 'geography.json').write_text('[]')
    with pytest.raises(ValueError, match='pinned'):
        datasets.prepare_geography(source, tmp_path / 'bad')
    assert not (tmp_path / 'bad').exists()


def test_gold_failures_remain_in_denominator(source, tmp_path, monkeypatch):
    cases = datasets.geography_cases(source)
    cases[1]['gold_sql'] = 'SELECT missing FROM nowhere'
    monkeypatch.setattr(datasets, 'geography_cases', lambda *a, **k: cases)
    output = tmp_path / 'bundle'
    report = datasets.prepare_geography(source, output)
    assert report['case_count'] == 2 and report['reference_error_count'] == 1
    assert json.loads((output / 'private.json').read_text())['geo-0000-001'] is None
    assert len(json.loads((output / 'public.json').read_text())['cases']) == 2


@pytest.mark.skipif(not Path('data/external_v2/geography/source/geography.json').exists(),
                    reason='Pinned external download not installed')
def test_official_geography_dev_subset_executes(tmp_path):
    source = Path('data/external_v2/geography/source')
    report = datasets.prepare_geography(source, tmp_path / 'real')
    assert report['case_count'] == 20
    assert report['reference_error_count'] == 0
    assert sum(report['table_rows'].values()) == 925
