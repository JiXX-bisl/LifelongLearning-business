from __future__ import annotations

import unittest
from pathlib import Path

from src.lifelong_learning.tasks.enterprise_query_bench.schema_catalog import (
    TABLE_PATTERN,
    basic_table_schemas,
    build_basic_schema_catalog,
    build_compact_schemas,
    render_basic_schema_catalog,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = (
    PROJECT_ROOT / "data/sql/synthetic_enterprise_v30_120_v1.0.2/database_schema.sql"
)


class BasicSchemaMappingTests(unittest.TestCase):
    """Pure-file schema contracts; no database or model calls are required."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
        # Labels deliberately differ from the guided dictionary.  The basic
        # catalog must use these explicit labels rather than guided fallbacks.
        cls.mapping = {
            line.strip().split("`", 2)[1]: f"短标签:{line.strip().split('`', 2)[1]}"
            for line in cls.schema_sql.splitlines()
            if line.strip().startswith("`")
        }

    def build(self, schema_sql: str | None = None, mapping: dict[str, str] | None = None):
        return build_basic_schema_catalog(
            self.schema_sql if schema_sql is None else schema_sql,
            self.mapping if mapping is None else mapping,
        )

    def test_seven_tables_have_all_120_declarations_and_80_shared_definitions(self) -> None:
        catalog = self.build()
        self.assertEqual(len(catalog["table_fields"]), 7)
        self.assertEqual(sum(map(len, catalog["table_fields"].values())), 120)
        self.assertEqual(len(catalog["field_definitions"]), 80)
        self.assertEqual(set(catalog["field_definitions"]), set(self.mapping))
        self.assertEqual(catalog["field_mapping"], self.mapping)
        self.assertEqual(
            catalog["field_definitions"]["registered_capital"],
            "DECIMAL(18,2) NOT NULL",
        )
        self.assertEqual(catalog["field_definitions"]["related_person_id"], "VARCHAR(24) DEFAULT NULL")

    def test_render_defines_each_shared_field_once_and_concentrates_common_foreign_keys(self) -> None:
        rendered = render_basic_schema_catalog(self.build())
        mapping_lines = [line for line in rendered.splitlines() if " → " in line]
        self.assertEqual(len(mapping_lines), 80)
        self.assertEqual(len(set(line.split(" → ", 1)[0] for line in mapping_lines)), 80)
        self.assertEqual(rendered.count("enterprise_id → "), 1)
        self.assertEqual(
            rendered.count("各表enterprise_id外键均关联enterprise_basic.enterprise_id"), 1,
        )
        self.assertNotIn(
            "FOREIGN KEY (`enterprise_id`) REFERENCES `enterprise_basic` (`enterprise_id`)",
            rendered,
        )
        self.assertIn(
            "FOREIGN KEY (`related_person_id`, `enterprise_id`) "
            "REFERENCES `enterprise_person` (`person_id`, `enterprise_id`)",
            rendered,
        )

    def test_catalog_preserves_raw_keys_and_describe_uses_only_short_labels(self) -> None:
        catalog = self.build()
        expected_key = (
            "CONSTRAINT `fk_v30_negative_person` FOREIGN KEY (`related_person_id`, `enterprise_id`) "
            "REFERENCES `enterprise_person` (`person_id`, `enterprise_id`)"
        )
        self.assertIn(expected_key, catalog["table_keys"]["enterprise_negative_event"])
        schemas = basic_table_schemas(catalog)
        self.assertEqual(set(schemas), set(catalog["table_fields"]))
        self.assertIn(expected_key, schemas["enterprise_negative_event"])
        self.assertIn("- person_entity_id: VARCHAR(40) NOT NULL；短标签:person_entity_id", schemas["enterprise_person"])
        self.assertNotIn("短标签:person_entity_id", schemas["enterprise_basic"])
        self.assertNotIn("修订号仅能", "\n".join(schemas.values()))

    def test_missing_or_unexpected_mapping_is_rejected(self) -> None:
        incomplete = dict(self.mapping)
        del incomplete["person_entity_id"]
        with self.assertRaisesRegex(ValueError, "missing=.*person_entity_id"):
            self.build(mapping=incomplete)
        extra = {**self.mapping, "private_goal": "私有目标"}
        with self.assertRaisesRegex(ValueError, "unexpected=.*private_goal"):
            self.build(mapping=extra)
        with self.assertRaisesRegex(ValueError, "nonempty strings"):
            self.build(mapping={**self.mapping, "enterprise_id": ""})

    def test_missing_duplicate_and_unknown_tables_are_rejected(self) -> None:
        first_table = TABLE_PATTERN.search(self.schema_sql)
        assert first_table is not None
        incomplete = self.schema_sql[:first_table.start()] + self.schema_sql[first_table.end():]
        with self.assertRaisesRegex(ValueError, "complete.*seven-table"):
            self.build(schema_sql=incomplete)
        with self.assertRaisesRegex(ValueError, "duplicate table"):
            self.build(schema_sql=self.schema_sql + "\n" + first_table.group(0))
        unknown = self.schema_sql.replace("CREATE TABLE `enterprise_basic`", "CREATE TABLE `other_table`", 1)
        with self.assertRaisesRegex(ValueError, "Unexpected.*other_table"):
            self.build(schema_sql=unknown)

    def test_conflicting_shared_definition_and_duplicate_fields_are_rejected(self) -> None:
        conflicting = self.schema_sql.replace(
            "`enterprise_id` VARCHAR(20) NOT NULL", "`enterprise_id` VARCHAR(21) NOT NULL", 1,
        )
        with self.assertRaisesRegex(ValueError, "Conflicting shared field definition for enterprise_id"):
            self.build(schema_sql=conflicting)
        duplicate = self.schema_sql.replace(
            "  `enterprise_id` VARCHAR(20) NOT NULL,",
            "  `enterprise_id` VARCHAR(20) NOT NULL,\n  `enterprise_id` VARCHAR(20) NOT NULL,",
            1,
        )
        with self.assertRaisesRegex(ValueError, "Duplicate field in enterprise_basic: enterprise_id"):
            self.build(schema_sql=duplicate)

    def test_changed_or_missing_foreign_key_is_rendered_without_invented_shared_rule(self) -> None:
        changed = self.schema_sql.replace(
            "CONSTRAINT `fk_v30_data_coverage_enterprise` FOREIGN KEY (`enterprise_id`) "
            "REFERENCES `enterprise_basic` (`enterprise_id`)",
            "CONSTRAINT `fk_v30_data_coverage_enterprise` FOREIGN KEY (`enterprise_id`) "
            "REFERENCES `enterprise_basic` (`enterprise_id`) ON DELETE CASCADE",
            1,
        )
        rendered = render_basic_schema_catalog(self.build(schema_sql=changed))
        self.assertNotIn("各表enterprise_id外键均关联", rendered)
        self.assertIn("ON DELETE CASCADE", rendered)
        self.assertEqual(rendered.count("FOREIGN KEY (`enterprise_id`)"), 6)

    def test_new_catalog_does_not_mutate_guided_schemas(self) -> None:
        before = build_compact_schemas(self.schema_sql, schema_version="3.0")
        self.build()
        after = build_compact_schemas(self.schema_sql, schema_version="3.0")
        self.assertEqual(after, before)
        self.assertNotIn("短标签:", "\n".join(after.values()))

    def test_new_interface_rejects_other_schema_versions(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires Schema 3.0"):
            build_basic_schema_catalog(self.schema_sql, self.mapping, schema_version="2.1")


if __name__ == "__main__":
    unittest.main()
