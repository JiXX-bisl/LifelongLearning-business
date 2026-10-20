"""用户独立运行：材料包安装边界；仅临时文件，不访问数据库或模型。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
import tempfile
import unittest
import zipfile

from scripts.enterprise_query_bench.install_reproduction_inputs import install


class ReproductionBundleInstallTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name)
        self.project = self.workspace / "project"
        marker = self.project / "src/psu_memory/validation/runner.py"
        marker.parent.mkdir(parents=True)
        marker.write_text("# Fake cloned-project marker; never imported.\n", encoding="utf-8")
        self.archive = self.workspace / "inputs.zip"
        self.checksum = self.workspace / "inputs.zip.sha256"
        self.bundle_id = "test-inputs-v1"
        self.record = f"data/catalog/reproduction-bundles/{self.bundle_id}.json"
        # The .sqlite3 suffix is intentional: installation must preserve opaque
        # bytes without trying to open or validate a database.
        self.payloads = {
            "data/questions/questions.json": b'{"question":"example"}\r\n',
            "data/seeds/seed.sqlite3": b"opaque seed bytes\x00\xff\r\n",
            "provenance/formation.json": '{"来源":"形成题"}\n'.encode("utf-8"),
        }

    def write_bundle(self, payloads=None, *, expected_payloads=None, extra=None, symlinks=()):
        payloads = self.payloads if payloads is None else payloads
        expected_payloads = payloads if expected_payloads is None else expected_payloads
        manifest = {
            "schema_version": "1.0",
            "bundle_id": self.bundle_id,
            "files": [
                {"path": name, "bytes": len(content),
                 "sha256": hashlib.sha256(content).hexdigest()}
                for name, content in expected_payloads.items()
            ],
        }
        manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
        with zipfile.ZipFile(self.archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            bundle.writestr("bundle-manifest.json", manifest_bytes)
            for name, content in payloads.items():
                if name in symlinks:
                    info = zipfile.ZipInfo(name)
                    info.create_system = 3
                    info.external_attr = (stat.S_IFLNK | 0o777) << 16
                    bundle.writestr(info, content)
                else:
                    bundle.writestr(name, content)
            for name, content in (extra or {}).items():
                bundle.writestr(name, content)
        digest = hashlib.sha256(self.archive.read_bytes()).hexdigest()
        self.checksum.write_text(f"{digest}  {self.archive.name}\n", encoding="utf-8")
        return manifest_bytes

    def snapshot(self):
        return {
            path.relative_to(self.project).as_posix(): path.read_bytes()
            for path in self.project.rglob("*") if path.is_file()
        }

    def assert_no_payload_published(self, before):
        self.assertEqual(self.snapshot(), before)
        self.assertFalse((self.project / self.record).exists())
        self.assertEqual(list(self.project.glob(".reproduction-stage-*")), [])

    def test_archive_checksum_failure_publishes_nothing(self):
        self.write_bundle()
        self.checksum.write_text(f"{'0' * 64}  {self.archive.name}\n", encoding="utf-8")
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "Archive SHA-256 mismatch"):
            install(self.archive, self.checksum, self.project)
        self.assert_no_payload_published(before)
        self.assertFalse((self.project / "data").exists())

    def test_unsafe_member_paths_are_rejected_before_publication(self):
        invalid_names = [
            "../outside.txt",
            "data/../../outside.txt",
            "/data/absolute.txt",
            "data\\outside.txt",
            "data/./question.json",
            "data//question.json",
            "data/C:/question.json",
        ]
        before = self.snapshot()
        for name in invalid_names:
            with self.subTest(name=name):
                self.write_bundle({name: b"untrusted"})
                with self.assertRaisesRegex(ValueError, "Invalid bundle path"):
                    install(self.archive, self.checksum, self.project)
                self.assert_no_payload_published(before)
        self.assertFalse((self.workspace / "outside.txt").exists())

    def test_zip_symlink_is_rejected(self):
        name = "data/source-link"
        self.write_bundle({name: b"../../outside"}, symlinks=(name,))
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "Unsupported ZIP entry"):
            install(self.archive, self.checksum, self.project)
        self.assert_no_payload_published(before)

    def test_destination_symlink_is_rejected_without_writing_through_it(self):
        outside = self.workspace / "outside"
        outside.mkdir()
        link = self.project / "data"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("This platform does not permit creating a test symlink")
        self.write_bundle()
        with self.assertRaisesRegex(ValueError, "Refusing symlink"):
            install(self.archive, self.checksum, self.project)
        self.assertTrue(link.is_symlink())
        self.assertEqual(list(outside.iterdir()), [])
        self.assertFalse((self.project / "provenance").exists())

    def test_existing_conflict_rejects_all_new_files_before_writing(self):
        # The conflicting file comes after the new file in archive order.
        # Preflight must inspect the whole bundle, not publish that first file.
        conflict = self.project / "data/seeds/seed.sqlite3"
        conflict.parent.mkdir(parents=True)
        conflict.write_bytes(b"locally reviewed seed; retain")
        self.write_bundle()
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "Existing file differs"):
            install(self.archive, self.checksum, self.project)
        self.assert_no_payload_published(before)
        self.assertFalse((self.project / "data/questions/questions.json").exists())

    def test_success_preserves_bytes_and_writes_exact_installation_manifest(self):
        manifest_bytes = self.write_bundle()
        result = install(self.archive, self.checksum, self.project)
        for name, content in self.payloads.items():
            with self.subTest(name=name):
                installed = self.project / name
                self.assertEqual(installed.read_bytes(), content)
                self.assertEqual(hashlib.sha256(installed.read_bytes()).hexdigest(),
                                 hashlib.sha256(content).hexdigest())
        self.assertEqual((self.project / self.record).read_bytes(), manifest_bytes)
        self.assertEqual(result["bundle_id"], self.bundle_id)
        self.assertEqual(result["manifest"], self.record)
        self.assertEqual(result["installed_files"], len(self.payloads) + 1)
        self.assertEqual(result["identical_existing_files"], 0)
        self.assertEqual(list(self.project.glob(".reproduction-stage-*")), [])

    def test_identical_existing_files_and_repeated_install_are_allowed(self):
        existing_name = "data/seeds/seed.sqlite3"
        existing = self.project / existing_name
        existing.parent.mkdir(parents=True)
        existing.write_bytes(self.payloads[existing_name])
        before_stat = existing.stat()
        self.write_bundle()
        first = install(self.archive, self.checksum, self.project)
        self.assertEqual(first["installed_files"], len(self.payloads))
        self.assertEqual(first["identical_existing_files"], 1)
        self.assertEqual(existing.stat().st_mtime_ns, before_stat.st_mtime_ns)
        before = self.snapshot()
        second = install(self.archive, self.checksum, self.project)
        self.assertEqual(second["installed_files"], 0)
        self.assertEqual(second["identical_existing_files"], len(self.payloads) + 1)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(existing.stat().st_mtime_ns, before_stat.st_mtime_ns)

    def test_bad_payload_hash_prevents_publication_of_all_payloads(self):
        corrupted = dict(self.payloads)
        name = "data/seeds/seed.sqlite3"
        corrupted[name] = b"x" * len(corrupted[name])
        # The outer archive digest is valid. Only the per-file checksum is bad,
        # after a valid first payload has already been staged.
        self.write_bundle(corrupted, expected_payloads=self.payloads)
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "Payload checksum mismatch"):
            install(self.archive, self.checksum, self.project)
        self.assert_no_payload_published(before)
        self.assertFalse((self.project / "data").exists())
        self.assertFalse((self.project / "provenance").exists())

    def test_unlisted_zip_member_is_rejected(self):
        self.write_bundle(extra={"data/unlisted.txt": b"not approved by manifest"})
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "Archive contents differ from the manifest"):
            install(self.archive, self.checksum, self.project)
        self.assert_no_payload_published(before)

    def test_existing_installation_record_conflict_prevents_new_payloads(self):
        record = self.project / self.record
        record.parent.mkdir(parents=True)
        record.write_bytes(b'{"previous":"different manifest"}')
        self.write_bundle()
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "Existing file differs"):
            install(self.archive, self.checksum, self.project)
        self.assertEqual(self.snapshot(), before)
        self.assertFalse((self.project / "data/questions").exists())
        self.assertFalse((self.project / "provenance").exists())

    def test_bundle_file_cannot_also_be_a_parent_directory(self):
        before = self.snapshot()
        for child_path in ("data/source/item.json", "data/SOURCE/item.json"):
            with self.subTest(child_path=child_path):
                self.write_bundle({
                    "data/first.json": b"must not be published",
                    "data/source": b"regular file",
                    child_path: b"child requiring a directory",
                })
                with self.assertRaisesRegex(ValueError, "also used as a directory"):
                    install(self.archive, self.checksum, self.project)
                self.assert_no_payload_published(before)
                self.assertFalse((self.project / "data").exists())

    def test_case_colliding_installation_record_is_rejected_before_writing(self):
        colliding_record = self.record.replace(
            f"{self.bundle_id}.json", f"{self.bundle_id.upper()}.JSON")
        self.write_bundle({
            "data/first.json": b"must not be published",
            colliding_record: b"payload impersonating installation record",
        })
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "Installation record has a case-colliding path"):
            install(self.archive, self.checksum, self.project)
        self.assert_no_payload_published(before)
        self.assertFalse((self.project / "data").exists())


if __name__ == "__main__":
    unittest.main()
