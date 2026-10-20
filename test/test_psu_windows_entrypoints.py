"""Offline Windows launcher checks; never invoke a real validation command."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.enterprise_query_bench import run_windows_tests as test_launcher
from scripts.enterprise_query_bench import run_windows_validation as launcher


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "enterprise_query_bench"
POWERSHELL = shutil.which("powershell") or shutil.which("pwsh")


class WindowsEntryPointTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "中文 space"
        self.root.mkdir()
        self.addCleanup(patch.stopall)
        patch.object(launcher, "ROOT", self.root).start()
        patch.dict(os.environ, {}, clear=True).start()
        self.output = io.StringIO()
        self.stdout = redirect_stdout(self.output)
        self.stderr = redirect_stderr(self.output)
        self.stdout.__enter__()
        self.stderr.__enter__()
        self.addCleanup(self.stdout.__exit__, None, None, None)
        self.addCleanup(self.stderr.__exit__, None, None, None)

    def test_existing_directory_or_file_does_not_start_validation(self):
        for is_directory in (True, False):
            with self.subTest(is_directory=is_directory):
                destination = self.root / str(is_directory)
                if is_directory:
                    destination.mkdir()
                else:
                    destination.write_bytes(b"keep existing file")
                with patch.object(launcher.subprocess, "run") as child:
                    result = launcher.main(["--profile", "direct", "--run-dir", str(destination)])
                self.assertEqual(result, 1)
                child.assert_not_called()
                if not is_directory:
                    self.assertEqual(destination.read_bytes(), b"keep existing file")

    def test_dangling_symlink_does_not_start_validation(self):
        destination = self.root / "dangling"
        try:
            destination.symlink_to(self.root / "missing", target_is_directory=True)
        except OSError as error:
            self.skipTest(f"Creating symlinks is unavailable: {error}")
        with patch.object(launcher.subprocess, "run") as child:
            result = launcher.main(["--profile", "direct", "--run-dir", str(destination)])
        self.assertEqual(result, 1)
        child.assert_not_called()
        self.assertTrue(destination.is_symlink())

    def test_sequence_uses_environment_interpreter_and_unicode_paths(self):
        os.environ.update(PSU_COVERAGE_RUN="运行 空格", PSU_COVERAGE_CASE_LIMIT="7")
        with patch.object(launcher.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as child:
            self.assertEqual(launcher.main(["--profile", "coverage"]), 0)
        calls = child.call_args_list
        self.assertEqual(len(calls), 5)
        commands = [call.args[0] for call in calls]
        self.assertEqual([command[6] for command in commands],
                         ["prepare", "run", "run", "run", "summarize"])
        self.assertEqual([command[-1] for command in commands[1:4]], ["baseline", "fifo", "adaptive"])
        self.assertEqual(commands[0][-2:], ["--case-limit", "7"])
        self.assertTrue(commands[0][8].endswith("natural_coverage_v3_windows_cuda.json"))
        for call in calls:
            self.assertEqual(call.args[0][:6], [
                sys.executable, "-B", "-X", "utf8", "-m", "src.psu_memory.cli.validate_local",
            ])
            self.assertEqual(call.kwargs["cwd"], self.root)
            environment = call.kwargs["env"]
            self.assertEqual(environment["PSU_COVERAGE_RUN"], str(self.root / "运行 空格"))
            self.assertEqual(environment["PYTHONUTF8"], "1")
            self.assertEqual(environment["PYTHONIOENCODING"], "utf-8")

    def test_explicit_arguments_override_environment(self):
        os.environ.update(PSU_DIRECT_RUN="old", PSU_DIRECT_CASE_LIMIT="99")
        with patch.object(launcher.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as child:
            result = launcher.main([
                "--profile", "direct", "--run-dir", "新 目录",
                "--config", "配置 文件.json", "--case-limit", "4",
            ])
        self.assertEqual(result, 0)
        self.assertEqual(child.call_args_list[0].args[0][6:], [
            "prepare", "--config", str(self.root / "配置 文件.json"),
            "--run-dir", str(self.root / "新 目录"), "--case-limit", "4",
        ])

    def test_baseline3_is_isolated_and_fixed_at_three_cases(self):
        os.environ.update(PSU_DIRECT_RUN="old", PSU_DIRECT_CASE_LIMIT="99",
                          PSU_COVERAGE_RUN="old", PSU_BASELINE3_CASE_LIMIT="99")
        with patch.object(launcher.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as child:
            self.assertEqual(launcher.main(["--profile", "baseline3"]), 0)
        self.assertEqual(child.call_count, 3)
        self.assertEqual(child.call_args_list[0].args[0][-2:], ["--case-limit", "3"])
        self.assertEqual(child.call_args_list[1].args[0][-2:], ["--mode", "baseline"])
        self.assertIn("baseline3", child.call_args_list[0].kwargs["env"]["PSU_BASELINE3_RUN"])
        with patch.object(launcher.subprocess, "run") as child:
            with self.assertRaises(SystemExit):
                launcher.main(["--profile", "baseline3", "--case-limit", "4"])
            child.assert_not_called()

    def test_failed_prepare_or_condition_stops_before_remaining_steps(self):
        for failed_step in range(4):
            with self.subTest(failed_step=failed_step):
                results = [SimpleNamespace(returncode=0)] * failed_step + [SimpleNamespace(returncode=23)]
                with patch.object(launcher.subprocess, "run", side_effect=results) as child:
                    self.assertEqual(launcher.main(["--profile", "direct"]), 23)
                self.assertEqual(child.call_count, failed_step + 1)

    def test_invalid_case_limit_is_rejected_before_start(self):
        for limit in ("0", "-1", "not-a-number"):
            with self.subTest(limit=limit):
                os.environ["PSU_DIRECT_CASE_LIMIT"] = limit
                with patch.object(launcher.subprocess, "run") as child:
                    with self.assertRaises(SystemExit):
                        launcher.main(["--profile", "direct"])
                child.assert_not_called()

    def test_offline_test_launcher_stops_after_failed_group(self):
        with patch.object(test_launcher.subprocess, "run", return_value=SimpleNamespace(returncode=8)) as child:
            self.assertEqual(test_launcher.main(["--profile", "coverage"]), 8)
        child.assert_called_once()
        self.assertEqual(child.call_args.args[0][:7], [
            sys.executable, "-B", "-X", "utf8", "-m", "pytest", "-q",
        ])

    def test_windows_configs_preserve_experiment_semantics(self):
        expected_changes = {
            "retrieval_config": "src/psu_memory/retrieval/retrieval_config.windows_cuda.json",
            "source_store": "data/derived/psu_memory/seeds/windows-cuda/seed.sqlite3",
            "acceptance_report": "data/reports/enterprise_v30_120/database_acceptance_windows.json",
        }
        for suffix in ("coverage_v3", "direct_v1", "baseline3_v1"):
            with self.subTest(profile=suffix):
                name = f"enterprise_v30_120_v1_0_2_natural_{suffix}"
                directory = ROOT / "configs" / "validation"
                original = json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
                windows = json.loads((directory / f"{name}_windows_cuda.json").read_text(encoding="utf-8"))
                self.assertEqual(windows, {**original, **expected_changes})


@unittest.skipUnless(POWERSHELL, "PowerShell is unavailable")
class PowerShellEntryPointTests(unittest.TestCase):
    def test_validation_wrappers_preserve_arguments_and_exit_code(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "中文 space"
            folder.mkdir()
            marker = folder / "arguments.json"
            (folder / "run_windows_validation.py").write_text(
                "import json, os, pathlib, sys\n"
                "pathlib.Path(os.environ['PSU_TEST_MARKER']).write_text("
                "json.dumps(sys.argv[1:], ensure_ascii=False), encoding='utf-8')\n"
                "sys.exit(23)\n", encoding="utf-8",
            )
            for name, profile in (("coverage_sql", "coverage"), ("direct_sql", "direct"),
                                  ("baseline_3", "baseline3")):
                with self.subTest(profile=profile):
                    script = folder / f"run_{name}_validation.ps1"
                    shutil.copyfile(SCRIPTS / script.name, script)
                    result = subprocess.run([
                        POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                        "-File", str(script), "-Python", sys.executable, "-RunDir", "运行 路径",
                        "-Config", "配置 文件.json", "-CaseLimit", "3",
                    ], env={**os.environ, "PSU_TEST_MARKER": str(marker)},
                        capture_output=True, timeout=20)
                    self.assertEqual(result.returncode, 23, result.stderr)
                    self.assertEqual(json.loads(marker.read_text(encoding="utf-8")), [
                        "--profile", profile, "--run-dir", "运行 路径",
                        "--config", "配置 文件.json", "--case-limit", "3",
                    ])

    def test_test_wrappers_select_offline_suite_and_preserve_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "中文 space"
            folder.mkdir()
            marker = folder / "arguments.json"
            (folder / "run_windows_tests.py").write_text(
                "import json, os, pathlib, sys\n"
                "pathlib.Path(os.environ['PSU_TEST_MARKER']).write_text("
                "json.dumps(sys.argv[1:]), encoding='utf-8')\n"
                "sys.exit(19)\n", encoding="utf-8",
            )
            for profile in ("direct", "coverage"):
                with self.subTest(profile=profile):
                    script = folder / f"run_{profile}_sql_tests.ps1"
                    shutil.copyfile(SCRIPTS / script.name, script)
                    result = subprocess.run([
                        POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                        "-File", str(script), "-Python", sys.executable,
                    ], env={**os.environ, "PSU_TEST_MARKER": str(marker)},
                        capture_output=True, timeout=20)
                    self.assertEqual(result.returncode, 19, result.stderr)
                    self.assertEqual(json.loads(marker.read_text(encoding="utf-8")),
                                     ["--profile", profile])

    def test_existing_directory_is_protected_through_real_powershell_wrapper(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "已有 运行"
            run_dir.mkdir()
            marker = run_dir / "config.json"
            previous = b'{"existing": true}\n'
            marker.write_bytes(previous)
            result = subprocess.run([
                POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                "-File", str(SCRIPTS / "run_direct_sql_validation.ps1"),
                "-Python", sys.executable, "-RunDir", str(run_dir),
            ], capture_output=True, timeout=20)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn(b"Run path already exists", result.stderr)
            self.assertEqual(marker.read_bytes(), previous)
            self.assertEqual(list(run_dir.iterdir()), [marker])


if __name__ == "__main__":
    unittest.main()
