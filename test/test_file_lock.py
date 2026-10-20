"""Exercise native interprocess locks without models, databases or services."""
from __future__ import annotations

import errno
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from src.file_lock import lock_exclusive_nonblocking


ROOT = Path(__file__).resolve().parents[1]
BUSY_EXIT = 23
ATTEMPT = """
import sys
from pathlib import Path
from src.file_lock import lock_exclusive_nonblocking

with Path(sys.argv[1]).open("a+b") as lock:
    try:
        lock_exclusive_nonblocking(lock)
    except BlockingIOError:
        raise SystemExit(23)
"""
HOLD = """
import sys
from pathlib import Path
from src.file_lock import lock_exclusive_nonblocking

with Path(sys.argv[1]).open("a+b") as lock:
    lock_exclusive_nonblocking(lock)
    Path(sys.argv[2]).write_text("ready", encoding="ascii")
    sys.stdin.readline()
"""


class FileLockTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "工作库.lock"

    def attempt_in_child(self):
        result = subprocess.run(
            [sys.executable, "-B", "-c", ATTEMPT, str(self.path)],
            cwd=ROOT, capture_output=True, text=True, timeout=5,
        )
        self.assertIn(result.returncode, (0, BUSY_EXIT), result.stderr)
        return result.returncode

    def test_empty_lock_blocks_another_process_and_close_releases_it(self):
        with self.path.open("a", encoding="utf-8") as lock:
            lock_exclusive_nonblocking(lock)
            self.assertEqual(self.attempt_in_child(), BUSY_EXIT)
        self.assertEqual(self.attempt_in_child(), 0)
        self.assertEqual(self.path.read_bytes(), b"")

    def test_existing_file_position_and_contents_are_preserved(self):
        self.path.write_bytes(b"persistent lock file")
        with self.path.open("a+b") as lock:
            lock.seek(7)
            lock_exclusive_nonblocking(lock)
            self.assertEqual(lock.tell(), 7)
            # The child starts at EOF; its lock must still conflict at byte 0.
            self.assertEqual(self.attempt_in_child(), BUSY_EXIT)
        self.assertEqual(self.path.read_bytes(), b"persistent lock file")

    def test_exception_closes_the_handle_and_releases_the_lock(self):
        with self.assertRaisesRegex(RuntimeError, "interrupted operation"):
            with self.path.open("a+b") as lock:
                lock_exclusive_nonblocking(lock)
                raise RuntimeError("interrupted operation")
        self.assertEqual(self.attempt_in_child(), 0)

    def test_process_termination_releases_the_lock(self):
        ready = self.path.with_suffix(".ready")
        child = subprocess.Popen(
            [sys.executable, "-B", "-c", HOLD, str(self.path), str(ready)],
            cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        try:
            deadline = time.monotonic() + 5
            while not ready.exists():
                if child.poll() is not None:
                    self.fail(child.communicate(timeout=5)[1].decode(errors="replace"))
                if time.monotonic() >= deadline:
                    self.fail("Child did not acquire its lock within 5 seconds")
                time.sleep(.01)
            with self.path.open("a+b") as lock:
                with self.assertRaises(BlockingIOError):
                    lock_exclusive_nonblocking(lock)
            # Bypass context-manager cleanup to exercise OS release on exit.
            child.kill()
            child.communicate(timeout=5)
        finally:
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=5)
        self.assertEqual(self.attempt_in_child(), 0)

    def test_invalid_descriptor_is_not_reported_as_contention(self):
        descriptor = os.open(self.path, os.O_CREAT | os.O_RDWR)
        os.close(descriptor)

        class InvalidFile:
            def fileno(self):
                return descriptor

        with self.assertRaises(OSError) as raised:
            lock_exclusive_nonblocking(InvalidFile())
        self.assertNotIsInstance(raised.exception, BlockingIOError)
        self.assertEqual(raised.exception.errno, errno.EBADF)


if __name__ == "__main__":
    unittest.main()
