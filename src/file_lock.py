"""Nonblocking process locks for persistent, writable sidecar lock files."""
from __future__ import annotations

import errno
import os
from typing import IO, Any

if os.name == "nt":
    import msvcrt
else:
    import fcntl


def lock_exclusive_nonblocking(file: IO[Any]) -> None:
    """Hold an exclusive lock until ``file`` closes; fail immediately if busy.

    Every participant must lock the same sidecar file and leave it in place.
    Windows locks its first byte, including beyond EOF for an empty file, so no
    contents need to be written. The caller still owns the open file and its
    lifetime. Errors other than lock contention are allowed to propagate.
    """
    descriptor = file.fileno()
    if os.name != "nt":
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return

    position = os.lseek(descriptor, 0, os.SEEK_CUR)
    os.lseek(descriptor, 0, os.SEEK_SET)
    try:
        try:
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        except OSError as error:
            # For _locking(LK_NBLCK), EACCES specifically means a conflicting
            # byte-range lock. Bad descriptors/arguments and seek errors keep
            # their original exception instead of masquerading as contention.
            if error.errno == errno.EACCES:
                raise BlockingIOError(errno.EAGAIN, "File lock is already held") from error
            raise
    finally:
        os.lseek(descriptor, position, os.SEEK_SET)
