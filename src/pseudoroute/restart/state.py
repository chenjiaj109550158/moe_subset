"""Atomic run-local artifacts, identity validation and OS advisory locks."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast


def digest(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def object_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def guarded_run_dir(path: Path, repository: Path) -> Path:
    p, root = path.resolve(), (repository / "artifacts/restart_v1").resolve()
    if not p.is_relative_to(root) or p == root:
        raise ValueError(
            "output must be a new run under artifacts/restart_v1; historical paths protected"
        )
    return p


@contextlib.contextmanager
def run_lock(path: Path, run_id: str) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        f.seek(0)
        f.truncate()
        json.dump(
            {
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "run_id": run_id,
                "process_start_ticks": Path("/proc/self/stat").read_text().split()[21],
            },
            f,
        )
        f.flush()
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def load_row(path: Path, identity: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    checksum = data.pop("checksum")
    if data.get("identity") != identity or checksum != object_hash(data):
        raise ValueError(f"row identity/checksum changed: {path}; never merge incompatible results")
    if data.get("state") != "COMPLETE":
        return None
    return cast(dict[str, Any], data)


def save_row(path: Path, row: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"never overwrite measured row: {path}")
    write_json(path, {**row, "checksum": object_hash(row)})
