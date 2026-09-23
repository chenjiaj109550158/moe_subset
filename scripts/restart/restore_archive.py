"""Restore the complete restart evidence with Python's standard library; never overwrite a run."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

RUN_ID = "20260922T031933Z"
REPO = Path(__file__).resolve().parents[2]


def relative(root: Path, name: str) -> Path:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != name:
        raise ValueError(f"Unsafe archive path: {name}")
    return root.joinpath(*path.parts)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as data:
        for block in iter(lambda: data.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_archive(archive: Path, manifest: dict) -> None:
    for part in manifest["parts"]:
        path = relative(archive, part["path"])
        if path.stat().st_size != part["bytes"] or sha256(path) != part["sha256"]:
            raise ValueError(f"Archive part checksum mismatch: {part['path']}")
    for category in ("readable_copies", "external_run_metadata"):
        for name, expected in manifest.get(category, {}).items():
            if sha256(relative(archive, name)) != expected:
                raise ValueError(f"Archive metadata checksum mismatch: {name}")


def restore(archive: Path, output: Path, manifest: dict) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing destination: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=output.name + ".restore-", dir=output.parent))
    promoted = False
    try:
        with tempfile.TemporaryFile() as combined:
            for part in manifest["parts"]:
                with relative(archive, part["path"]).open("rb") as data:
                    shutil.copyfileobj(data, combined)
            combined.seek(0)
            seen = set()
            with tarfile.open(fileobj=combined, mode="r:gz") as tar:
                for member in tar:
                    expected = manifest["files"].get(member.name)
                    if (
                        not member.isfile()
                        or expected is None
                        or member.name in seen
                        or member.size != expected["bytes"]
                    ):
                        raise ValueError(f"Unexpected archive member: {member.name}")
                    destination = relative(scratch, member.name)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    data = tar.extractfile(member)
                    if data is None:
                        raise ValueError(f"Missing file content: {member.name}")
                    digest = hashlib.sha256()
                    with data, destination.open("xb") as target:
                        while block := data.read(1024 * 1024):
                            target.write(block)
                            digest.update(block)
                    if digest.hexdigest() != expected["sha256"]:
                        raise ValueError(f"Restored checksum mismatch: {member.name}")
                    seen.add(member.name)
            if seen != set(manifest["files"]):
                raise ValueError("Archive does not contain all expected files")
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Destination appeared during restore: {output}")
        scratch.rename(output)
        promoted = True
    finally:
        if not promoted:
            shutil.rmtree(scratch)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, default=REPO / "docs/restart/archive" / RUN_ID)
    parser.add_argument("--output", type=Path, default=REPO / "artifacts/restart_v1" / RUN_ID)
    parser.add_argument(
        "--verify-only", action="store_true", help="Check compressed parts and readable copies"
    )
    args = parser.parse_args()
    manifest = json.loads((args.archive_dir / "archive_manifest.json").read_text())
    if manifest["schema_version"] != 1 or manifest["run_id"] != RUN_ID:
        raise ValueError("Unsupported archive format or run ID")
    check_archive(args.archive_dir, manifest)
    if not args.verify_only:
        restore(args.archive_dir, args.output.absolute(), manifest)
    print(
        json.dumps(
            {
                "state": "PASS",
                "operation": (
                    "verify_compressed_archive"
                    if args.verify_only
                    else "restore_and_verify_all_files"
                ),
                "files": manifest["archived_files"],
                "measured_rows": manifest["measured_rows"],
                "compressed_bytes": manifest["compressed_bytes"],
                "output": None if args.verify_only else str(args.output.absolute()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
