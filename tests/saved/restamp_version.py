"""Re-stamp the frozen saved folders under ``tests/saved/`` with this MIME's version, at a release's version bump.

A saved folder records the MIME that wrote it: the document's ``mime`` line, and in the lock the environment's
``mime``, the resolved record's ``vocabulary.mime`` and each of MIME's own entries' ``distribution_version``. The
resolved record and the document are inside the input hash, so a load under another version refuses the folder
(``load.input-hash``). The folders here are of earlier saved forms, written by MIME at earlier commits, and no code of
this version writes them again; their tests hold the earlier forms' upgrade, which needs the folder's MIME to be the
one that loads it. So at a bump each is re-stamped: the version strings replaced, and the lock's ``document`` and
``input`` hashes, which those strings move, computed again the way a load computes them. Nothing else changes.

For each folder it checks, before writing anything, that the hashes it computes from the stored files are the stored
hashes (so the computation is the one the load makes) and that the lock's text is the text ``save`` writes for it (so a
rewrite changes values, not layout); and after, that the document differs from the stored one only in its ``mime``
line, that the ``derived`` and ``observations`` hashes did not move, and that the old version is nowhere in either file.

    python tests/saved/restamp_version.py 0.3.0.dev0          # to mime.__version__
    python tests/saved/restamp_version.py 0.3.0.dev0 --check  # write nothing; exit 1 if a folder needs it

Prints each folder with the hashes that moved."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import mime
from mime.saved._document import document_hash, parse_yaml
from mime.saved._folder import lock_text
from mime.saved._record import input_hash

SAVED = Path(__file__).resolve().parent
DOCUMENT, LOCK = "experiment.mime.yaml", "mime.lock"


def folders() -> list[Path]:
    return sorted(lock.parent for lock in SAVED.rglob(LOCK))


def hashes_of(document: Any, lock: Any) -> dict[str, str]:
    return {"document": document_hash(document), "input": input_hash(document, lock["resolved"], lock["entries"])}


def restamped_lock(lock: Any, old: str, new: str) -> Any:
    held = json.loads(json.dumps(lock))
    if held["environment"].get("mime") == old:
        held["environment"]["mime"] = new
    if held["resolved"]["vocabulary"].get("mime") == old:
        held["resolved"]["vocabulary"]["mime"] = new
    for entry in held["entries"]:
        if entry.get("distribution") == "mime-engine" and entry.get("distribution_version") == old:
            entry["distribution_version"] = new
    return held


def restamp(folder: Path, old: str, new: str, write: bool) -> list[str]:
    """What moved in ``folder``, or raises ``SystemExit`` naming the check that failed."""
    where = folder.relative_to(SAVED)
    document_text = (folder / DOCUMENT).read_text(encoding="utf-8")
    lock_file = (folder / LOCK).read_text(encoding="utf-8")
    lock = json.loads(lock_file)
    document = parse_yaml(document_text)
    if lock_text(lock) != lock_file:
        raise SystemExit(f"{where}: the lock is not the text save writes for it; it would be rewritten in another layout")
    stored = {key: lock["hashes"][key] for key in ("document", "input")}
    if hashes_of(document, lock) != stored:
        raise SystemExit(f"{where}: the hashes computed from the stored files are not the stored ones")
    line = f"mime: {old}\n"
    if document_text.count(line) != 1:
        return []
    new_document_text = document_text.replace(line, f"mime: {new}\n")
    new_document = parse_yaml(new_document_text)
    if {**new_document, "mime": old} != document:  # type: ignore[dict-item]
        raise SystemExit(f"{where}: the document changed beyond its mime line")
    new_lock = restamped_lock(lock, old, new)
    new_lock["hashes"] = {**lock["hashes"], **hashes_of(new_document, new_lock)}
    text = lock_text(new_lock)
    if old in new_document_text or old in text:
        raise SystemExit(f"{where}: {old} is still in the folder after the re-stamp")
    if [new_lock["hashes"][key] for key in ("derived", "observations")] != \
            [lock["hashes"][key] for key in ("derived", "observations")]:
        raise SystemExit(f"{where}: a derived or observations hash moved")
    if write:
        (folder / DOCUMENT).write_text(new_document_text, encoding="utf-8", newline="\n")
        (folder / LOCK).write_text(text, encoding="utf-8", newline="\n")
    return [key for key in ("document", "input") if new_lock["hashes"][key] != lock["hashes"][key]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("old", help="the version the folders hold now")
    parser.add_argument("--to", default=mime.__version__, help="the version to stamp (default: this MIME's)")
    parser.add_argument("--check", action="store_true", help="write nothing; exit 1 if a folder holds the old version")
    arguments = parser.parse_args(argv)
    needed = False
    for folder in folders():
        moved = restamp(folder, arguments.old, arguments.to, write=not arguments.check)
        needed |= bool(moved)
        state = f"{', '.join(moved)} hashes moved" if moved else f"does not hold {arguments.old}"
        print(f"{folder.relative_to(SAVED)}: {state}")
    return 1 if arguments.check and needed else 0


if __name__ == "__main__":
    sys.exit(main())
