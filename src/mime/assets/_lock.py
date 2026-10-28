"""The asset references an experiment folder's lock names, read without loading the experiment.

The lock is JSON, so reading it runs no code and needs no entry installed: ``mime assets fetch <folder>`` can store what
a folder needs before the folder can be loaded. The references sit under the lock's ``assets`` key, a list of
:class:`~mime.assets.AssetRecord`. The key is absent until a build records its first asset, and then this reading
needs no change: the build writes the key, and this finds it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import cast

from mime.assets._reference import AssetRef
from mime.build import BuildRefused, Refusal, RefusalClass

#: The lock's file name in a folder (``mime.saved.LOCK_FILE``, which a test holds this to) and the key of its references.
LOCK_FILE = "mime.lock"
ASSETS_KEY = "assets"


def _unreadable(folder: Path, problem: str) -> BuildRefused:
    return BuildRefused([Refusal(
        "asset.lock", RefusalClass.USER, f"{folder / LOCK_FILE}: {problem}",
        "give the folder a saved experiment was written to, or an asset's name")])


def folder_assets(folder: str | os.PathLike[str]) -> tuple[AssetRef, ...]:
    """The asset references the lock in ``folder`` names, in the lock's order and each once; none when the lock
    records none."""
    where = Path(folder)
    try:
        lock: object = json.loads((where / LOCK_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise _unreadable(where, f"cannot be read as JSON ({error})") from error
    if not isinstance(lock, dict):
        raise _unreadable(where, "is not a JSON object")
    listed = cast("dict[str, object]", lock).get(ASSETS_KEY, [])
    if not isinstance(listed, list):
        raise _unreadable(where, f"{ASSETS_KEY!r} is not a list")
    found: dict[str, AssetRef] = {}
    for record in cast("list[object]", listed):
        if not isinstance(record, dict):
            raise _unreadable(where, f"an entry of {ASSETS_KEY!r} is not an object")
        ref = AssetRef.from_record(cast("dict[str, object]", record))
        found.setdefault(ref.sha256, ref)
    return tuple(found.values())
