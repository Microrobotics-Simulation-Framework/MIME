"""The staged lock (design §2.9): the folder's lock with the record it will be published as, packed into the bundle in
place of the folder's own, so the published bundle names its own record and DOI.

A bundle cannot name the record it is uploaded to unless the record's DOI is known first, so the draft is made and its
DOI reserved before the bundle is packed. The staged lock is the folder's lock with a ``publication`` key (the service,
the version's record and DOI, its concept's record, the concept's DOI where the service states it, the version's
number) and, for each asset the record holds, the record's own location of it put first, its earlier locations kept as
backups: ``doi:<DOI>/<file>`` on production; on the sandbox, whose test-prefix DOIs do not resolve through doi.org, the
file's address in the sandbox's records API. Both are outside the input hash, so a folder whose lock is the staged lock
loads and verifies as before, and staging is idempotent: the same publication staged again gives the same bytes.

The folder is packed with the staged lock through a copy of it beside push's record, its files linked where the file
system allows (copied where it does not), its lock replaced; the folder itself is never written, except by
:func:`rewrite_lock` after a production publish, so that ``mime pack <folder>`` makes the published bundle again.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import cast
from urllib.parse import quote

from mime.build import BuildRefused, Refusal, RefusalClass
from mime.push._service import Service
from mime.saved._schema import PublicationRecord

#: The folder push packs with the staged lock, below its staging directory.
COPY = "folder"


def own_location(service: Service, record: str, doi: str, key: str) -> str:
    """Where the record ``record`` holds its file ``key``, as a lock names it: its DOI and the file on production, the
    file's address in the records API elsewhere (the sandbox's DOIs do not resolve through doi.org)."""
    if service.production:
        return f"doi:{doi}/{key}"
    return f"{service.api}/records/{record}/files/{quote(key, safe='')}/content"


def staged_lock(lock: bytes, publication: PublicationRecord, own: Mapping[str, str]) -> bytes:
    """The folder's ``lock`` (its bytes) with ``publication``, and each asset named in ``own`` with its location there
    first, its others after it in their order."""
    from mime.saved._folder import lock_text

    held = cast(dict[str, object], json.loads(lock))
    held["publication"] = dict(publication)
    for asset in cast(list[dict[str, object]], held.get("assets", [])):
        mine = own.get(str(asset["name"]))
        if mine is not None:
            asset["locations"] = [mine, *(each for each in cast(list[str], asset["locations"]) if each != mine)]
    return lock_text(held).encode("utf-8")


def publication_of(lock: bytes) -> PublicationRecord | None:
    """The ``publication`` a lock's bytes hold, or ``None``."""
    try:
        held: object = json.loads(lock)
    except ValueError:
        return None
    found = cast(dict[str, object], held).get("publication") if isinstance(held, dict) else None
    return cast(PublicationRecord, found) if isinstance(found, dict) else None


def _linked(source: str, target: str) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def pack_staged(folder: Path, lock: bytes, staging: Path, out: Path) -> Path:
    """Pack ``folder`` with ``lock`` in place of its own into ``out``, through a copy of the folder in ``staging``
    (the folder's own ``sharing.yaml`` left out, as ``pack`` leaves it out). A refusal names the folder, never the
    copy."""
    from mime.push._sharing import SHARING_FILE
    from mime.saved import pack
    from mime.saved._folder import LOCK_FILE

    copy = staging / COPY
    shutil.rmtree(copy, ignore_errors=True)
    shutil.copytree(folder, copy, symlinks=True, copy_function=_linked,
                    ignore=lambda directory, names: [SHARING_FILE] if Path(directory) == folder else [])
    target = copy / LOCK_FILE
    target.unlink(missing_ok=True)  # a link to the folder's own lock: never written through
    target.write_bytes(lock)
    try:
        return pack(copy, out=out)
    except BuildRefused as refused:
        raise BuildRefused([replace(each, message=each.message.replace(str(copy), str(folder)),
                                    action=each.action.replace(str(copy), str(folder)))
                            for each in refused.refusals]) from None
    finally:
        shutil.rmtree(copy, ignore_errors=True)


def rewrite_lock(folder: Path, before: bytes, staged: bytes) -> Refusal | None:
    """Replace the folder's lock by the staged one, atomically, when it is still ``before`` (its bytes when the bundle
    was packed); otherwise leave it and say why (``push.lock-changed``)."""
    from mime.saved._folder import LOCK_FILE

    path = folder / LOCK_FILE
    try:
        now = path.read_bytes()
    except OSError:
        now = b""
    if now == staged:
        return None
    if now != before:
        return Refusal("push.lock-changed", RefusalClass.USER,
                       f"the folder's lock {path} changed after the published bundle was packed from it, so push "
                       f"left it as it is, and the folder does not name the record it was published as",
                       "the published bundle's lock is in push's record; to make the folder the published one, "
                       "`mime pull` the record into a new folder")
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}")
    try:
        temporary.write_bytes(staged)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return None


__all__ = ["COPY", "own_location", "pack_staged", "publication_of", "rewrite_lock", "staged_lock"]
