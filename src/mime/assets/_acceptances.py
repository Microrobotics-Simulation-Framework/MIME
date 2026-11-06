"""The licences a person has accepted, kept on this machine so that each is asked about once.

A download under a licence with conditions beyond credit asks for them to be accepted (:mod:`mime.assets._gates`), and a
yes is written here: the licence's id, its title, its link, its conditions and the date. The file is JSON, at
``MIME_LICENCE_ACCEPTANCES`` when that is set, else ``licences.json`` in the user's config directory (``platformdirs``),
beside the credentials and the locations file. It never travels with an experiment, a bundle or a lock: a person's yes is
theirs. An acceptance is of a licence by its **id and version**, which is its title (a licence's version is in its
title): when the title changes, the licence is asked about again. The link and the condition lines are kept as they were
shown, for the person who lists them, but they are not what an acceptance is keyed by: the same licence read from a
manifest and from a saved lock (whose form has no key for the lines) is one acceptance, asked about once.

``mime assets licences`` lists the acceptances and ``mime assets licences --revoke <id>`` removes one. A file that cannot
be read is refused (``asset.licences-file``), never read as no acceptance and written over.
"""

from __future__ import annotations

import datetime
import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from mime._terminal import plain
from mime.build import BuildRefused, Refusal, RefusalClass

ACCEPTANCES_VARIABLE = "MIME_LICENCE_ACCEPTANCES"
ACCEPTANCES_FILE = "licences.json"
ACCEPTANCES_FORMAT = "mime.licence_acceptances/1"


@dataclass(frozen=True)
class Licence:
    """A licence a download needs accepted: its id, its title and link, and its conditions beyond credit in short lines.
    An acceptance is of its id and its title (its version): when either changes, the licence is asked about again."""

    id: str
    title: str
    url: str
    conditions: tuple[str, ...]


@dataclass(frozen=True)
class Acceptance:
    """A licence accepted, and the date (``YYYY-MM-DD``) it was."""

    licence: Licence
    date: str


def acceptances_path() -> Path:
    """The file: ``MIME_LICENCE_ACCEPTANCES`` when set, else ``licences.json`` in the user's config directory."""
    chosen = os.environ.get(ACCEPTANCES_VARIABLE)
    if chosen:
        return Path(chosen)
    import platformdirs

    return Path(platformdirs.user_config_dir("mime")) / ACCEPTANCES_FILE


def _refused(path: Path, problem: str) -> BuildRefused:
    return BuildRefused([Refusal(
        "asset.licences-file", RefusalClass.USER, f"the licence acceptances file {path}: {problem}",
        f"correct it, or delete it to be asked about each licence again (it holds only your own acceptances, which "
        f"`mime assets licences` lists); {ACCEPTANCES_VARIABLE} names another file")])


def read_acceptances() -> list[Acceptance]:
    """Every acceptance on this machine, in the order given; none where there is no file."""
    path = acceptances_path()
    if not path.is_file():
        return []
    try:
        held: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise _refused(path, f"cannot be read as JSON ({error})") from None
    listed = cast("dict[str, object]", held).get("accepted") if isinstance(held, dict) else None
    if not isinstance(held, dict) or cast("dict[str, object]", held).get("format") != ACCEPTANCES_FORMAT \
            or not isinstance(listed, list):
        raise _refused(path, f"it is not a {ACCEPTANCES_FORMAT} file: an object with that format and a list "
                             f"\"accepted\"")
    found: list[Acceptance] = []
    for entry in cast("list[object]", listed):
        row = cast("dict[str, object]", entry) if isinstance(entry, dict) else {}
        lines = row.get("conditions")
        if (set(row) != {"id", "title", "url", "conditions", "date"}
                or not all(isinstance(row[key], str) for key in ("id", "title", "url", "date"))
                or not isinstance(lines, list) or not all(isinstance(each, str) for each in cast("list[object]", lines))):
            raise _refused(path, "each acceptance is an object of id, title, url, conditions (a list of texts) and "
                                 "date, as texts")
        found.append(Acceptance(Licence(str(row["id"]), str(row["title"]), str(row["url"]),
                                        tuple(str(each) for each in cast("list[object]", lines))), str(row["date"])))
    return found


def _write(held: Sequence[Acceptance]) -> None:
    path = acceptances_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps({"format": ACCEPTANCES_FORMAT,
                       "accepted": [{"id": each.licence.id, "title": each.licence.title, "url": each.licence.url,
                                     "conditions": list(each.licence.conditions), "date": each.date}
                                    for each in held]}, indent=2) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def accepted(licence: Licence) -> bool:
    """Whether this licence, by its id and its title (its version), was accepted on this machine."""
    return any(each.licence.id == licence.id and each.licence.title == licence.title for each in read_acceptances())


def remember(licences: Sequence[Licence]) -> None:
    """Write the acceptance of each licence, today's date; one of the same id that was accepted before (another title,
    link or conditions) is replaced."""
    today = datetime.date.today().isoformat()
    ids = {each.id for each in licences}
    held = [each for each in read_acceptances() if each.licence.id not in ids]
    _write([*held, *(Acceptance(each, today) for each in licences)])


def revoke(identifier: str) -> int:
    """Remove the acceptance of the licence ``identifier``; how many were removed (0 where none was kept)."""
    held = read_acceptances()
    kept = [each for each in held if each.licence.id != identifier]
    if len(kept) != len(held):
        _write(kept)
    return len(held) - len(kept)


def acceptances_text() -> str:
    """The acceptances as ``mime assets licences`` prints them, one line each; a line saying so where there are none."""
    held = read_acceptances()
    if not held:
        return f"no licence accepted on this machine ({plain(str(acceptances_path()))})"
    return "\n".join(f"{plain(each.licence.id)}  {plain(each.licence.title)}  {plain(each.licence.url)}  "
                     f"accepted {plain(each.date)}" for each in held)


__all__ = ["ACCEPTANCES_FILE", "ACCEPTANCES_FORMAT", "ACCEPTANCES_VARIABLE", "Acceptance", "Licence", "accepted",
           "acceptances_path", "acceptances_text", "read_acceptances", "remember", "revoke"]
