"""``mime push <folder> --setup``: ``sharing.yaml`` made by a walk through its fields, one at a time, in the order
``--init`` writes them, each with one line on what it is for.

**Enter skips** a field. **A field that holds a value already** (what the file holds, or the person's default where the
file is new) is asked as ``use <value>? [Y/n]``: Enter or ``y`` keeps it, ``n`` asks for another (and Enter then leaves
it empty, but for the title and the licence, which keep theirs). **A value that is not valid is said and asked again**,
by the checks push holds ``sharing.yaml`` to (an ORCID iD by its check digit, a ROR identifier by its form, a
contributor's type and a relation by DataCite's lists, the licence to the one the experiment declares). The fields
written in the records API's own shapes (``fields:``) are not asked: one line says so, and they are kept as the file
holds them.

**At the end the file is written**: a new one as ``--init`` writes it, each value in its place under its explanation;
an existing one edited key by key, every other line and comment kept, in its own line endings (the comment that ends a
changed key's own line kept on its new one), and the keys it changed said, never replaced: a link is edited through,
its target in place, and a read-only file (or one in a read-only folder) is refused before anything is asked. What
push would refuse in it as it stands (no creator named) is said, not refused: the walk may be left part done. Then the
person is asked whether to save what they set of the keys a person repeats from record to record as their defaults
(:mod:`mime.push._defaults`), ``[y/N]``.

It asks in a terminal only (``push.setup`` otherwise, which names ``--init``), and through the questions it is given:
the command line's display, on the terminal the person answers at (its controlling terminal where standard output and
standard error both go to files).
"""

from __future__ import annotations

import datetime
import os
import re
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

import yaml

from mime._said import DONE, QUESTION_CONFIRM, Said
from mime._shell import quoted
from mime._terminal import plain
from mime.build import BuildRefused, Refusal, RefusalClass
from mime.push._defaults import read_defaults, write_defaults
from mime.push._fields import ACCESS_LEVELS, access_problems
from mime.push._sharing import (
    CONTRIBUTOR_TYPES,
    DEFAULT_KEYS,
    DEFAULT_LICENCE,
    RELATION_TYPES,
    bare_orcid,
    identifier_problem,
    orcid_problem,
    read_sharing,
    ror_problem,
    scaffold,
    sharing_path,
    value_line,
)
from mime.terms import licence_problem

type Typed = Callable[[str], str]
type Ask = Callable[[str], bool]
type Say = Callable[[str], None]

#: Each field the walk asks, in the order ``--init`` writes them, with the one line on what it is for.
FIELDS: Mapping[str, str] = {
    "title": "the record's title, as Zenodo shows it and DataCite registers it",
    "description": "a paragraph shown above the card, which is the record's description (empty: the card alone)",
    "licence": "the experiment's own licence, which covers its files: the one the experiment declares",
    "creators": "each person who made the experiment, as the record names them; everyone named must have agreed to "
                "be named (an organisation is written by hand, organisation: <its name>)",
    "contributors": "others credited, each with a DataCite contributor type (DataCurator, Supervisor, ...); push adds "
                    "the rights holder of each third-party file itself",
    "keywords": "searchable words; push adds mime-experiment",
    "related": "papers and other works: a DOI or an https:// address, and DataCite's relation (IsSupplementTo, ...)",
    "uploader": "the person whose token pushes, and their institution: a declared permission's grantee is one of them",
    "funding": "who funded the work: each funder (its ROR identifier where you know it) and the award's number and title",
    "conference": "a conference the experiment was shown at (Zenodo's meeting field)",
    "fields": "any other field of the record, in the records API's own names and shapes (subjects, dates, languages, "
              "...)",
    "access": "who may read the record and its files once published, where that is not everyone",
}
#: What a field left empty is written as.
_EMPTY: Mapping[str, object] = {"description": "", "creators": [], "contributors": [], "keywords": [], "related": [],
                                "funding": [], "uploader": None, "conference": None, "access": None, "fields": {}}
#: The fields that keep their value where a new one is not given: a record always has a title and a licence.
_KEPT_UNLESS_GIVEN = ("title", "licence")


def _filled(value: object) -> bool:
    return value not in (None, "", [], {})


def _words(key: str, value: object) -> str:
    """A value as the question shows it: text in quotes, anything else as ``sharing.yaml`` writes it."""
    if isinstance(value, str):
        return f'"{plain(value)}"'
    line = value_line(key, value)
    return plain(line.partition(": ")[2]) if "\n" not in line else plain(repr(value))


class _Walk:
    """The questions of one walk, through ``typed``: each is set apart by the display, the hint in its words."""

    def __init__(self, typed: Typed) -> None:
        self._typed = typed

    def answer(self, prompt: str) -> str:
        return self._typed(Said(prompt, QUESTION_CONFIRM)).strip()

    def keep(self, head: str, key: str, value: object) -> bool:
        """``use <value>? [Y/n]``: Enter or y keeps it, n asks for another."""
        prompt = f"{head}\nuse {_words(key, value)}? [Y/n] "
        while True:
            given = self.answer(prompt).lower()
            if given in ("", "y", "yes"):
                return True
            if given in ("n", "no"):
                return False
            prompt = f"answer y to keep it, or n to give another\nuse {_words(key, value)}? [Y/n] "

    def value(self, prompt: str, check: Callable[[str], str | None] | None = None, *,
              needed: str | None = None) -> str | None:
        """A value typed at ``prompt``; ``None`` for Enter, unless ``needed`` says why one is needed; one ``check``
        refuses is said and asked again."""
        asked = prompt
        while True:
            given = self.answer(asked)
            if not given and needed is None:
                return None
            problem = needed if not given else check(given) if check is not None else None
            if problem is None:
                return given
            asked = f"{problem}\n{prompt.rpartition(chr(10))[2]}"


def _lead(head: str) -> str:
    """The field's line on what it is for, above the first question asked of it (none where it was shown with its
    value)."""
    return f"{head}\n" if head else ""


def _one_of(words: tuple[str, ...], what: str) -> Callable[[str], str | None]:
    def check(given: str) -> str | None:
        return None if given in words else f"{plain(given)} is not {what} ({', '.join(words)})"
    return check


def _people(walk: _Walk, head: str, key: str) -> list[dict[str, str]]:
    """Each person of ``creators`` or ``contributors``, asked part by part; Enter at a family name ends the list."""
    noun = "creator" if key == "creators" else "contributor"
    found: list[dict[str, str]] = []
    while True:
        number = len(found) + 1
        lead = _lead(head) if not found else ""
        family = walk.value(f"{lead}{noun} {number}, family name (Enter: no more {key}): ")
        if family is None:
            return found
        person = {"family": family}
        given = walk.value(f"{noun} {number}, given name (Enter: none): ")
        orcid = walk.value(f"{noun} {number}, ORCID iD (Enter: none): ", orcid_problem)
        affiliation = walk.value(f"{noun} {number}, affiliation (Enter: none): ")
        ror = walk.value(f"{noun} {number}, the affiliation's ROR identifier (Enter: none): ", ror_problem) \
            if affiliation is not None else None
        role = walk.value(f"{noun} {number}, DataCite contributor type (DataCurator, Supervisor, ProjectLeader, ...): ",
                          _one_of(CONTRIBUTOR_TYPES, "one of DataCite's contributor types"),
                          needed="a contributor is credited with a DataCite contributor type") \
            if key == "contributors" else None
        for part, value in (("given", given), ("orcid", bare_orcid(orcid) if orcid else None),
                            ("affiliation", affiliation), ("ror", ror), ("type", role)):
            if value is not None:
                person[part] = value
        found.append(person)


def _related(walk: _Walk, head: str) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    while True:
        lead = _lead(head) if not found else ""
        identifier = walk.value(f"{lead}related work {len(found) + 1}, its DOI or https:// address (Enter: no more): ",
                                identifier_problem)
        if identifier is None:
            return found
        relation = walk.value(f"related work {len(found) + 1}, its relation (IsSupplementTo, References, Cites, "
                              f"IsDerivedFrom, ...): ", _one_of(RELATION_TYPES, "one of DataCite's relation types"),
                              needed="a related work is stated with its relation to this one")
        found.append({"identifier": identifier, "relation": str(relation)})


def _funding(walk: _Walk, head: str) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    while True:
        lead = _lead(head) if not found else ""
        number = len(found) + 1
        funder = walk.value(f"{lead}funder {number}, its name (Enter: no more): ")
        if funder is None:
            return found
        entry = {"funder": funder}
        ror = walk.value(f"funder {number}, its ROR identifier (Enter: none): ", ror_problem)
        award = walk.value(f"funder {number}, the award's number (Enter: none): ")
        title = walk.value(f"funder {number}, the award's title (Enter: none): ") if award is not None else None
        url = walk.value(f"funder {number}, the award's https:// address (Enter: none): ",
                         lambda given: None if given.startswith("https://") else "it is not an https:// address") \
            if award is not None else None
        for part, value in (("ror", ror), ("award", award), ("title", title), ("url", url)):
            if value is not None:
                entry[part] = value
        found.append(entry)


def _conference(walk: _Walk, head: str) -> dict[str, str] | None:
    title = walk.value(f"{_lead(head)}conference, its name (Enter: none): ")
    if title is None:
        return None
    meeting = {"title": title}
    for part, what in (("acronym", "its acronym"), ("dates", "its dates"), ("place", "its place")):
        value = walk.value(f"conference, {what} (Enter: none): ")
        if value is not None:
            meeting[part] = value
    url = walk.value("conference, its https:// address (Enter: none): ",
                     lambda given: None if given.startswith("https://") else "it is not an https:// address")
    if url is not None:
        meeting["url"] = url
    return meeting


def _access(walk: _Walk, head: str) -> dict[str, object] | None:
    levels = _one_of(ACCESS_LEVELS, "public or restricted")
    record = walk.value(f"{_lead(head)}the record, public or restricted (Enter: public): ", levels)
    files = walk.value("its files, public or restricted (Enter: public): ", levels)
    found: dict[str, object] = {key: value for key, value in (("record", record), ("files", files))
                                if value is not None and value != "public"}
    if not found:
        return None

    def day(given: str) -> str | None:
        problems = access_problems({**found, "embargo": {"until": given}})[1]
        if problems:
            return "; ".join(problem.partition(": ")[2] for problem in problems)
        try:
            datetime.date.fromisoformat(given)
        except ValueError:
            return f"{plain(given)} is not a day of the calendar (2030-01-01 is)"
        return None

    until = walk.value("an embargo, the day it ends, 2030-01-01 (Enter: none): ", day)
    if until is not None:
        embargo = {"until": until}
        reason = walk.value("the embargo's reason (Enter: none): ")
        if reason is not None:
            embargo["reason"] = reason
        found["embargo"] = embargo
    return found


def _asked(walk: _Walk, key: str, head: str, declared: str | None) -> object:
    """A new value of ``key``, asked part by part; ``None`` where none was given."""
    if key in ("creators", "contributors"):
        return _people(walk, head, key) or None
    if key == "related":
        return _related(walk, head) or None
    if key == "funding":
        return _funding(walk, head) or None
    if key == "conference":
        return _conference(walk, head)
    if key == "access":
        return _access(walk, head)
    if key == "keywords":
        given = walk.value(f"{_lead(head)}keywords, separated by commas (Enter: none): ")
        words = [word.strip() for word in (given or "").split(",") if word.strip()]
        return words or None
    if key == "uploader":
        name = walk.value(f"{_lead(head)}uploader, your name as \"family name, given name\" (Enter: none): ")
        if name is None:
            return None
        institution = walk.value("uploader, your institution (Enter: none): ")
        return {"name": name, **({"institution": institution} if institution is not None else {})}
    if key == "licence":
        def licensed(given: str) -> str | None:
            problem = licence_problem(given)
            if problem is not None:
                return plain(problem)
            if declared is not None and given != declared:
                return (f"the experiment declares {declared}, bound to its files: give that, or declare {plain(given)} "
                        f"where the experiment is built, exp.licence(...), and save it again")
            return None
        return walk.value(f"{_lead(head)}licence: ", licensed)
    return walk.value(f"{_lead(head)}{key}: ")


def _setup_refusal(folder: Path) -> BuildRefused:
    return BuildRefused([Refusal(
        "push.setup", RefusalClass.USER,
        "--setup asks for sharing.yaml's fields one at a time, and there is no terminal to ask in; nothing was written",
        f"write sharing.yaml from what the folder knows and your defaults, `mime push {quoted(str(folder))} --to "
        f"<service> --init`, then edit it by hand")])


def _replaced(path: Path, text: str) -> None:
    """``path`` holding ``text``, whole or not at all, its mode kept: the file itself, a link's target where ``path`` is
    a link (the link left as it is)."""
    target = path.resolve(strict=True)
    temporary = target.with_name(f".{target.name}.setup")
    try:
        temporary.write_bytes(text.encode("utf-8"))
        shutil.copymode(target, temporary)
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _reads(text: str) -> object:
    """What ``text`` reads as in YAML; where it does not read, an object equal to nothing else."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return object()


def _trailing_comment(text: str, lines: list[str], at: int) -> str | None:
    """The comment that ends line ``at`` of ``text`` (``lines``), with the space before it: the first ``#`` after a
    space whose cut leaves ``text`` reading as it did (a ``#`` inside a quoted value is no comment); ``None`` where
    none."""
    line, ending = lines[at].removesuffix("\r"), "\r" if lines[at].endswith("\r") else ""
    held = _reads(text)
    for found in re.finditer(r"[ \t]+#", line):
        if _reads("\n".join([*lines[:at], line[:found.start()] + ending, *lines[at + 1:]])) == held:
            return line[found.start():]
    return None


def _edited(before: str, changed: Mapping[str, object]) -> str:
    """``before`` with each key of ``changed`` written in place (:func:`~mime.push._edits.written_text`, which P2b's
    write-back shares and which this leaves as it is), the comment that ended a rewritten key's own line put back at
    the end of its new first line, where the file reads the same with it as without (never inside a quoted value)."""
    from mime.push._edits import key_line, written_text

    after = written_text(before, changed)
    old, new = before.split("\n"), after.split("\n")
    for key in changed:
        at, to = key_line(old, key), key_line(new, key)
        comment = _trailing_comment(before, old, at) if at is not None else None
        if comment is None or to is None:
            continue
        line, ending = new[to].removesuffix("\r"), "\r" if new[to].endswith("\r") else ""
        commented = [*new[:to], line + comment + ending, *new[to + 1:]]
        if _reads("\n".join(commented)) == _reads(after):
            new = commented
    return "\n".join(new)


def _unwritable(path: Path, root: Path) -> BuildRefused | None:
    """Why the existing ``sharing.yaml`` at ``path`` cannot be edited in place (it, its folder or its link's target not
    writable), said before anything is asked; ``None`` where it can."""
    target = path.resolve()
    if os.access(target, os.W_OK) and os.access(target.parent, os.W_OK | os.X_OK):
        return None
    where = f"{path}" + (f", a link to {target}," if path.is_symlink() else "")
    why = "is read-only" if not os.access(target, os.W_OK) else f"is in a folder that is read-only, {target.parent}"
    return BuildRefused([Refusal(
        "push.sharing", RefusalClass.USER,
        f"{where} {why}, so --setup cannot edit it, and it never replaces it; nothing was asked or written",
        f"make it writable, `chmod u+w {quoted(str(target))}`, then `mime push {quoted(str(root))} --setup` again; or "
        f"edit it by hand")])


def setup(folder: str | os.PathLike[str], *, typed: Typed | None, ask: Ask | None, say: Say,
          service: str | None = None) -> Path:
    """Walk the person through ``sharing.yaml`` beside the saved folder, one field at a time (``typed`` asks each,
    ``ask`` the last question, whether to save their defaults), then write it: a new file as ``--init`` writes one, or the
    existing one edited key by key and said so. Without ``typed`` (no terminal) it is refused (``push.setup``). The path
    of ``sharing.yaml``."""
    from mime.saved import read_folder

    root = Path(os.path.abspath(folder))
    document = read_folder(root, absent_referenced=True).document
    if typed is None or ask is None:
        raise _setup_refusal(root)
    path = sharing_path(root)
    name, declared = document["experiment"], document.get("licence")
    if path.is_file():
        refused = _unwritable(path, root)
        if refused is not None:
            raise refused
    # Read as bytes: its own line endings kept (a file of CRLF lines is edited in CRLF), every other byte as it was.
    before = path.read_bytes().decode("utf-8") if path.is_file() else None
    try:
        loaded: object = yaml.safe_load(before) if before is not None else None
    except yaml.YAMLError as error:
        raise BuildRefused([Refusal(
            "push.sharing", RefusalClass.USER, f"{path} is not YAML that reads ({type(error).__name__}), so --setup "
                                               f"cannot edit it", f"correct it by hand, or move it aside and run `mime "
                                                                   f"push {quoted(str(root))} --setup` again")]) from None
    held = cast(dict[str, object], loaded) if isinstance(loaded, dict) else {}
    defaults = read_defaults()
    current: dict[str, object] = dict(held) if before is not None else {
        "title": name, "description": "", "licence": declared or DEFAULT_LICENCE, **defaults}
    walk = _Walk(typed)
    say(f"Setting up {path} for {plain(name)!r}, one field at a time: Enter skips a field, or keeps the value shown.")
    chosen: dict[str, object] = {}
    for key, why in FIELDS.items():
        head = f"{key}: {why}"
        value = current.get(key)
        if key == "fields":
            say(f"{head}: written by hand in sharing.yaml" + (f", which holds {_words(key, value)}"
                                                             if _filled(value) else ""))
            continue
        if _filled(value) and walk.keep(head, key, value):
            chosen[key] = value
            continue
        given = _asked(walk, key, "" if _filled(value) else head, declared)
        if given is not None:
            chosen[key] = given
        elif _filled(value) and key in _KEPT_UNLESS_GIVEN:
            chosen[key] = value
        elif _filled(value):
            chosen[key] = _EMPTY[key]
    if before is None:
        text = scaffold(name, declared, {key: value for key, value in chosen.items() if _filled(value)})
        path.write_text(text, encoding="utf-8", newline="\n")
        say(Said(f"wrote {path}", DONE))
    else:
        changed = {key: value for key, value in chosen.items() if value != held.get(key)
                   and (_filled(value) or _filled(held.get(key)))}
        if changed:
            _replaced(path, _edited(before, changed))
            through = f" (through the link, in {path.resolve()})" if path.is_symlink() else ""
            say(Said(f"edited {path}{through}: {', '.join(changed)}, each written in place; every other line and "
                     f"comment kept", DONE))
        else:
            say(f"kept {path} as it was: every field is as it held it")
    try:
        read_sharing(root)
    except BuildRefused as refused:
        for each in refused.refusals:
            say(f"! push would refuse it as it stands: {each.message}")
    _offer_defaults({key: chosen[key] for key in DEFAULT_KEYS if _filled(chosen.get(key))}, defaults, ask, say)
    say(f"next: `mime push {quoted(str(folder))} --to {service or '<service>'}` shows the plan, then makes the draft"
        + ("" if service else " (zenodo-sandbox to try it, zenodo to share it)"))
    return path


def _offer_defaults(values: dict[str, object], defaults: Mapping[str, object], ask: Ask, say: Say) -> None:
    """Ask whether to save ``values`` as the person's defaults, where they hold any and are not those already."""
    if not values or values == {key: defaults[key] for key in DEFAULT_KEYS if key in defaults}:
        return
    if ask(Said(f"Save these as your defaults ({', '.join(values)}), which --init and --setup fill a new sharing.yaml "
                f"from?", QUESTION_CONFIRM)):
        saved = write_defaults(values)
        say(Said(f"saved your defaults in {saved}: `mime push --defaults` shows them", DONE))


__all__ = ["FIELDS", "setup"]
