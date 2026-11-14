"""``mime push <folder> --setup``: ``mime.toml`` made by a walk through its fields, one at a time, in the order ``--init``
writes them, each with one line on what it is for; and ``mime defaults edit``, the same walk over a person's defaults.

**Enter skips** a field. **A field that holds a value already** (what the file holds, what ``sharing.yaml`` held where
``mime.toml`` is written from it, or the person's default where the file is new) is asked as ``use <value>? [Y/n]``:
Enter or ``y`` keeps it, ``n`` asks for another (and Enter then leaves it empty, but for the title and the licence, which
keep theirs). **A value that is not valid is said and asked again**, by the checks push holds ``[record]`` to (an ORCID
iD by its check digit, a ROR identifier by its form, a contributor's type and a relation by DataCite's lists, the
licence to the one the experiment declares). The fields written in the records API's own shapes (``fields``) are not
asked: one line says so, and they are kept as the file holds them.

**At the end the file is written**: a new one as ``--init`` writes one, each value in its place under its explanation;
an existing one edited key by key (:mod:`mime.push._toml`), every other line and comment kept, in its own line endings,
and the keys it changed said, never replaced: a link is edited through, its target in place, and a read-only file (or
one in a read-only folder) is refused before anything is asked (``manifest.write``). A folder holding ``sharing.yaml``
alone gets ``mime.toml`` written from it; ``sharing.yaml`` is left for the person to delete, since MIME never deletes a
person's file. What push would refuse in it as it stands (no creator named) is said, not refused: the walk may be left
part done. Then the person is asked whether to save what they set of the keys a person repeats from record to record as
their defaults (:mod:`mime.push._defaults`), ``[y/N]``.

It asks in a terminal only (``push.setup`` otherwise, which names ``--init``; ``defaults.no-terminal`` for ``mime defaults
edit``), and through the questions it is given: the command line's display, on the terminal the person answers at (its
controlling terminal where standard output and standard error both go to files).
"""

from __future__ import annotations

import datetime
import os
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

from mime._said import DONE, QUESTION_CONFIRM, Said
from mime._shell import quoted
from mime._terminal import plain
from mime.build import BuildRefused, Refusal, RefusalClass
from mime.push._defaults import (
    DEFAULT_KEYS,
    LICENCE_KEYS,
    namespace_problem,
    read_all,
    write_defaults,
)
from mime.push._fields import ACCESS_LEVELS, access_problems
from mime.push._manifest import (
    BUNDLE,
    OLD_FILE,
    folder_kind,
    kind_of,
    manifest_problems,
    manifest_text,
    old_values,
    present,
    read_toml,
)
from mime.push._sharing import (
    CONTRIBUTOR_TYPES,
    DEFAULT_LICENCE,
    RELATION_TYPES,
    bare_orcid,
    identifier_problem,
    orcid_problem,
    ror_problem,
)
from mime.push._toml import REMOVE, Unedited, edited, value_text
from mime.terms import licence_problem

type Typed = Callable[[str], str]
type Ask = Callable[[str], bool]
type Say = Callable[[str], None]

#: Each field the walk asks, in the order ``--init`` writes them, with the one line on what it is for.
FIELDS: Mapping[str, str] = {
    "title": "the record's title, as Zenodo shows it and DataCite registers it",
    "description": "a paragraph shown above the card, which is the record's description (empty: the card alone)",
    "licence": "the experiment's own licence, which covers its files: the one the experiment declares",
    "creators": "each person who made it, as the record names them; everyone named must have agreed to be named (an "
                "organisation is written by hand, organisation = \"<its name>\")",
    "contributors": "others credited, each with a DataCite contributor type (DataCurator, Supervisor, ...); push adds "
                    "the rights holder of each third-party file itself",
    "keywords": "searchable words; push adds mime-experiment",
    "related": "papers and other works: a DOI or an https:// address, and DataCite's relation (IsSupplementTo, ...)",
    "also_cite": "works whoever uses this is asked to cite as well (a paper): a DOI or an https:// address, and how it "
                 "is cited",
    "uploader": "the person whose token pushes, and their institution: a declared permission's grantee is one of them",
    "funding": "who funded the work: each funder (its ROR identifier where you know it) and the award's number and title",
    "conference": "a conference it was shown at (Zenodo's meeting field)",
    "fields": "any other field of the record, in the records API's own names and shapes (subjects, dates, languages, "
              "...)",
    "access": "who may read the record and its files once published, where that is not everyone",
}
#: Each field of a person's defaults the walk of ``mime defaults edit`` asks, with what it is for.
DEFAULT_FIELDS: Mapping[str, str] = {
    "creators": "you, and whoever else you name on every record, as a record names them",
    "contributors": "others you credit on every record, each with a DataCite contributor type",
    "keywords": "searchable words for every record",
    "uploader": "the person whose token pushes, and their institution",
    "funding": "who funds your work: each funder and the award's number and title",
    "content": "the licence a new record's content takes where its experiment declares none (and a scaffold's)",
    "code": "the licence a scaffolded plugin's code takes",
    "namespace": "the namespace a scaffold prefills: a block's id and a plugin's entries",
}
#: What a field left empty is written as.
_EMPTY: Mapping[str, object] = {"description": "", "creators": [], "contributors": [], "keywords": [], "related": [],
                                "also_cite": [], "funding": [], "uploader": REMOVE, "conference": REMOVE,
                                "access": REMOVE, "fields": {}}
#: The fields that keep their value where a new one is not given: a record always has a title and a licence.
_KEPT_UNLESS_GIVEN = ("title", "licence")


def _filled(value: object) -> bool:
    return value is not REMOVE and value not in (None, "", [], {})


def _words(value: object) -> str:
    """A value as the question shows it, as ``mime.toml`` writes it."""
    return plain(value_text(value))


class _Walk:
    """The questions of one walk, through ``typed``: each is set apart by the display, the hint in its words."""

    def __init__(self, typed: Typed) -> None:
        self._typed = typed

    def answer(self, prompt: str) -> str:
        return self._typed(Said(prompt, QUESTION_CONFIRM)).strip()

    def keep(self, head: str, value: object) -> bool:
        """``use <value>? [Y/n]``: Enter or y keeps it, n asks for another."""
        prompt = f"{head}\nuse {_words(value)}? [Y/n] "
        while True:
            given = self.answer(prompt).lower()
            if given in ("", "y", "yes"):
                return True
            if given in ("n", "no"):
                return False
            prompt = f"answer y to keep it, or n to give another\nuse {_words(value)}? [Y/n] "

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


def _also_cite(walk: _Walk, head: str) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    while True:
        lead = _lead(head) if not found else ""
        number = len(found) + 1
        identifier = walk.value(f"{lead}work to cite {number}, its DOI or https:// address (Enter: no more): ",
                                identifier_problem)
        if identifier is None:
            return found
        text = walk.value(f"work to cite {number}, how it is cited (Enter: its identifier alone): ")
        found.append({"identifier": identifier, **({"text": text} if text is not None else {})})


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


def _licence(walk: _Walk, head: str, label: str, declared: str | None) -> str | None:
    def licensed(given: str) -> str | None:
        problem = licence_problem(given)
        if problem is not None:
            return plain(problem)
        if declared is not None and given != declared:
            return (f"the experiment declares {declared}, bound to its files: give that, or declare {plain(given)} "
                    f"where the experiment is built, exp.licence(...), and save it again")
        return None
    return walk.value(f"{_lead(head)}{label}: ", licensed)


def _asked(walk: _Walk, key: str, head: str, declared: str | None) -> object:
    """A new value of ``key``, asked part by part; ``None`` where none was given."""
    if key in ("creators", "contributors"):
        return _people(walk, head, key) or None
    if key == "related":
        return _related(walk, head) or None
    if key == "also_cite":
        return _also_cite(walk, head) or None
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
    if key in ("licence", *LICENCE_KEYS):
        return _licence(walk, head, key, declared)
    if key == "namespace":
        return walk.value(f"{_lead(head)}namespace: ", namespace_problem)
    return walk.value(f"{_lead(head)}{key}: ")


def _setup_refusal(folder: Path) -> BuildRefused:
    return BuildRefused([Refusal(
        "push.setup", RefusalClass.USER,
        "--setup asks for mime.toml's fields one at a time, and there is no terminal to ask in; nothing was written",
        f"write mime.toml from what the folder knows and your defaults, `mime push {quoted(str(folder))} --to "
        f"<service> --init`, then edit it by hand")])


def replaced(path: Path, text: str) -> None:
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


def edit_in_place(path: Path, tables: Mapping[str, Mapping[str, object]], again: str) -> None:
    """``path``'s keys set table by table (:func:`mime.push._toml.edited`; :data:`~mime.push._toml.REMOVE` removes one),
    every other line, comment and line ending kept, written whole (through a link, to its target); refused
    (``manifest.write``), the file left byte for byte, where an edit cannot be made safely. ``again`` is the command the
    refusal names to run once the file is put right."""
    text = path.read_bytes().decode("utf-8")
    try:
        for table, updates in tables.items():
            if updates:
                text = edited(text, table, updates)
    except Unedited as why:
        names = ", ".join(key for updates in tables.values() for key in updates)
        raise BuildRefused([Refusal(
            "manifest.write", RefusalClass.USER, f"{path}: {names} cannot be written in place: {why}; nothing was written",
            f"write it as one key a line by hand (`mime check <folder>` checks it), or move it aside and run {again} "
            f"again")]) from None
    replaced(path, text)


def unwritable(path: Path, root: Path) -> BuildRefused | None:
    """Why the existing ``mime.toml`` at ``path`` cannot be edited in place (it, its folder or its link's target not
    writable), said before anything is asked; ``None`` where it can."""
    target = path.resolve()
    if os.access(target, os.W_OK) and os.access(target.parent, os.W_OK | os.X_OK):
        return None
    where = f"{path}" + (f", a link to {target}," if path.is_symlink() else "")
    why = "is read-only" if not os.access(target, os.W_OK) else f"is in a folder that is read-only, {target.parent}"
    return BuildRefused([Refusal(
        "manifest.write", RefusalClass.USER,
        f"{where} {why}, so --setup cannot edit it, and it never replaces it; nothing was asked or written",
        f"make it writable, `chmod u+w {quoted(str(target))}`, then `mime push {quoted(str(root))} --setup` again; or "
        f"edit it by hand")])


def folder_facts(root: Path) -> tuple[str, str, str | None]:
    """The kind of the folder ``root`` (a saved folder's ``bundle``, a study or publishing folder's ``study``), the title
    a new ``mime.toml`` takes (the experiment's name; a study or publishing folder's own name), and the licence its
    experiment declares (a study's base experiment's; a publishing folder's first study's). A folder of neither kind is
    refused as loading it refuses."""
    from mime.results._study_folder import BASE_DIR
    from mime.saved import read_folder
    from mime.saved._studies import study_folders

    kind = folder_kind(root)
    document = read_folder(study_folders(root)[0] / BASE_DIR if kind == "study" else root,
                           absent_referenced=True).document
    return kind or BUNDLE, document["experiment"] if kind != "study" else root.name, document.get("licence")


def _record_and_licence(held: Mapping[str, object], kind: str) -> dict[str, object]:
    """``[record]``'s keys and the kind table's ``licence``, as the walk asks them, from a ``mime.toml`` as read."""
    record = held.get("record")
    found = dict(cast(dict[str, object], record)) if isinstance(record, dict) else {}
    definition = held.get(kind)
    if isinstance(definition, dict) and "licence" in definition:
        found["licence"] = cast(dict[str, object], definition)["licence"]
    return found


def setup(folder: str | os.PathLike[str], *, typed: Typed | None, ask: Ask | None, say: Say,
          service: str | None = None) -> Path:
    """Walk the person through ``mime.toml`` beside the saved or study folder, one field at a time (``typed`` asks each,
    ``ask`` the last question, whether to save their defaults), then write it: a new file as ``--init`` writes one
    (from ``sharing.yaml``'s values where the folder holds it alone), or the existing one edited key by key and said so.
    Without ``typed`` (no terminal) it is refused (``push.setup``). The path of ``mime.toml``."""
    root = Path(os.path.abspath(folder))
    kind, name, declared = folder_facts(root)
    if typed is None or ask is None:
        raise _setup_refusal(root)
    path = present(root, new=True)
    old = root / OLD_FILE
    before: str | None = None
    held: dict[str, object] = {}
    if path.is_file():
        refused = unwritable(path, root)
        if refused is not None:
            raise refused
        # Read as bytes: its own line endings kept (a file of CRLF lines is edited in CRLF), every other byte as it was.
        before, loaded = read_toml(path)
        kind = kind_of(loaded, path, root)
        held = _record_and_licence(loaded, kind)
    defaults = read_all(say)
    fallback = defaults.licences.get("content", DEFAULT_LICENCE)
    migrating = before is None and old.is_file()
    current: dict[str, object] = held if before is not None else old_values(root) if migrating else {
        "title": name, "description": "", "licence": declared or fallback, **defaults.values}
    walk = _Walk(typed)
    say(f"Setting up {path} for {plain(name)!r}, one field at a time: Enter skips a field, or keeps the value shown.")
    if migrating:
        say(f"It is written from {old}, which MIME no longer reads: each value it holds is offered to keep.")
    chosen: dict[str, object] = {}
    for key, why in FIELDS.items():
        head = f"{key}: {why}"
        value = current.get(key)
        if key == "fields":
            say(f"{head}: written by hand in mime.toml" + (f", which holds {_words(value)}" if _filled(value) else ""))
            if migrating and _filled(value):
                chosen[key] = value
            continue
        if _filled(value) and walk.keep(head, value):
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
        text = manifest_text(kind, name, declared, {key: value for key, value in chosen.items() if _filled(value)},
                             fallback=fallback)
        path.write_text(text, encoding="utf-8", newline="\n")
        say(Said(f"wrote {path}", DONE))
        if migrating:
            say(f"{old} is no longer read: delete it once you have checked {path} (MIME never deletes your file)")
    else:
        changed = {key: value for key, value in chosen.items() if value != held.get(key)
                   and (_filled(value) or _filled(held.get(key)))}
        if changed:
            edit_in_place(path, {kind: {key: changed[key] for key in ("licence",) if key in changed},
                                 "record": {key: value for key, value in changed.items() if key != "licence"}},
                          f"`mime push {quoted(str(root))} --setup`")
            through = f" (through the link, in {path.resolve()})" if path.is_symlink() else ""
            say(Said(f"edited {path}{through}: {', '.join(changed)}, each written in place; every other line and "
                     f"comment kept", DONE))
        else:
            say(f"kept {path} as it was: every field is as it held it")
    _, problems = manifest_problems(root, old_ignored=migrating)
    for each in problems:
        say(f"! push would refuse it as it stands: {each.refusal.message}")
    _offer_defaults({key: chosen[key] for key in DEFAULT_KEYS if _filled(chosen.get(key))}, defaults.values, ask, say)
    if kind == BUNDLE:
        say(f"next: `mime push {quoted(str(folder))} --to {service or '<service>'}` shows the plan, then makes the draft"
            + ("" if service else " (zenodo-sandbox to try it, zenodo to share it)"))
    else:
        say(f"next: `mime check {quoted(str(folder))}` checks it; pushing a study comes with a later release")
    return path


def _offer_defaults(values: dict[str, object], defaults: Mapping[str, object], ask: Ask, say: Say) -> None:
    """Ask whether to save ``values`` as the person's defaults, where they hold any and are not those already."""
    if not values or values == {key: defaults[key] for key in DEFAULT_KEYS if key in defaults}:
        return
    if ask(Said(f"Save these as your defaults ({', '.join(values)}), which --init and --setup fill a new mime.toml "
                f"from?", QUESTION_CONFIRM)):
        saved = write_defaults(values)
        say(Said(f"saved your defaults in {saved}: `mime defaults` shows them", DONE))


def edit_defaults(*, typed: Typed | None, say: Say) -> Path:
    """``mime defaults edit``: walk the person's defaults a field at a time, as ``--setup`` walks a record, and write
    them whole or not at all. Without ``typed`` (no terminal) it is refused (``defaults.no-terminal``)."""
    if typed is None:
        raise BuildRefused([Refusal(
            "defaults.no-terminal", RefusalClass.USER,
            "mime defaults edit asks for your defaults one at a time, and there is no terminal to ask in; nothing was "
            "written", "run it in a terminal, or write the file `mime defaults` names by hand")])
    found = read_all(say)
    current: dict[str, object] = {**found.values, **found.licences,
                                  **({"namespace": found.namespace} if found.namespace else {})}
    walk = _Walk(typed)
    say("Your defaults, one field at a time: Enter skips a field, or keeps the value shown.")
    chosen: dict[str, object] = {}
    for key, why in DEFAULT_FIELDS.items():
        head = f"{key}: {why}"
        value = current.get(key)
        if _filled(value) and walk.keep(head, value):
            chosen[key] = value
            continue
        given = _asked(walk, key, "" if _filled(value) else head, None)
        if given is not None:
            chosen[key] = given
    path = write_defaults({key: chosen[key] for key in DEFAULT_KEYS if key in chosen},
                          licences={key: cast(str, chosen[key]) for key in LICENCE_KEYS if key in chosen},
                          namespace=cast(str | None, chosen.get("namespace")), whole=True)
    say(Said(f"saved your defaults in {path}: `mime defaults` shows them", DONE))
    return path


__all__ = ["DEFAULT_FIELDS", "FIELDS", "edit_defaults", "edit_in_place", "folder_facts", "replaced", "setup",
           "unwritable"]
