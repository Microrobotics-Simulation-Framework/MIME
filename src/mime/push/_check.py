"""``mime check <folder>``: a shared folder's ``mime.toml`` checked offline, on the profile workflow.

Every key's check is run (:mod:`mime.push._manifest`), with the licence its kind table states against the one its
experiment declares; nothing is built, packed or sent. Each item has a class, and the profile says what each class is, as
the profile rules say what a build does with a finding (:data:`RULES`):

- **a defect** (a key unknown or of another type, an ORCID iD's check digit, a licence unlike the experiment's) is a
  finding in every profile, since no profile relaxes an input error;
- **what push still needs** (no creator named, no title, an experiment that declares no licence, a declared permission
  with no uploader named to hold its grantee to) is a to-do in
  exploratory and a finding in research, which refuses what exploratory waives;
- **what a reviewer would ask** (a person among the creators with no ORCID iD, no description, no keyword of the author's
  own) is listed by research and never a finding, as research lists a value with no source.

``strict`` is refused by name (``profile.reserved``), as a build is. The profile is the experiment's own (a study's base
experiment's) unless one is given. Each item comes with the line to write, as ``mime pending`` gives the line to paste.
A finding exits 1; to-dos and listed items alone exit 0; a folder that cannot be checked is refused (exit 2)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from mime._terminal import plain
from mime.build import BuildRefused, Refusal, RefusalClass
from mime.push._manifest import BUNDLE, MANIFEST_FILE, folder_kind, manifest_problems, read_toml
from mime.push._sharing import LINES
from mime.validity import Profile, profile_of

DEFECT, NEEDED, ASKED = "defect", "needed", "asked"
FINDING, TO_DO, LISTED = "finding", "to do", "listed"
#: What each profile makes of each class of item; a class with no row in a profile is not shown in it.
RULES: Mapping[tuple[Profile, str], str] = {
    (Profile.EXPLORATORY, DEFECT): FINDING, (Profile.RESEARCH, DEFECT): FINDING,
    (Profile.EXPLORATORY, NEEDED): TO_DO, (Profile.RESEARCH, NEEDED): FINDING,
    (Profile.RESEARCH, ASKED): LISTED,
}
#: How each verdict's items are introduced.
_HEADS = {FINDING: "findings: what must change, which push refuses as it stands",
          TO_DO: "to do: what push still needs, which exploratory lists and research makes a finding",
          LISTED: "listed by research, never a finding: what a reviewer would ask"}


@dataclass(frozen=True)
class Item:
    """One item of a check: its class, what it says, the line to write (``None`` where there is none to give), and
    where that line goes."""

    cls: str
    what: str
    line: str | None
    place: str


@dataclass(frozen=True)
class Checked:
    """A check of one folder's ``mime.toml``: the file, the folder's kind, the profile and whether it was given (else it
    is the experiment's), and each item shown with its verdict."""

    path: Path
    kind: str
    profile: Profile
    given: bool
    items: tuple[tuple[str, Item], ...]

    def of(self, verdict: str) -> list[Item]:
        return [item for held, item in self.items if held == verdict]

    @property
    def findings(self) -> list[Item]:
        return self.of(FINDING)

    def text(self) -> str:
        """Each item in plain words with its line to write, as ``mime pending`` prints what is pending."""
        counts = ", ".join(_count(len(self.of(verdict)), verdict) for verdict in (FINDING, TO_DO, LISTED)
                           if self.of(verdict) or verdict == FINDING)
        whose = "given" if self.given else "the experiment's" if self.kind == BUNDLE else "its base experiment's"
        lines = [f"{self.path}, a {self.kind}'s manifest, checked under the {self.profile.value} profile ({whose}): "
                 f"{counts}."]
        for verdict in (FINDING, TO_DO, LISTED):
            held = self.of(verdict)
            if not held:
                continue
            lines.append(_HEADS[verdict])
            for number, item in enumerate(held, start=1):
                lines.append(f"{number}. {item.what}")
                if item.line is not None:
                    lines += [f"   {item.place}:", f"       {item.line}"]
        if not self.items:
            lines.append("nothing to do: push would take the manifest as it stands (the folder's own files are "
                         "`mime pack`'s to check).")
        return "\n".join(lines) + "\n"


def _count(number: int, verdict: str) -> str:
    return f"{number} {verdict}{'s' if verdict == FINDING and number != 1 else ''}"


def _studies(root: Path) -> list[Path]:
    from mime.saved._studies import study_folders

    return study_folders(root)


def _document(root: Path, kind: str) -> Mapping[str, object]:
    """The experiment whose profile the check takes by default: a saved folder's own; a study's base experiment (a
    publishing folder's first study's)."""
    from mime.results._study_folder import BASE_DIR
    from mime.saved import read_folder

    return cast(Mapping[str, object],
                read_folder(root if kind == BUNDLE else _studies(root)[0] / BASE_DIR, absent_referenced=True).document)


def _documents(root: Path, kind: str) -> list[tuple[str, Mapping[str, object]]]:
    """The experiments whose licence the manifest's kind table states, each with how an item names it: a saved folder's
    own; a study's base experiment and each group's own build; those of every study a publishing folder holds."""
    from mime.results._study_folder import BASE_DIR, GROUPS_DIR
    from mime.saved import read_folder

    if kind == BUNDLE:
        return [("the experiment", _document(root, kind))]
    found: list[tuple[str, Mapping[str, object]]] = []
    for study in _studies(root):
        prefix = f"{plain(study.name)}: " if study != root else ""
        builds = sorted((each for each in (study / GROUPS_DIR).glob(f"*/{BASE_DIR}") if each.is_dir()),
                        key=lambda each: each.parent.name)
        for folder, whose in [(study / BASE_DIR, "the study's base experiment"),
                              *((each, f"the study's group {plain(each.parent.name)} build") for each in builds)]:
            found.append((f"{prefix}{whose}",
                          cast(Mapping[str, object], read_folder(folder, absent_referenced=True).document)))
    return found


def _push_refusals(root: Path, kind: str, held: Mapping[str, object]) -> list[Item]:
    """What push refuses on a saved folder and its ``[record]`` that no key's check finds, each by push's own refusal
    (so the two cannot drift): the folder's own terms (a file the saved form does not list), a declared permission whose
    grantee is not the uploader ``[record]`` names (``push.permission``: with no uploader named it is what push still
    needs, with another grantee a defect), and a path of this machine or a token MIME holds in what ``[record]`` would
    send (``bundle.machine-path``, ``bundle.credential``). **Not checked, since each needs the bundle packed (which loads
    the experiment) or a service chosen** and a check builds, packs and sends nothing: the folder's own files (a file the
    saved form does not name, a link, a path or token inside a file: ``mime pack``'s refusals), the record's size and file
    count (``push.too-large``, ``push.too-many-files``), a file whose bytes may not travel (``push.referenced-copy``), a
    cached asset's bytes (``push.checksum``), and a sandbox location in a lock pushed to production
    (``push.sandbox-location``). That is why a clean check says the manifest, not the folder, would pass."""
    import json

    from mime.credentials import held_tokens
    from mime.push._plan import grantee_problems
    from mime.push._sharing import Reader, uploader_of
    from mime.saved._bundle import scan_text
    from mime.saved._card import judged_folder

    found = [_refused_item(DEFECT, each)
             for each in scan_text("the record's metadata", json.dumps(held.get("record", {}), sort_keys=True,
                                                                       default=str).encode("utf-8"), held_tokens())]
    if kind != BUNDLE:
        return found
    try:
        judged, _, _ = judged_folder(root)
    except BuildRefused as refused:
        return [*(_refused_item(DEFECT, each) for each in refused.refusals), *found]
    uploader = uploader_of(Reader(root / MANIFEST_FILE), _table(held.get("record")))
    for each in grantee_problems(judged.permissions, uploader, root) if judged.permissions else ():
        found.append(_refused_item(NEEDED, each, LINES["uploader"]) if uploader is None
                     else _refused_item(DEFECT, each))
    return found


def _refused_item(cls: str, refusal: Refusal, line: str | None = None) -> Item:
    """An item made from one of push's own refusals: its message, and its action as the place the fix goes where there
    is no one line to write."""
    if line is not None:
        return Item(cls, refusal.message, line, f"in {MANIFEST_FILE}, under [record]")
    return Item(cls, f"{refusal.message}: {refusal.action}", None, "")


def _profile(value: object) -> Profile:
    found, refusal = profile_of(value)
    if refusal is not None:
        code, message, action = refusal
        raise BuildRefused([Refusal(code, RefusalClass.USER, message, action)])
    assert found is not None
    return found


def _table(value: object) -> Mapping[str, object]:
    return cast(Mapping[str, object], value) if isinstance(value, dict) else {}


def _asked(path: Path, held: Mapping[str, object]) -> list[Item]:
    """What a reviewer would ask of ``[record]`` (``held``, as read): each person among the creators with no ORCID iD, no
    description, no keyword of the author's own."""
    table = _table(held.get("record"))
    place = f"in {MANIFEST_FILE}, under [record]"
    found: list[Item] = []
    creators = table.get("creators")
    for index, person in enumerate(cast(list[object], creators) if isinstance(creators, list) else []):
        each = _table(person)
        if isinstance(each.get("family"), str) and "orcid" not in each:
            found.append(Item(ASKED, f"{path.name}, [record] creators[{index}] ({plain(str(each['family']))}): names a "
                                     f"person with no ORCID iD, by which an index tells two people of one name apart",
                              'orcid = "0000-0000-0000-0000"', f"{place}, in creators[{index}]"))
    if not (isinstance(table.get("description"), str) and str(table["description"]).strip()):
        found.append(Item(ASKED, f"{path.name}, [record] description: is empty, so the record opens with the card alone",
                          'description = "<a paragraph on what this is and what it shows>"', place))
    if not table.get("keywords"):
        found.append(Item(ASKED, f"{path.name}, [record] keywords: names none, so the record is found by "
                                 f"mime-experiment alone", 'keywords = ["<a keyword>"]', place))
    return found


def check(folder: str | os.PathLike[str], profile: object | None = None) -> Checked:
    """Check ``mime.toml`` of a saved or study folder under ``profile`` (its experiment's where ``None``): every key's
    check, the licence against the experiment's, and what a reviewer would ask. Refused (nothing checked) where there is
    no ``mime.toml`` to check, where it does not read, is of a newer form or names a kind its folder is not, and for a
    profile that is not one or is not built (``strict``)."""
    root = Path(os.path.abspath(folder))
    _, problems = manifest_problems(root)
    kind = folder_kind(root) or BUNDLE
    path = root / MANIFEST_FILE
    documents = _documents(root, kind)
    chosen = _profile(profile if profile is not None else _document(root, kind).get("profile", Profile.EXPLORATORY.value))
    items = [Item(NEEDED if each.needed else DEFECT, each.refusal.message, each.line,
                  f"in {MANIFEST_FILE}" + (f", under [{each.table}]" if each.table else ""))
             for each in problems]
    _, held = read_toml(path)
    stated = _table(held.get(kind)).get("licence")
    for whose, document in documents:
        declared = document.get("licence")
        if declared is None:
            licence = stated if isinstance(stated, str) else "CC-BY-4.0"
            items.append(Item(NEEDED, f"{whose} declares no licence, and a record is never published without one",
                              f"exp.licence({licence!r})", "where the experiment is built, then save it again"))
        elif isinstance(stated, str) and stated != declared:
            items.append(Item(DEFECT, f"{path.name}, [{kind}] licence: is {plain(stated)}, and {whose} declares "
                                      f"{plain(str(declared))}, which its files carry",
                              f'licence = "{plain(str(declared))}"', f"in {MANIFEST_FILE}, under [{kind}]"))
    items += _push_refusals(root, kind, held)
    items += _asked(path, held)
    shown = tuple((verdict, item) for item in items if (verdict := RULES.get((chosen, item.cls))) is not None)
    return Checked(path, kind, chosen, profile is not None, shown)


__all__ = ["ASKED", "DEFECT", "FINDING", "LISTED", "NEEDED", "RULES", "TO_DO", "Checked", "Item", "check"]
