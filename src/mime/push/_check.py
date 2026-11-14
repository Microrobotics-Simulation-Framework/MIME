"""``mime check <folder>``: a shared folder's ``mime.toml`` checked offline, on the profile workflow.

Every key's check is run (:mod:`mime.push._manifest`), with the licence its kind table states against the one its
experiment declares; nothing is built, packed or sent. Each item has a class, and the profile says what each class is, as
the profile rules say what a build does with a finding (:data:`RULES`):

- **a defect** (a key unknown or of another type, an ORCID iD's check digit, a licence unlike the experiment's) is a
  finding in every profile, since no profile relaxes an input error;
- **what push still needs** (no creator named, no title, an experiment that declares no licence) is a to-do in
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
            lines.append("nothing to do: push would take it as it stands.")
        return "\n".join(lines) + "\n"


def _count(number: int, verdict: str) -> str:
    return f"{number} {verdict}{'s' if verdict == FINDING and number != 1 else ''}"


def _document(root: Path, kind: str) -> Mapping[str, object]:
    from mime.results._study_folder import BASE_DIR
    from mime.saved import read_folder

    return cast(Mapping[str, object],
                read_folder(root if kind == BUNDLE else root / BASE_DIR, absent_referenced=True).document)


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
    document = _document(root, kind)
    chosen = _profile(profile if profile is not None else document.get("profile", Profile.EXPLORATORY.value))
    items = [Item(NEEDED if each.needed else DEFECT, each.refusal.message, each.line,
                  f"in {MANIFEST_FILE}" + (f", under [{each.table}]" if each.table else ""))
             for each in problems]
    _, held = read_toml(path)
    declared = document.get("licence")
    stated = _table(held.get(kind)).get("licence")
    whose = "the experiment" if kind == BUNDLE else "the study's base experiment"
    if declared is None:
        licence = stated if isinstance(stated, str) else "CC-BY-4.0"
        items.append(Item(NEEDED, f"{whose} declares no licence, and a record is never published without one",
                          f"exp.licence({licence!r})", "where the experiment is built, then save it again"))
    elif isinstance(stated, str) and stated != declared:
        items.append(Item(DEFECT, f"{path.name}, [{kind}] licence: is {plain(stated)}, and {whose} declares "
                                  f"{plain(str(declared))}, which its files carry",
                          f'licence = "{plain(str(declared))}"', f"in {MANIFEST_FILE}, under [{kind}]"))
    items += _asked(path, held)
    shown = tuple((verdict, item) for item in items if (verdict := RULES.get((chosen, item.cls))) is not None)
    return Checked(path, kind, chosen, profile is not None, shown)


__all__ = ["ASKED", "DEFECT", "FINDING", "LISTED", "NEEDED", "RULES", "TO_DO", "Checked", "Item", "check"]
