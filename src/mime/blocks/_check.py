"""``mime check <block folder>``: a block checked offline, on the profile workflow of every kind's check
(:mod:`mime.push._check`): nothing is built, run, packed or sent.

- **Defects**, in every profile: a key of its definition (``manifest.key``), a socket (``block.socket``), the fragment
  (``block.fragment``, ``block.format``), a requirement this environment does not meet (``block.requires``: the range,
  or the environment, is not the one the block is made in, so its entries were not resolved), an entry from a
  distribution ``[requires]`` does not name (``block.entry``), a case saved against other content
  (``block.case-stale``), and what ``[record]`` holds wrongly.
- **What publishing still needs** (a to-do in exploratory, a finding in research): a case (``block.no-case``: ``mime
  pack`` refuses a block with none), and what push needs of ``[record]``. A case that does not read, or does not use the
  block, is ``block.case-stale`` too: it tests nothing of this content.
- **What a reviewer would ask** (listed by research): ``[record]``'s.

A block has no profile of its own: exploratory, unless one is given. Where its licence makes it referenced, the check
says what that tier cannot hide (the values its fragment sets reach every user's saved lock)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from mime._manifest_file import MANIFEST_FILE
from mime._terminal import plain
from mime.blocks._definition import CASES, Definition, read_definition
from mime.blocks._fragment import read_fragment
from mime.build import BuildRefused, Refusal, RefusalClass
from mime.push._check import DEFECT, NEEDED, RULES, Checked, Item, asked_items, profile_named, refused_item
from mime.push._manifest import BLOCK, read_toml
from mime.validity import Profile

#: Section 2.8's words to the author of a block used by reference.
REFERENCED_WORDS = ("Its licence makes this block referenced: its files stay out of every user's bundle and record, but "
                    "not its numbers. The values its fragment sets appear in every user's saved lock (the resolved "
                    "record) and in what their builds compute; keep what you protect in an asset with no location (a "
                    "file had by request), never in the fragment.")


def stale_uses(uses: list[dict[str, object]], content: str) -> list[dict[str, object]]:
    """The uses of a case saved against content other than ``content``, the block's now."""
    return [use for use in uses if use.get("content") != content]


def case_refusals(definition: Definition, content: str) -> list[tuple[str, Refusal]]:
    """Each case's problems, with its class: a case that does not read or does not use the block, one saved against
    other content (``block.case-stale``), and none at all (``block.no-case``)."""
    from mime.saved import DOCUMENT_FILE
    from mime.saved._document import parse_yaml

    folder = definition.folder / CASES
    cases = sorted(each for each in folder.iterdir() if each.is_dir()) if folder.is_dir() else []
    if not cases:
        return [(NEEDED, Refusal(
            "block.no-case", RefusalClass.USER,
            f"{definition.words} has no case: `mime pack` refuses a block with none, since `mime verify` reruns its "
            f"cases", f"save an experiment that uses it into {folder}/<case>/, `exp.save(...)` after "
                      f"`exp.use(...)`"))]
    found: list[tuple[str, Refusal]] = []
    for case in cases:
        where = f"{CASES}/{plain(case.name)}"
        try:
            document = cast(dict[str, Any], parse_yaml((case / DOCUMENT_FILE).read_text(encoding="utf-8")))
        except (OSError, ValueError) as error:
            found.append((DEFECT, Refusal("block.case-stale", RefusalClass.USER,
                                          f"{where} holds no saved document that reads ({plain(str(error))})",
                                          f"save the case again from its script, `exp.save(<the case's folder>)`")))
            continue
        held: dict[str, object] = cast(dict[str, Any], document.get("blocks") or {})
        uses = [cast(dict[str, object], use) for use in held.values()
                if isinstance(use, dict) and cast(dict[str, object], use).get("id") == definition.id]
        if not uses:
            found.append((DEFECT, Refusal("block.case-stale", RefusalClass.USER,
                                          f"{where}: its experiment does not use {definition.words}, so it tests "
                                          f"nothing of the block",
                                          f"save an experiment that uses it there, `exp.use(...)` then "
                                          f"`exp.save(<the case's folder>)`")))
        stale = stale_uses(uses, content)
        if stale:
            found.append((DEFECT, Refusal(
                "block.case-stale", RefusalClass.USER,
                f"{where} was saved against other content of {definition.words} (content "
                f"{plain(str(stale[0].get('content')))}, and the block's is now {content})",
                "save the case again from its script after the block's change, `exp.save(<the case's folder>)`, so "
                "that it tests this content")))
    return found


def check_block(root: Path, profile: object | None) -> Checked:
    """``mime check`` of the block folder ``root`` under ``profile`` (exploratory where ``None``)."""
    from mime.blocks._use import entry_refusals, expanded_for_check, requires_refusal, requirement_rows

    path = root / MANIFEST_FILE
    chosen = profile_named(profile if profile is not None else Profile.EXPLORATORY.value)
    definition, problems = read_definition(root)
    items = [Item(NEEDED if each.needed else DEFECT, each.refusal.message, each.line,
                  f"in {MANIFEST_FILE}" + (f", under [{each.table}]" if each.table else "")) if each.line is not None
             else refused_item(NEEDED if each.needed else DEFECT, each.refusal) for each in problems]
    notes: list[str] = []
    if definition is not None:
        try:
            fragment = read_fragment(root, definition.sockets)
        except BuildRefused as refused:
            fragment = None
            items += [refused_item(DEFECT, each) for each in refused.refusals]
        unmet = [row for row in requirement_rows(definition) if not row.agrees]
        if unmet:
            items.append(refused_item(DEFECT, requires_refusal(
                definition, unmet, "its entries were not resolved here: either its range is wrong or this environment "
                                   "is not one the block is made in")))
        elif fragment is not None:
            try:
                expanded = expanded_for_check(definition, fragment)
            except BuildRefused as refused:
                items += [refused_item(DEFECT, each) for each in refused.refusals]
            else:
                items += [refused_item(DEFECT, each) for each in entry_refusals(definition, expanded.sections)]
        items += [refused_item(cls, refusal) for cls, refusal in case_refusals(definition, definition.content_hash())]
        if definition.tier == "referenced":
            notes.append(REFERENCED_WORDS)
    _, held = read_toml(path)
    items += asked_items(path, held)
    shown = tuple((verdict, item) for item in items if (verdict := RULES.get((chosen, item.cls))) is not None)
    return Checked(path, BLOCK, chosen, profile is not None, shown, tuple(notes))


__all__ = ["REFERENCED_WORDS", "case_refusals", "check_block", "stale_uses"]
