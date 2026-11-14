"""A block's card (``mime card <block folder>``, and its record's description when it is pushed) and its verification
(``mime verify <block folder | .mimeblock>``).

**The card** says what the block is and needs: its id, version, licence and tier, its content hash, what its
``[requires]`` names, its sockets, its assets with their tiers, its author's validity lines, its cases, what to cite with
it, and how to use it. The author's words are quoted as theirs: MIME's voice never carries an author's claim.

**Verifying** a block is an acceptance, never a bound on physics. Its ``[requires]`` is held to what is installed first:
a requirement not met refuses the whole (``block.requires``, the environment's to fix, exit 2), since a block that could
not be checked has neither passed nor moved. Each case must be this block's (``block.case-stale``, ``block.no-case``:
refused, nothing run). Then each case reruns as ``mime verify <folder>`` reruns a saved folder, a referenced block read
from this folder (a case's enclosing block, matched by its content hash): bit for bit in the lock's environment, within
tolerance in another. A line per case, then MIME's words, then the author's."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from mime._terminal import plain
from mime.blocks._definition import CASES, Definition, definition_of
from mime.build import BuildRefused
from mime.saved._schema import PublicationRecord

#: MIME's words after a block's verification: what it is, and what it is not.
ACCEPTANCE = ("An acceptance: the block's cases reproduce here. It bounds nothing about physics, and says nothing of whether "
              "the block applies to your experiment.")


def _cases(root: Path) -> list[Path]:
    folder = root / CASES
    return sorted(each for each in folder.iterdir() if each.is_dir()) if folder.is_dir() else []


def block_card_text(folder: str | os.PathLike[str], publication: PublicationRecord | None = None) -> str:
    """The card of the block folder ``folder`` as Markdown, naming the record ``publication`` names, or none."""
    from mime.blocks._definition import read_definition
    from mime.blocks._fragment import read_fragment
    from mime.blocks._show import adds
    from mime.blocks._use import requirement_rows

    root = Path(os.path.abspath(folder))
    definition = definition_of(root)
    full, _ = read_definition(root)
    record = full.record if full is not None else None
    verdict = definition.verdict
    title = plain(record.title) if record is not None else f"{plain(definition.id)} {plain(definition.version)}"
    doi = publication.get("doi") if publication is not None else None
    lines = [f"# {title}", "", f"- **Block** {plain(definition.id)} {plain(definition.version)}"
             + (f", DOI {plain(str(doi))}" if doi else ""),
             f"- **Licence of its content:** {plain(definition.licence)} ({verdict.words()}): "
             + ("copied into the document of an experiment that uses it" if definition.tier == "copied" else
                "used by reference: its files stay out of every user's folder, but the values its fragment sets reach "
                "their saved locks"),
             f"- **Content hash:** {definition.content_hash()}", ""]
    if record is not None and record.creators:
        lines += [f"Made by {'; '.join(person.words() for person in record.creators)}.", ""]
    lines += ["## What it needs", "",
              *(f"- {plain(row.name)} {plain(row.wanted)}" for row in requirement_rows(definition)), "",
              "Sockets, which its user binds: " + (", ".join(f"`{plain(name)}` ({kind})"
                                                             for name, kind in sorted(definition.sockets.items()))
                                                   or "none") + ".", ""]
    try:
        lines += [adds(cast(Mapping[str, Any], read_fragment(root, definition.sockets).doc)), ""]
    except BuildRefused:
        lines += ["Its fragment is refused here (`mime check <folder>` says why).", ""]
    if definition.assets:
        lines += ["## Assets", "", "| Asset | Size | Licence | Tier |", "|---|---|---|---|",
                  *(f"| {plain(ref.name)} | {ref.size} bytes | {plain(ref.licence)} | {ref.tier} |"
                    for ref in definition.assets), ""]
    lines += ["## Validity, as its author states it", ""]
    lines += [f"- {plain(each.line)} ({plain(each.reason)})" for each in definition.validity] or ["None stated."]
    cases = _cases(root)
    lines += ["", "## Cases", "", f"{len(cases)}: " + (", ".join(f"`{plain(each.name)}`" for each in cases) or "none")
              + ". `mime verify <folder>` reruns each: an acceptance, never a bound on physics."]
    if definition.shows:
        lines.append(f"The author says its cases show: {plain(definition.shows)}")
    lines += ["", "## How to use it", "", "```", *(f"pip install '{plain(row.name)}{plain(row.wanted)}'"
                                                  for row in requirement_rows(definition)),
              f"mime pull doi:{plain(str(doi))}" if doi else "mime pull <its record's DOI>",
              "```", "", "Then, in your experiment, `exp.use(<its folder>, " + ", ".join(
                  f"{plain(name)}=<your {kind}>" for name, kind in sorted(definition.sockets.items())) + ")`.", ""]
    if record is not None and record.also_cite:
        lines += [f"If you use this, please also cite: {'; '.join(each.words() for each in record.also_cite)}.", ""]
    return "\n".join(lines)


@dataclass(frozen=True)
class BlockVerified:
    """A block verified: its definition, and each case with whether it reproduced and its verification's text."""

    definition: Definition
    cases: tuple[tuple[str, bool, str], ...]

    @property
    def reproduced(self) -> bool:
        return all(held for _, held, _ in self.cases)

    def lines(self) -> list[str]:
        words = self.definition.words
        head = (f"\N{CHECK MARK} {words}: its {len(self.cases)} case{'s' if len(self.cases) != 1 else ''} reproduce"
                if self.reproduced else f"\N{BALLOT X} {words}: {sum(1 for _, held, _ in self.cases if not held)} of "
                                        f"its cases did not reproduce")
        lines = [head]
        for name, held, text in self.cases:
            lines.append(f"  - {plain(name)}: {'reproduced' if held else 'did not reproduce'}")
            if not held:
                lines += [f"    {line}" for line in text.splitlines()]
        lines.append(ACCEPTANCE)
        if self.definition.shows:
            lines.append(f"The author says: {plain(self.definition.shows)}")
        return lines


def verify_block(folder: str | os.PathLike[str]) -> BlockVerified:
    """Verify the block folder ``folder`` (the module says how); refused, nothing run, where it cannot be checked."""
    from mime.blocks._check import case_refusals
    from mime.blocks._use import requires_refused
    from mime.env import apply_lock_precision
    from mime.experiment import Experiment

    root = Path(os.path.abspath(folder))
    definition = definition_of(root)
    cases = _cases(root)
    refused = requires_refused(definition, f"none of its {len(cases)} case{'s' if len(cases) != 1 else ''} ran")
    if refused is not None:
        raise refused
    problems = [refusal for _, refusal in case_refusals(definition, definition.content_hash())]
    if problems:
        raise BuildRefused(problems)
    found: list[tuple[str, bool, str]] = []
    for case in cases:
        apply_lock_precision(case)
        experiment = Experiment.load(case)
        report = experiment.build().verify()
        load = experiment.load_report.text() if experiment.load_report is not None else ""
        found.append((f"{CASES}/{case.name}", report.reproduced, "\n".join(text for text in (load, report.text())
                                                                            if text)))
    return BlockVerified(definition, tuple(found))


__all__ = ["ACCEPTANCE", "BlockVerified", "block_card_text", "verify_block"]
