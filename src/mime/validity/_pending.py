"""What is left pending, in plain words with the lines to paste, through one formatter; and the stamp every output
carries.

Each item says what it is about, why it is pending, and where its reason goes, as a line of Python to paste into the
script and a fragment of the saved document to merge into ``experiment.mime.yaml``, each with :data:`REASON` where the
reason goes. The formatter fills that slot with ``pending()`` (``{pending: true}`` in the document), the line that
builds as it is in exploratory; sealing replaces it with a reason. a1's drafts and named bases fill the same slot."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import yaml

#: Where an item's reason goes in its lines.
REASON = "<reason>"
#: The reason the formatter writes in each slot: what builds in exploratory as it is.
PENDING_PYTHON = "pending()"
PENDING_YAML: Mapping[str, object] = {"pending": True}

type Json = None | bool | int | float | str | list[Json] | dict[str, Json]


@dataclass(frozen=True)
class PendingItem:
    """One thing a sealed experiment states and this one does not: what it is about, in the experiment's names; what
    is pending, in plain words; where the Python line goes; the line, and the document fragment, each with
    :data:`REASON` where its reason goes. ``listed`` marks an item research lists (a physical value's missing source, a
    numerical value's missing convergence evidence), which is never refused."""

    target: str
    what: str
    where: str
    python: str
    document: Json
    listed: bool = False
    #: The item in a few words (``its reason pending``, ``a numerical default``), as the stamp names it.
    kind: str = "pending"
    #: A second line, written after the experiment is made (an ``exp.source(...)``), with its reason's slot.
    also: str = ""

    def line(self) -> str:
        """The item in one line, as the stamp carries it: what it is about, and what kind of item it is."""
        return f"{self.target} ({self.kind})"


def _filled(value: Json, reason: object) -> object:
    if value == REASON:
        return reason
    if isinstance(value, dict):
        return {key: _filled(item, reason) for key, item in value.items()}
    if isinstance(value, list):
        return [_filled(item, reason) for item in value]
    return value


def python_line(item: PendingItem, reason: str = PENDING_PYTHON) -> str:
    """The item's Python line with ``reason`` in its slot."""
    return item.python.replace(REASON, reason)


def also_line(item: PendingItem, reason: str = PENDING_PYTHON) -> str:
    """The item's second line, written after the experiment is made, with ``reason`` in its slot (empty for none)."""
    return item.also.replace(REASON, reason)


def document_fragment(item: PendingItem, reason: object = PENDING_YAML) -> object:
    """The item's document fragment with ``reason`` in its slot."""
    return _filled(item.document, dict(cast(Mapping[str, object], reason)) if isinstance(reason, Mapping) else reason)


def _yaml(fragment: object) -> str:
    return yaml.safe_dump(fragment, sort_keys=False, default_flow_style=None, width=120, allow_unicode=True)


def pending_text(profile: str, items: Sequence[PendingItem]) -> str:
    """Every item in plain words with its lines to paste, in Python and in the document's YAML: the one formatter
    ``built.report.pending()`` and ``mime pending <folder>`` print through."""
    pending = [item for item in items if not item.listed]
    listed = [item for item in items if item.listed]
    lines: list[str] = []
    if pending:
        lines.append(f"profile {profile}: {len(pending)} item{'s' if len(pending) != 1 else ''} pending, each recorded "
                     f"in the saved form and stamped on every output. To seal the experiment, give each its reason "
                     f"in place of {PENDING_PYTHON} (in the document, in place of {{pending: true}}) and build with "
                     f"profile=\"research\", which refuses a pending reason.")
    else:
        lines.append(f"profile {profile}: nothing pending.")
    for number, item in enumerate(pending, start=1):
        lines += _item_lines(number, item)
    if listed:
        lines.append(f"listed by {profile}, never refused: {len(listed)} suggestion"
                     f"{'s' if len(listed) != 1 else ''}, each the line that states it.")
        for number, item in enumerate(listed, start=1):
            lines += _item_lines(number, item, reason='"..."')
    return "\n".join(lines) + "\n"


def _item_lines(number: int, item: PendingItem, reason: str | None = None) -> list[str]:
    python = python_line(item, reason) if reason is not None else python_line(item)
    fragment = document_fragment(item, reason.strip('"') if reason is not None else PENDING_YAML)
    lines = [f"{number}. {item.target}, {item.kind}: {item.what}", f"   in Python, {item.where}:", f"       {python}"]
    if item.also:
        lines += ["   and after the experiment is made:",
                  f"       {item.also.replace(REASON, reason if reason is not None else PENDING_PYTHON)}"]
    lines.append("   in the document (experiment.mime.yaml), merged in:")
    lines += [f"       {line}" for line in _yaml(fragment).rstrip("\n").splitlines()]
    return lines


@dataclass(frozen=True)
class Stamp:
    """What every output of a build carries: the profile it was built under and each item left pending, one line
    each. Empty ``pending`` is a sealed build's."""

    profile: str
    pending: tuple[str, ...] = ()

    def text(self) -> str:
        if not self.pending:
            return f"profile {self.profile}: nothing pending"
        return (f"profile {self.profile}: {len(self.pending)} item{'s' if len(self.pending) != 1 else ''} pending, "
                f"not sealed: " + "; ".join(self.pending))

    def record(self) -> dict[str, Any]:
        return {"profile": self.profile, "pending": list(self.pending)}

    @classmethod
    def from_record(cls, held: Mapping[str, Any]) -> Stamp:
        return cls(str(held["profile"]), tuple(str(line) for line in cast(Sequence[object], held.get("pending", ()))))


__all__ = ["PENDING_PYTHON", "PENDING_YAML", "REASON", "PendingItem", "Stamp", "also_line", "document_fragment",
           "pending_text", "python_line"]
