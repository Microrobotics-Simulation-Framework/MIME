"""What every generated page is written with: Markdown text, links between pages, and a value shown compactly. A
page is a list of lines; nothing here reads a clock, a path of the machine or the order of a set."""

from __future__ import annotations

import json
import posixpath
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast


@dataclass(frozen=True)
class Hatch:
    """An honest-state escape hatch a page prints (``none declared``, ``not measured``, ``undocumented``): its kind,
    and the id of its source, never a page, which is not committed."""

    kind: str
    item: str


@dataclass(frozen=True)
class Page:
    """A generated page: its path under the pages directory, its text, the free text from code it renders (a label
    and the string, for the number rule), and the hatches it prints."""

    path: str
    text: str
    rendered: tuple[tuple[str, str], ...] = ()
    hatches: tuple[Hatch, ...] = ()


@dataclass
class Writer:
    """A page being written."""

    lines: list[str] = field(default_factory=list[str])
    rendered: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])
    hatches: list[Hatch] = field(default_factory=list[Hatch])

    def add(self, *lines: str) -> None:
        self.lines.extend(lines)

    def paragraph(self, text: str) -> None:
        self.lines += [text, ""]

    def heading(self, level: int, text: str) -> None:
        self.lines += [f"{'#' * level} {text}", ""]

    def bullets(self, items: Iterable[str]) -> None:
        self.lines += [f"- {item}" for item in items] + [""]

    def table(self, headers: Sequence[str], rows: Iterable[Sequence[str]]) -> None:
        self.lines.append("| " + " | ".join(headers) + " |")
        self.lines.append("|" + "|".join("---" for _ in headers) + "|")
        self.lines += ["| " + " | ".join(cell(item) for item in row) + " |" for row in rows]
        self.lines.append("")

    def prose(self, label: str, text: str) -> str:
        """``text`` from the code, rendered as it stands: recorded for the number rule under ``label``."""
        self.rendered.append((label, text))
        return text

    def hatch(self, kind: str, item: str) -> None:
        self.hatches.append(Hatch(kind, item))

    def page(self, path: str) -> Page:
        return Page(path, "\n".join(self.lines).rstrip("\n") + "\n", tuple(self.rendered), tuple(self.hatches))


def cell(text: str) -> str:
    """A table cell: a pipe escaped, a line break a space."""
    return " ".join(text.replace("|", "\\|").split())


def code(text: object) -> str:
    """Inline code; a backtick in the text takes a longer fence."""
    shown = str(text)
    fence = "`" * (max((len(run) for run in re.findall(r"`+", shown)), default=0) + 1)
    pad = " " if shown.startswith("`") or shown.endswith("`") else ""
    return f"{fence}{pad}{shown}{pad}{fence}"


def link(from_page: str, to_page: str, label: str, anchor: str = "") -> str:
    """A relative link from one page's path to another's, both under the pages directory."""
    target = posixpath.relpath(to_page, posixpath.dirname(from_page) or ".")
    return f"[{label}]({target}{'#' + anchor if anchor else ''})"


def anchor(text: str) -> str:
    """The id a heading gets: lower case, words joined by hyphens."""
    return "-".join(re.findall(r"[a-z0-9]+", text.lower()))


def plain(value: object, limit: int = 160) -> str:
    """A value from the registry export, compactly: scalars as they are, a short list or mapping as JSON, anything
    longer left to the export. Keys are sorted, so the text does not depend on the order a mapping was built in."""
    if isinstance(value, str):
        return value
    shown = json.dumps(value, sort_keys=True, separators=(", ", ": "), ensure_ascii=False)
    return shown if len(shown) <= limit else "too long to show; see `mime registry --json`"


def mapping(value: object) -> Mapping[str, Any]:
    """``value`` as a mapping of the export, empty if it is not one."""
    return cast(Mapping[str, Any], value) if isinstance(value, Mapping) else {}


def items(value: object) -> list[Any]:
    """``value`` as a list of the export, empty if it is not one."""
    return list(cast(list[Any], value)) if isinstance(value, list) else []


def strings(value: object) -> list[str]:
    return [str(item) for item in items(value)]


__all__ = ["Hatch", "Page", "Writer", "anchor", "cell", "code", "items", "link", "mapping", "plain", "strings"]
