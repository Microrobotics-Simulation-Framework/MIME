"""The compiled graph as Mermaid text, with MIME's names.

MADDENING's ``GraphManager.to_mermaid`` draws every node, edge, coupling group and external input of the graph it holds.
Its nodes are labelled with the node's name (which MIME already makes out of the entry's id and what it binds,
``drag:umr``) and the name of the node's class (``SurrogateDragNode``), an internal name. This adds, under the node's
name, the registered entry that built it and its version (``drag 0.4.0, rung surrogate``), and leaves the rest of
MADDENING's text as it is: the edges, the groups and the external inputs are MADDENING's, not drawn a second time here.

The label is rewritten in the text MADDENING returns, by the node's name, because MADDENING gives no hook for a node's
label; :func:`_escaped` is MADDENING's escaping, held to it by a test. A node whose line is not found keeps MADDENING's
label, and a test holds that to none, so a change in MADDENING's text is seen and not silently drawn without MIME's
names.

**Determinism.** MADDENING numbers the nodes in the order they were added, and MIME adds them in its canonical order,
which the order of a script's calls cannot change, so the same experiment gives the same text byte for byte.

**In a notebook** the text is a :class:`Mermaid`, a ``str`` that also tells a notebook how to show itself as a
``text/markdown`` Mermaid fence, which a front end that draws Mermaid in Markdown draws, and which one that does not shows
as the Mermaid text, with a line saying where to draw it. It loads no script and makes no network call."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, override

if TYPE_CHECKING:
    from maddening import GraphManager

    from mime.build._report import BuildReport

#: Where to draw Mermaid text that a front end shows as text.
RENDERER = "https://mermaid.live"
#: The line a notebook shows under the Mermaid text, when the front end draws it as text and not as a diagram.
POINTER = f"Mermaid source: if it shows as text, paste it into {RENDERER} to draw it."

_NODE_LINE = re.compile(r'^(?P<head>\s*n\d+\[")(?P<name>.*?)<br/>(?P<type>.*)(?P<tail>"\])$')


def _entity(code: str) -> str:
    """Mermaid's entity code: ``#`` and the code, then ``;``."""
    return f"#{code};"


def _escaped(text: str) -> str:
    """Text made safe inside a quoted Mermaid label, as MADDENING makes it (``inspection._mermaid_text``): ``#`` first,
    since it opens an entity code, then the characters that would end the label or open HTML or Markdown."""
    out = text.replace("#", _entity("35"))
    for character, code in (('"', "quot"), ("<", "lt"), (">", "gt"), ("&", "38"), ("`", "96")):
        out = out.replace(character, _entity(code))
    return out.replace("\r\n", "<br/>").replace("\n", "<br/>").replace("\r", "<br/>")


class Mermaid(str):
    """Mermaid flowchart text. It is a ``str`` in every way (``print`` it, write it, paste it into a renderer), and a
    notebook shows it as a diagram where the front end draws a Mermaid fence in Markdown, and as the text, with a
    pointer to a renderer, where it does not."""

    __slots__ = ()

    def fence(self) -> str:
        """The text as a Markdown fence, with the pointer under it."""
        return f"```mermaid\n{self}```\n\n*{POINTER}*\n"

    def _repr_mimebundle_(self, include: object = None, exclude: object = None) -> dict[str, str]:
        return {"text/markdown": self.fence(), "text/plain": str(self)}

    @override
    def __repr__(self) -> str:
        return str.__repr__(self)


def mermaid_of(graph: GraphManager, report: BuildReport, *, direction: str = "LR") -> Mermaid:
    """MADDENING's Mermaid for ``graph``, each node's label carrying the registered entry that built it (``report``'s
    node lines say which). ``direction`` is MADDENING's: ``LR``, ``RL``, ``TB`` or ``BT``."""
    entries: Mapping[str, str] = {_escaped(line.node): _escaped(line.entry) for line in report.nodes}
    drawn: Any = graph.to_mermaid(direction=direction)
    lines: list[str] = []
    for line in str(drawn).splitlines():
        found = _NODE_LINE.match(line)
        if found is not None and found["name"] in entries:
            line = f"{found['head']}{found['name']}<br/>{entries[found['name']]}<br/>{found['type']}{found['tail']}"
        lines.append(line)
    return Mermaid("\n".join(lines) + "\n")
