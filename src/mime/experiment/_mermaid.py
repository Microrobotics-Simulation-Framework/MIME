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

**In a notebook** the text is a :class:`Mermaid`, a ``str`` that also tells a notebook how to show itself. Colab does not
draw a Markdown Mermaid fence (measured by the maintainer, 2026-11-05), so there it is ``text/html``: a page that loads
Mermaid's library from jsDelivr, pinned to an exact version and held to its hash (``integrity``), as the viewer of
:mod:`mime.viz._web` is, and draws the diagram with an explicit ``mermaid.run()``. Anywhere else it is a ``text/markdown``
Mermaid fence, which a front end that draws Mermaid in Markdown draws (JupyterLab 4.1 and after, Notebook 7), and which
one that does not shows as the Mermaid text, with a line saying where to draw it; that loads no script and makes no
network call. :meth:`Mermaid.display` with ``offline=True`` shows the text and the line anywhere and loads nothing."""

from __future__ import annotations

import hashlib
import html
import re
import sys
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, override

if TYPE_CHECKING:
    from maddening import GraphManager

    from mime.build._report import BuildReport

#: Where to draw Mermaid text that a front end shows as text.
RENDERER = "https://mermaid.live"
#: The line a notebook shows under the Mermaid, drawn or not: a ``text/markdown`` output cannot be shown to some front ends
#: and not to others, so the line is worded to read well under a diagram and under the text.
POINTER = f"Mermaid diagram. If your notebook shows its text instead, paste it into {RENDERER} to draw it."

#: The library Colab's output loads to draw the diagram: Mermaid's single-file build (a classic script, so that
#: ``integrity`` holds every byte it runs: the ES-module build loads further files that an ``integrity`` attribute does not
#: cover), pinned to an exact version. The integrity is sha384 as SRI writes it, of the file jsDelivr serves at
#: ``MERMAID_CDN``; to check a bump: ``curl -sL <MERMAID_CDN> | openssl dgst -sha384 -binary | openssl base64 -A``.
MERMAID_VERSION = "11.17.2"
MERMAID_SRI = "sha384-EOXBFmc3gx5mb+vn0vPvvGqACToJD24hhacX5Yx+8NUUQrHIle/Qi5Bg9o3zKwW2"
MERMAID_CDN = f"https://cdn.jsdelivr.net/npm/mermaid@{MERMAID_VERSION}/dist/mermaid.min.js"

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


def in_colab() -> bool:
    """Whether the kernel is Google Colab's (its ``google.colab`` package is loaded)."""
    return "google.colab" in sys.modules


class Mermaid(str):
    """Mermaid flowchart text. It is a ``str`` in every way (``print`` it, write it, paste it into a renderer), and a
    notebook shows it as a diagram: in Colab by loading the pinned library (see :meth:`html`), elsewhere as a Markdown
    Mermaid fence, which a front end that draws one draws and any other shows as the text with a pointer to a
    renderer."""

    __slots__ = ()

    def fence(self) -> str:
        """The text as a Markdown fence, with the pointer under it."""
        return f"```mermaid\n{self}```\n\n*{POINTER}*\n"

    def html(self) -> str:
        """The text as an HTML fragment that draws the diagram: Mermaid's library from jsDelivr (held to its
        ``integrity``), then a module script that draws the element with ``mermaid.run()``. The element holds the text, so
        where the library does not load (no network, a changed file) the text is shown, with the pointer line."""
        ident = "mime-mermaid-" + hashlib.sha256(str(self).encode("utf-8")).hexdigest()[:12]
        return (
            f'<pre class="mermaid" id="{ident}">{html.escape(str(self), quote=False)}</pre>\n'
            f'<p><em>{html.escape(POINTER, quote=False)}</em></p>\n'
            f'<script src="{MERMAID_CDN}" integrity="{MERMAID_SRI}" crossorigin="anonymous"></script>\n'
            '<script type="module">\n'
            f'const element = document.getElementById("{ident}");\n'
            "if (globalThis.mermaid && element) {\n"
            "  mermaid.initialize({startOnLoad: false});\n"
            "  await mermaid.run({nodes: [element]});\n"
            "}\n"
            "</script>\n")

    def text_with_pointer(self) -> str:
        """The text as a Markdown code block (not a diagram) with the pointer line: what is shown where nothing may be
        loaded, and nothing draws a diagram."""
        return f"```text\n{self}```\n\n*{POINTER}*\n"

    def display(self, *, offline: bool = False) -> None:
        """Show the diagram in the notebook now, as it would be shown by returning it from a cell: drawn in Colab by the
        pinned library from jsDelivr, as a Markdown Mermaid fence elsewhere. ``offline=True`` shows the text and the
        pointer line and loads nothing, anywhere."""
        try:
            from IPython.display import display as show  # pyright: ignore[reportMissingImports, reportUnknownVariableType]
        except ImportError as error:
            raise RuntimeError("Mermaid.display shows in a notebook, and IPython is not installed: print the "
                               "diagram instead (print(built.to_mermaid()))") from error
        show(self._bundle(offline=offline), raw=True)

    def _bundle(self, *, offline: bool = False) -> dict[str, str]:
        if offline:
            return {"text/markdown": self.text_with_pointer(), "text/plain": str(self)}
        if in_colab():
            return {"text/html": self.html(), "text/plain": str(self)}
        return {"text/markdown": self.fence(), "text/plain": str(self)}

    def _repr_mimebundle_(self, include: object = None, exclude: object = None) -> dict[str, str]:
        return self._bundle()

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
