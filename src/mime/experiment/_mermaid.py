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

**In a notebook** the text is a :class:`Mermaid`, a ``str`` that also tells a notebook how to show itself: as a page of its own
in an ``<iframe srcdoc>``, in which Mermaid draws the diagram and a reader pans and zooms it (:mod:`mime.experiment._mermaid_page`),
the same in Colab, JupyterLab, Notebook 7 and VS Code. The page loads Mermaid from jsDelivr, pinned to an exact version
and held to its hash, because an output that carried the library would be 4.8 MB and a Jupyter server drops an output
that large; ``display(inline=True)`` carries the copy shipped in the wheel instead and ``save(path)`` writes the page with
it, so a page can be had with no network. ``display(offline=True)`` shows the text and a line saying where to draw it and
loads nothing. A front end that runs no script (a static preview of the notebook) draws nothing of the page, and shows
the note under it and the ``text/plain`` part, which is the text. Colab showed the text and no diagram when the HTML was
put in its output directly (measured by the maintainer, 2026-11-07); the cause is not known, and the iframe is a
document that does not depend on how the front end inserts scripts."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, override

from mime.errors._frames import warn_at_script
from mime.experiment import _mermaid_page as view
from mime.experiment._mermaid_page import (MERMAID_CDN, MERMAID_SRI, MERMAID_VERSION, OUTPUT_WARNING_BYTES, RENDERER)

if TYPE_CHECKING:
    from maddening import GraphManager

    from mime.build._report import BuildReport

#: The line a notebook shows under the text it shows in place of a diagram (``display(offline=True)``).
POINTER = f"Mermaid diagram. If your notebook shows its text instead, paste it into {RENDERER} to draw it."

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
    notebook shows it as a diagram a reader can pan and zoom: a page in an ``<iframe srcdoc>`` (see :meth:`page`)."""

    __slots__ = ()

    def page(self, *, cdn: bool = False) -> str:
        """The text as one self-contained HTML page that draws it, with buttons and gestures to pan and zoom: Mermaid's
        library inlined from the copy shipped in the wheel, so the page needs no network, or from jsDelivr, held to its
        hash, with ``cdn=True`` (3.6 MB smaller). The same text gives the same page byte for byte."""
        return view.page(str(self), cdn=cdn)

    def save(self, path: str | Path, *, cdn: bool = False) -> Path:
        """Write :meth:`page` to ``path`` (to open in a browser) and return it."""
        written = Path(path)
        written.write_text(self.page(cdn=cdn), encoding="utf-8")
        return written

    def frame(self, *, inline: bool = False) -> str:
        """The page as a notebook shows it: in an ``<iframe srcdoc>`` the reader can drag taller or shorter, with a note
        under it. The library is jsDelivr's unless ``inline``, which puts the shipped copy in the output (4.8 MB more)."""
        return view.frame(self.page(cdn=not inline))

    def text_with_pointer(self) -> str:
        """The text as a Markdown code block (not a diagram) with the pointer line: what is shown where nothing may be
        loaded, and nothing draws a diagram."""
        return f"```text\n{self}```\n\n*{POINTER}*\n"

    def display(self, *, offline: bool = False, inline: bool = False) -> None:
        """Show the diagram in the notebook now, as it would be shown by returning it from a cell: a page that draws it,
        with pan and zoom, Mermaid from jsDelivr (held to its hash). ``inline=True`` carries the library in the output
        instead, so the page needs no network (and the output is 4.8 MB, which a Jupyter server may drop at its default
        output rate: it warns). ``offline=True`` shows the text and the pointer line and loads nothing, anywhere."""
        try:
            from IPython.display import display as show  # pyright: ignore[reportMissingImports, reportUnknownVariableType]
        except ImportError as error:
            raise RuntimeError("Mermaid.display shows in a notebook, and IPython is not installed: print the "
                               "diagram instead (print(built.to_mermaid()))") from error
        bundle = self._bundle(offline=offline, inline=inline)
        size = len(bundle.get("text/html", "").encode("utf-8"))
        if size > OUTPUT_WARNING_BYTES:
            warn_at_script(
                f"the diagram is {size / 1e6:.1f} MB as a notebook output, which Jupyter may drop at its default output "
                "rate: leave the library to jsDelivr (inline=False), or write a page instead (diagram.save('graph.html'))",
                UserWarning)
        show(bundle, raw=True)

    def _bundle(self, *, offline: bool = False, inline: bool = False) -> dict[str, str]:
        if offline:
            return {"text/markdown": self.text_with_pointer(), "text/plain": str(self)}
        return {"text/html": self.frame(inline=inline), "text/plain": str(self)}

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


__all__ = ["MERMAID_CDN", "MERMAID_SRI", "MERMAID_VERSION", "POINTER", "RENDERER", "Mermaid", "mermaid_of"]
