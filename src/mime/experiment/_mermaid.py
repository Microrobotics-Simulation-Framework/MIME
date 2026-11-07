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
the same in Colab, JupyterLab, Notebook 7 and VS Code. It takes the 3-D view's names and behaviour (:mod:`mime.viz._web`), one
word meaning one thing across both: ``display(offline=True)`` inlines the library from the copy shipped in the wheel, so the
page needs no network; the default loads it from jsDelivr, pinned to an exact version and held to its hash, because an inlined
page is a 4.8 MB output that a notebook would save into its file with every display. ``display`` returns a :class:`GraphPage`
(as the 3-D view returns a ``WebPage``): a cell shows it when it is the cell's last expression, and ``.save(path)`` writes the
page it holds. :meth:`Mermaid.save` is the file helper (the 3-D view's ``web``): the library inlined unless ``cdn``, a default
name that is never the 3-D page's (``<experiment>.graph.html``), refused inside a saved or run folder. A graph built before a
replacement or a removal says it is stale where it is shown, as the 3-D view of such a run does. A front end that runs no
script (a static preview of the notebook) draws nothing of the page, and shows the note under it and the ``text/plain`` part,
which is the text. Colab showed the text and no diagram when the HTML was put in its output directly (measured by the
maintainer, 2026-11-07); the cause is not known, and the iframe is a document that does not depend on how the front end
inserts scripts."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, override

from mime._lineage import Origin
from mime.build._refusals import ArgumentRefused, Refusal, RefusalClass
from mime.errors._frames import warn_at_script
from mime.experiment import _mermaid_page as view
from mime.experiment._mermaid_page import (
    FRAME_HEIGHT,
    MERMAID_CDN,
    MERMAID_SRI,
    MERMAID_VERSION,
    OUTPUT_WARNING_BYTES,
    RENDERER,
)

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


#: What the graph's staleness says to do (the 3-D view's says to build and run again).
STALE_ACTION = "build the experiment again"


def written_outside(path: Path, command: str) -> None:
    """Refuse a page written inside a saved or run folder, which never holds presentation (``visuals.web-out``, the 3-D
    view's code); ``command`` is what to run instead, in the words of the call or command that was used."""
    from mime.viz._web import held_by

    folder = held_by(path)
    if folder is not None:
        raise ArgumentRefused([Refusal(
            "visuals.web-out", RefusalClass.USER,
            f"{path} is inside {folder}, a saved or run folder, which never holds presentation: a page is written "
            f"beside it, never into it", f"name a file outside it: `{command}`")])


@dataclass(frozen=True)
class GraphPage:
    """A graph as a notebook shows it: the page in an ``<iframe srcdoc>``, as the 3-D view's ``WebPage`` is. A cell whose
    last expression it is shows it; :meth:`save` writes the page it holds. Its ``repr`` is one line, not the page.
    ``height`` is the frame's, in pixels."""

    html: str
    name: str
    height: int = FRAME_HEIGHT
    #: The declarations the graph's build came from, for a graph built in this process (``mime._lineage``).
    origin: Origin | None = None

    @property
    def stale(self) -> str | None:
        """Why this graph no longer matches its experiment, or ``None``: the replacements and removals made since the build,
        and what to do. Shown wherever the graph is shown."""
        return self.origin.stale("this graph", STALE_ACTION) if self.origin is not None else None

    @override
    def __repr__(self) -> str:
        """One line, not the page: Jupyter shows ``text/plain`` beside the HTML, and the page is megabytes."""
        stale = self.stale
        return (f"<GraphPage {self.name!r}, {len(self.html.encode('utf-8')) / 1e3:.0f} kB: a notebook cell shows it "
                f"when it is the cell's last expression; elsewhere .save('graph.html') writes the page, to open in a "
                f"browser" + (f"; STALE, {stale}" if stale is not None else "") + ">")

    def _repr_html_(self) -> str:
        return view.frame(self.html, height=self.height, stale=self.stale)

    def save(self, path: str | Path) -> Path:
        """Write the page as displayed (the library from jsDelivr unless it was displayed ``offline=True``)."""
        written = Path(path)
        written.write_text(self.html, encoding="utf-8")
        return written


class Mermaid(str):
    """Mermaid flowchart text. It is a ``str`` in every way (``print`` it, write it, paste it into a renderer), and a
    notebook shows it as a diagram a reader can pan and zoom: a page in an ``<iframe srcdoc>`` (see :meth:`display`).
    ``origin`` (the declarations its build came from) and ``name`` (the experiment's) ride on it for the stale mark and the
    default file name; they are not part of the text, which compares, hashes and slices as the ``str`` it is."""

    __slots__ = ("name", "origin")
    origin: Origin | None
    name: str | None

    def __new__(cls, text: object = "", *, origin: Origin | None = None, name: str | None = None) -> Mermaid:
        made = super().__new__(cls, str(text))
        made.origin = origin
        made.name = name
        return made

    @property
    def stale(self) -> str | None:
        """Why this graph no longer matches its experiment, or ``None``: a graph built before a replacement or a removal.
        A graph of a loaded experiment, or one written by hand, never is."""
        return self.origin.stale("this graph", STALE_ACTION) if self.origin is not None else None

    def page(self, *, cdn: bool = False, caption: str | None = None) -> str:
        """The text as one self-contained HTML page that draws it, with buttons and gestures to pan and zoom: Mermaid's
        library inlined from the copy shipped in the wheel, so the page needs no network, or from jsDelivr, held to its
        hash, with ``cdn=True`` (3.6 MB smaller). ``caption`` is a line above the diagram; a stale graph says so first.
        The same text gives the same page byte for byte."""
        return view.page(str(self), cdn=cdn, caption=caption, stale=self.stale)

    def save(self, path: str | Path | None = None, *, cdn: bool = False, caption: str | None = None) -> Path:
        """Write :meth:`page` to ``path`` (``<experiment>.graph.html`` here by default: never the 3-D page's name, so a
        graph written beside a view of the same experiment does not overwrite it) and return it. The library is inlined
        unless ``cdn``. Never inside a saved or run folder (``visuals.web-out``). A stale graph warns."""
        written = Path(path) if path is not None else Path(f"{self.name or 'graph'}.graph.html")
        written_outside(written, f"diagram.save('{self.name or 'graph'}.graph.html')")
        self._warn_if_stale()
        written.write_text(self.page(cdn=cdn, caption=caption), encoding="utf-8")
        return written

    def text_with_pointer(self) -> str:
        """The text as a Markdown code block (not a diagram) with the pointer line: what to show where nothing draws a
        diagram and nothing may be loaded."""
        return f"```text\n{self}```\n\n*{POINTER}*\n"

    def display(self, *, offline: bool = False, caption: str | None = None) -> GraphPage:
        """The diagram as a notebook output, as ``mime.viz.display`` gives the 3-D view: a :class:`GraphPage`, shown by a
        cell whose last expression it is, with Mermaid from jsDelivr (held to its hash). ``offline=True`` inlines the
        library instead, so the page draws with no network (3.6 MB more; it warns above ``OUTPUT_WARNING_BYTES``, which
        a notebook saves into its file with every display). ``caption`` is a line above the diagram. The text alone is
        ``print(diagram)``."""
        self._warn_if_stale()
        held = self.page(cdn=not offline, caption=caption)
        size = len(held.encode("utf-8"))
        if size > OUTPUT_WARNING_BYTES:
            warn_at_script(
                f"the diagram is {size / 1e6:.1f} MB as a notebook output, and every display saves it into the notebook's "
                "file: leave the library to jsDelivr (offline=False), or write a page instead (diagram.save('graph.html'))",
                UserWarning)
        return GraphPage(held, self.name or "graph", origin=self.origin)

    def _warn_if_stale(self) -> None:
        stale = self.stale
        if stale is not None:
            warn_at_script(stale, UserWarning)

    def _repr_mimebundle_(self, include: object = None, exclude: object = None) -> dict[str, str]:
        return {"text/html": view.frame(self.page(cdn=True), stale=self.stale), "text/plain": str(self)}

    @override
    def __repr__(self) -> str:
        return str.__repr__(self)


def mermaid_of(graph: GraphManager, report: BuildReport, *, direction: str = "LR", origin: Origin | None = None) -> Mermaid:
    """MADDENING's Mermaid for ``graph``, each node's label carrying the registered entry that built it (``report``'s
    node lines say which). ``direction`` is MADDENING's: ``LR``, ``RL``, ``TB`` or ``BT``. ``origin`` is the declarations the
    build came from, for the stale mark."""
    entries: Mapping[str, str] = {_escaped(line.node): _escaped(line.entry) for line in report.nodes}
    drawn: Any = graph.to_mermaid(direction=direction)
    lines: list[str] = []
    for line in str(drawn).splitlines():
        found = _NODE_LINE.match(line)
        if found is not None and found["name"] in entries:
            line = f"{found['head']}{found['name']}<br/>{entries[found['name']]}<br/>{found['type']}{found['tail']}"
        lines.append(line)
    return Mermaid("\n".join(lines) + "\n", origin=origin, name=report.experiment)


__all__ = ["MERMAID_CDN", "MERMAID_SRI", "MERMAID_VERSION", "POINTER", "RENDERER", "GraphPage", "Mermaid", "mermaid_of"]
