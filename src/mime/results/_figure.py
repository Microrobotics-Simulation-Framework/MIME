"""A figure of a result as a recipe: what it shows (:class:`FigureSpec`), how it is set (:class:`FigureOptions`: the
user's title, labels, legend, size and style) and the data files it reads, each by path and sha256. A recipe is saved
beside the run or study (``figures/<name>.json``), and redrawn from the folder's data, never by running the experiment
again; the data is checked against the recipe's sha256 first, and a file that changed refuses the redraw by its name.

Plotly's own JSON is saved beside it (``figures/<name>.plotly.json``: data and layout, safe to share, editable), the
same bytes each time for the same data. A matplotlib pickle is never written: matplotlib does not promise pickles across
versions, and unpickling runs code, so nothing a ``pack`` or a ``push`` carries is one.

A page of a figure (``display``, ``web``) follows the 3-D view's and the graph view's names and behaviour
(``mime.viz._web``, ``mime.experiment._mermaid_page``): ``offline=True`` inlines the library, ``cdn=`` on a file, a
notebook loads it from jsDelivr held to its hash by default, the same stale banner and frame height, and a default file
name (``<experiment>.<name>.figure.html``) that is never another view's."""

from __future__ import annotations

import base64
import hashlib
import html
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, override

from mime._lineage import Origin
from mime.build._refusals import ArgumentRefused, Refusal, RefusalClass
from mime.errors._frames import warn_at_script
from mime.results._draw import FigureOptions, draw_matplotlib, draw_plotly, plotly_json
from mime.results._study_folder import FIGURES_DIR as FIGURES
from mime.results._plot import KINDS, PlotDescription, describe_run, describe_study, refuse

if TYPE_CHECKING:
    from matplotlib.figure import Figure as MatplotlibFigure
    from plotly.graph_objects import Figure as PlotlyFigure

    from mime.results._data import ResultData, StudyData
    from mime.results._result import RunResult
    from mime.results._study import StudyResult

type Target = RunResult | StudyResult | ResultData | StudyData

RECIPE_FORMAT = "mime-figure"
RECIPE_VERSION = 1
RECIPE_KEYS = frozenset({"format", "version", "name", "target", "spec", "options", "reads", "description", "reading"})
SPEC_KEYS = frozenset({"kind", "outputs", "against", "members"})
READING = ("the figure is redrawn from the data files in reads (each by path, relative to the folder this recipe sits "
           "in, and sha256), never by running the experiment again; options are the user's overrides of the "
           "description's own titles and labels; description says what the figure shows, without its data; "
           "figures/<name>.plotly.json is plotly's own JSON of the same figure")

#: plotly.js, pinned: the version the installed ``plotly`` ships (the bundle ``offline=True`` inlines) and the sha384, as
#: SRI writes it, that jsDelivr's copy of the same file is held to. To check a bump: ``curl -sL <PLOTLY_CDN> | openssl
#: dgst -sha384 -binary | openssl base64 -A``; the bytes are the file the plotly package ships.
PLOTLY_JS_VERSION = "4.1.1"
PLOTLY_JS_SHA256 = "3b6e15d45dbb7fca5bd2094291e961ddc5472cd887009e6009a56dab668d721f"
PLOTLY_JS_SRI = "sha384-AFNp2MtSm5/oZbEs/J19F/Ah57MEiVsoed8nE3fq2Wnccdl1u/EkyJ9XKKU8rt7+"
PLOTLY_CDN = f"https://cdn.jsdelivr.net/npm/plotly.js-dist-min@{PLOTLY_JS_VERSION}/plotly.min.js"
#: The frame's height in a notebook, in pixels, and the size past which a displayed page warns: the 3-D view's.
FRAME_HEIGHT = 720
OUTPUT_WARNING_BYTES = 1_000_000


@dataclass(frozen=True)
class FigureSpec:
    """What a figure shows of a result: ``kind`` (``series`` for a run; ``curve``, ``overlay`` or ``distribution`` for a
    study), the ``outputs`` it draws (``<observation>.<output>``), the axis it is drawn ``against`` (a curve) and the
    ``members`` it draws (an overlay; all when ``None``)."""

    kind: str
    outputs: tuple[str, ...]
    against: str | None = None
    members: tuple[int, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "outputs": list(self.outputs), "against": self.against,
                "members": None if self.members is None else list(self.members)}

    @classmethod
    def from_dict(cls, held: Mapping[str, Any]) -> FigureSpec:
        if set(held) != SPEC_KEYS or held["kind"] not in KINDS:
            raise refuse("plot.recipe", f"a recipe's spec holds exactly {', '.join(sorted(SPEC_KEYS))} and a kind of "
                         f"{', '.join(KINDS)}", "write the recipe with result.figure(...).write_recipe()")
        members = held["members"]
        return cls(str(held["kind"]), tuple(map(str, held["outputs"])), held["against"],
                   None if members is None else tuple(int(m) for m in members))


def is_study(result: object) -> bool:
    return hasattr(result, "member_data")


def describe(result: Target, spec: FigureSpec, title: str | None = None) -> PlotDescription:
    """The figure ``spec`` of ``result``, read through the result protocol (``mime.results._data``): a run, a study, or a
    result built by hand (``Measured``)."""
    from mime.results._data import RunData
    from mime.results._result import RunResult

    if is_study(result):
        return describe_study(cast("StudyData", result), spec.outputs, kind=spec.kind, against=spec.against,
                              members=spec.members, title=title)
    if spec.kind != "series":
        raise refuse("plot.kind", f"a run's plot is a 'series', not {spec.kind!r}", "plot(...) with no kind=")
    data = RunData(result) if isinstance(result, RunResult) else cast("ResultData", result)
    return describe_run(data, spec.outputs, title=title)


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def reads_of(result: Target) -> tuple[Path, list[str]]:
    """The folder a result was read from and the data files a figure of it reads (relative to the folder)."""
    folder = getattr(result, "folder", None)
    if folder is None:
        raise refuse("plot.recipe-folder", "this result is in memory, and a recipe names the files it reads in a folder",
                     "write it to a folder first (a run: run_to; a study: study.run_to(folder)), and read it from there")
    if not is_study(result):
        return folder, ["outputs.npz"]
    return folder, [f"groups/{k}/outputs.npz" for k in sorted(set(cast("StudyData", result).groups))]


class FigureRecipe:
    """A figure of a result: ``plot()`` draws it with matplotlib, ``plot_interactive()`` with plotly, ``display()`` and
    ``web()`` as a page; ``with_options(...)`` sets the title, labels, legend, size or style; ``write_recipe()`` saves it
    beside the result's folder, and ``FigureRecipe.read(folder, name)`` redraws it from there."""

    def __init__(self, result: Target, spec: FigureSpec, options: FigureOptions | None = None,
                 name: str | None = None) -> None:
        self.result = result
        self.spec = spec
        self.options = options or FigureOptions()
        self.name = name
        self._description: PlotDescription | None = None

    @property
    def description(self) -> PlotDescription:
        """What the figure shows, backend-neutral: its panels, traces, flags as intervals and validity bands."""
        if self._description is None:
            self._description = describe(self.result, self.spec)
        return self._description

    def with_options(self, **changes: Any) -> FigureRecipe:
        """The same figure with ``title``, ``x_label``, ``y_label``, ``legend``, ``size`` or ``style`` set."""
        held = {**self.options.to_dict(), **changes}
        return FigureRecipe(self.result, self.spec, FigureOptions.from_dict(held), self.name)

    def plot(self) -> MatplotlibFigure:
        """The figure drawn with matplotlib (the ``plot`` extra)."""
        return draw_matplotlib(self.description, self.options)

    def plot_interactive(self) -> PlotlyFigure:
        """The figure drawn with plotly (the ``plot`` extra)."""
        return draw_plotly(self.description, self.options)

    def plotly_json(self) -> str:
        return plotly_json(self.plot_interactive())

    # -- the recipe, in the folder ----------------------------------------------------------------------------

    def document(self, name: str) -> dict[str, Any]:
        folder, reads = reads_of(self.result)
        return {"format": RECIPE_FORMAT, "version": RECIPE_VERSION, "name": name,
                "target": "study" if is_study(self.result) else "run",
                "spec": self.spec.to_dict(), "options": self.options.to_dict(),
                "reads": [{"path": path, "sha256": file_digest(folder / path)} for path in reads],
                "description": self.description.summary(), "reading": READING}

    def write_recipe(self, name: str | None = None) -> Path:
        """Save the recipe as ``figures/<name>.json`` in the result's folder, and plotly's own JSON of the figure beside
        it (``<name>.plotly.json``)."""
        held = name or self.name
        if not held or "/" in held or held.startswith("."):
            raise refuse("plot.recipe", f"a figure's name is a file name without folders, not {held!r}",
                         "name it: result.figure(..., name='step_out')")
        folder, _ = reads_of(self.result)
        document = self.document(held)
        target = folder / FIGURES
        target.mkdir(exist_ok=True)
        text = self.plotly_json()
        (target / f"{held}.plotly.json").write_text(text, encoding="utf-8", newline="\n")
        written = target / f"{held}.json"
        written.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        self.name = held
        return written

    @classmethod
    def read(cls, folder: str | Path, name: str) -> FigureRecipe:
        """Redraw the figure ``name`` of the run or study in ``folder`` from its data, which is checked against the
        recipe's sha256 first (``plot.data-changed`` names a file that is not the one the figure was made from)."""
        root = Path(folder)
        path = root / FIGURES / f"{name}.json"
        try:
            document = cast(object, json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError) as error:
            raise refuse("plot.recipe", f"{path} is not a recipe that can be read ({error})",
                         "write one with result.figure(...).write_recipe()") from error
        if not isinstance(document, dict) or cast(dict[str, Any], document).get("format") != RECIPE_FORMAT \
                or cast(dict[str, Any], document).get("version") != RECIPE_VERSION:
            raise refuse("plot.recipe", f"{path} is not a {RECIPE_FORMAT!r} version {RECIPE_VERSION} recipe: this MIME "
                         f"reads that one", "read it with the MIME that wrote it, or write the figure again")
        held = cast(dict[str, Any], document)
        if set(held) != RECIPE_KEYS:
            raise refuse("plot.recipe", f"{path} holds {sorted(set(held) ^ RECIPE_KEYS)}, which a recipe's closed keys do not",
                         "write the recipe with result.figure(...).write_recipe()")
        for read in cast(list[dict[str, str]], held["reads"]):
            data = root / read["path"]
            if not data.is_file() or file_digest(data) != read["sha256"]:
                raise ArgumentRefused([Refusal(
                    "plot.data-changed", RefusalClass.USER,
                    f"{data} is not the file the figure {name!r} was made from"
                    + (" (it is not there)" if not data.is_file() else f" (its sha256 is {file_digest(data)}, and the recipe records {read['sha256']})"),
                    "redraw from the folder the figure was made in, or make the figure again from this data: "
                    "result.figure(...).write_recipe()",
                    why="a figure is redrawn from the data it was made from, never from data that changed")])
        from mime.results._result import RunResult
        from mime.results._study import StudyResult

        result = RunResult.read(root) if held["target"] == "run" else StudyResult.read(root)
        return cls(result, FigureSpec.from_dict(held["spec"]), FigureOptions.from_dict(held["options"]), name)

    # -- a page ---------------------------------------------------------------------------------------------

    def _origin(self) -> Origin | None:
        from mime.results._result import RunResult

        if isinstance(self.result, RunResult):
            return self.result.origin
        if is_study(self.result) and len(cast("StudyResult", self.result)):
            return cast("StudyResult", self.result).member(0).origin
        return None

    def _stale(self) -> str | None:
        origin = self._origin()
        return origin.stale("this figure", "it shows a run of the declarations before: build and run again") \
            if origin is not None else None

    def page(self, *, cdn: bool = False, caption: str | None = None) -> str:
        """The figure as one HTML page: plotly.js inlined from the installed plotly (no network), or from jsDelivr held to
        its hash with ``cdn=True`` (4.8 MB smaller). The figure's div id is fixed, so the same figure is the same page."""
        stale = self._stale()
        return figure_page(self.plotly_json(), self.description.title, cdn=cdn,
                           caption=f"STALE: {stale}. {caption}" if stale and caption else (f"STALE: {stale}" if stale else caption),
                           name=self.name or "figure")

    def display(self, *, offline: bool = False, caption: str | None = None) -> FigurePage:
        """The figure as a notebook output, as ``mime.viz.display`` gives the 3-D view: a :class:`FigurePage` a cell shows
        when it is its last expression, plotly.js from jsDelivr (held to its hash); ``offline=True`` inlines it, 4.8 MB
        more. Warns above 1 MB, which a notebook saves into its file with every display."""
        stale = self._stale()
        if stale is not None:
            warn_at_script(stale, UserWarning)
        held = self.page(cdn=not offline, caption=caption)
        if len(held.encode("utf-8")) > OUTPUT_WARNING_BYTES:
            warn_at_script(f"the figure is {len(held) / 1e6:.1f} MB as a notebook output, and every display saves it into "
                           f"the notebook's file: leave plotly.js to jsDelivr (offline=False), or write a page instead "
                           f"(figure.web())", UserWarning)
        return FigurePage(held, self.name or "figure", origin=self._origin())

    def web(self, out: str | Path | None = None, *, cdn: bool = False, caption: str | None = None) -> Path:
        """Write the page to ``out`` (``<experiment>.<name>.figure.html`` in the working folder by default: never another
        view's name), plotly.js inlined unless ``cdn``. Never inside a saved or run folder (``visuals.web-out``)."""
        from mime.experiment._mermaid import written_outside

        written = Path(out) if out is not None else Path(f"{self.description.title.partition(':')[0]}.{self.name or 'figure'}.figure.html")
        written_outside(written, f"figure.web('{self.name or 'figure'}.figure.html')")
        stale = self._stale()
        if stale is not None:
            warn_at_script(stale, UserWarning)
        written.write_text(self.page(cdn=cdn, caption=caption), encoding="utf-8")
        return written


@dataclass(frozen=True)
class FigurePage:
    """A figure as a notebook shows it: the page in an ``<iframe srcdoc>``, as the 3-D view's ``WebPage`` and the graph's
    ``GraphPage`` are. A cell whose last expression it is shows it; :meth:`save` writes the page. Its ``repr`` is one
    line. ``height`` is the frame's, in pixels."""

    html: str
    name: str
    height: int = FRAME_HEIGHT
    origin: Origin | None = None

    @property
    def stale(self) -> str | None:
        return self.origin.stale("this figure", "it shows a run of the declarations before: build and run again") \
            if self.origin is not None else None

    @override
    def __repr__(self) -> str:
        stale = self.stale
        return (f"<FigurePage {self.name!r}, {len(self.html.encode('utf-8')) / 1e3:.0f} kB: a notebook cell shows it when it "
                f"is the cell's last expression; elsewhere .save('figure.html') writes the page, to open in a browser"
                + (f"; STALE, {stale}" if stale is not None else "") + ">")

    def _repr_html_(self) -> str:
        banner = f'<p style="color: #b3261e; font-weight: bold">STALE: {html.escape(self.stale)}</p>' if self.stale else ""
        return (banner + f'<iframe srcdoc="{html.escape(self.html, quote=True)}" sandbox="allow-scripts" '
                f'style="width: 100%; height: {self.height}px; border: 0" title="{html.escape(self.name)}"></iframe>')

    def save(self, path: str | Path) -> Path:
        written = Path(path)
        written.write_text(self.html, encoding="utf-8")
        return written


def library_tag(cdn: bool) -> str:
    """The script element that loads plotly.js: a data URI of the installed plotly's bundle, held to the sha256 of the
    version this MIME pins, or jsDelivr's copy of that file held to its sha384."""
    if cdn:
        return f'<script src="{PLOTLY_CDN}" integrity="{PLOTLY_JS_SRI}" crossorigin="anonymous"></script>'
    from mime.results._draw import _import

    data = _import("plotly.offline").get_plotlyjs().encode("utf-8")
    if hashlib.sha256(data).hexdigest() != PLOTLY_JS_SHA256:
        raise refuse("plot.library", f"the installed plotly ships a plotly.js that is not the {PLOTLY_JS_VERSION} this MIME "
                     f"pins, so a page cannot inline it held to a hash",
                     "write the page with cdn=True (jsDelivr's copy, held to its hash), or install the plotly the lock pins")
    return f'<script src="data:text/javascript;base64,{base64.b64encode(data).decode("ascii")}"></script>'


def figure_page(figure_json: str, title: str, *, cdn: bool, caption: str | None, name: str) -> str:
    """One self-contained page for a plotly figure's JSON, with a fixed id for the figure's element."""
    safe = figure_json.replace("<", "\\u003c")
    div = f"mime-figure-{''.join(c if c.isalnum() else '-' for c in name)}"
    where = ("plotly.js is loaded from jsDelivr, held to its hash; the figure is inside this page." if cdn else
             "plotly.js and the figure are both inside this page: it needs no network.")
    lines = ["<!doctype html>", '<html lang="en"><head><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width, initial-scale=1">', f"<title>{html.escape(title)}</title>",
             "<style>:root { --bg: #fff; --fg: #1f2328; } @media (prefers-color-scheme: dark) { :root:not([data-theme=\"light\"]) "
             "{ --bg: #0d1117; --fg: #e6edf3; } } :root[data-theme=\"dark\"] { --bg: #0d1117; --fg: #e6edf3; } "
             "body { margin: 0; background: var(--bg); color: var(--fg); font: 14px/1.5 system-ui, sans-serif; } "
             "p { margin: 4px 12px; font-size: 12px; opacity: .75; }</style>", "</head><body>"]
    if caption:
        lines.append(f"<p>{html.escape(caption)}</p>")
    lines += [f'<div id="{div}"></div>', f'<script type="application/json" id="{div}-data">{safe}</script>',
              library_tag(cdn),
              f'<script>(function () {{ var f = JSON.parse(document.getElementById("{div}-data").textContent); '
              f'Plotly.newPlot("{div}", f.data, f.layout, {{responsive: true}}); }})();</script>',
              f"<p>{html.escape(where)}</p>", "</body></html>"]
    return "\n".join(lines) + "\n"


def figure(result: Target, *outputs: object, kind: str | None = None, against: str | None = None,
           members: Sequence[int] | None = None, name: str | None = None, **options: Any) -> FigureRecipe:
    """A figure of any result that reads as the protocol (``mime.results.Measured``, a simulated run or study): the same
    as ``result.figure(...)`` of a run or a study."""
    return figure_of(result, outputs, kind=kind, against=against, members=members, name=name, options=options)


def figure_of(result: Target, outputs: Sequence[object], *, kind: str | None, against: str | None,
              members: Sequence[int] | None, name: str | None, options: Mapping[str, Any]) -> FigureRecipe:
    """The figure ``outputs`` of ``result`` as a recipe (what ``result.figure`` returns)."""
    paths = tuple(str(getattr(each, "path", each)) for each in outputs)
    if not is_study(result):
        if kind not in (None, "series") or against is not None or members is not None:
            raise refuse("plot.kind", "a run's plot is its recorded series against time: kind=, against= and members= "
                         "are for a study's", "plot(...) with the outputs alone")
        spec = FigureSpec("series", paths)
    else:
        from mime.results._plot import _kind_of  # the default kind is by the output

        if len(paths) == 1 and kind is None:
            kind = "overlay" if _kind_of(cast("StudyData", result), paths[0]) == "per_step" else "curve"
        spec = FigureSpec(kind or "curve", paths, against, None if members is None else tuple(members))
    return FigureRecipe(result, spec, FigureOptions.from_dict({**FigureOptions().to_dict(), **options}), name)


__all__ = ["FIGURES", "PLOTLY_CDN", "PLOTLY_JS_SHA256", "PLOTLY_JS_SRI", "PLOTLY_JS_VERSION", "RECIPE_FORMAT",
           "RECIPE_VERSION", "FigurePage", "FigureRecipe", "FigureSpec", "describe", "figure", "figure_of", "figure_page",
           "library_tag"]
