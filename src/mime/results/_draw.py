"""The two backends that draw a :class:`~mime.results._plot.PlotDescription`: matplotlib (``plot()``: static, for papers)
and plotly (``plot_interactive()``: cards and quick looks). Each only draws; what is on the figure is the description's.
Both are optional installs (the ``plot`` extra), imported when a figure is drawn and refused with the line that installs
them where absent.

A figure is made without pyplot (``matplotlib.figure.Figure``), so drawing opens no window and touches no global state;
a plotly figure carries a fixed ``uid`` on each trace and no other random id, so that its JSON is the same bytes for the
same description and options."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from mime.build._refusals import ArgumentRefused, Refusal, RefusalClass
from mime.errors._missing import missing_dependency
from mime.results._plot import Panel, PlotDescription

if TYPE_CHECKING:
    from matplotlib.figure import Figure as MatplotlibFigure
    from plotly.graph_objects import Figure as PlotlyFigure

LEGENDS = ("best", "upper right", "upper left", "lower right", "lower left", "center right", "center left", "lower center",
           "upper center", "center", "off")
OPTION_KEYS = frozenset({"title", "x_label", "y_label", "legend", "size", "style"})


@dataclass(frozen=True)
class FigureOptions:
    """What the user sets on a figure beyond what it shows: its ``title``, the ``x_label`` and ``y_label`` of every panel
    (the description's when not given), where the ``legend`` sits (``"off"`` hides it), its ``size`` in inches (width,
    height) and the ``style`` (matplotlib's style name, or plotly's template name). A recipe saves these, so that a
    figure is edited without being run again."""

    title: str | None = None
    x_label: str | None = None
    y_label: str | None = None
    legend: str = "best"
    size: tuple[float, float] = (6.4, 4.8)
    style: str | None = None

    def __post_init__(self) -> None:
        if self.legend not in LEGENDS:
            raise ArgumentRefused([Refusal("plot.option", RefusalClass.USER, f"legend= is one of {', '.join(LEGENDS)}, not {self.legend!r}",
                                           "place the legend with one of them, or legend='off'")])
        if len(self.size) != 2 or not all(v > 0 for v in self.size):
            raise ArgumentRefused([Refusal("plot.option", RefusalClass.USER, f"size= is (width, height) in inches, both positive, not {self.size!r}",
                                           "write size=(6.4, 4.8)")])

    def to_dict(self) -> dict[str, Any]:
        return {"title": self.title, "x_label": self.x_label, "y_label": self.y_label, "legend": self.legend,
                "size": list(self.size), "style": self.style}

    @classmethod
    def from_dict(cls, held: Mapping[str, Any]) -> FigureOptions:
        extra = sorted(set(held) - OPTION_KEYS)
        if extra:
            raise ArgumentRefused([Refusal("plot.option", RefusalClass.USER, f"a figure's options hold {', '.join(map(repr, extra))}, which this format does not "
                                           f"(its keys: {', '.join(sorted(OPTION_KEYS))})", "edit the recipe to the keys it names, or set them with `figure.with_options(legend=\"best\")`")])
        size = held.get("size", (6.4, 4.8))
        return cls(held.get("title"), held.get("x_label"), held.get("y_label"), str(held.get("legend", "best")),
                   (float(size[0]), float(size[1])), held.get("style"))


_BAND = {"std": "± 1 standard deviation", "bound": "± the stated bound"}


def import_optional(module: str) -> Any:
    import importlib

    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as error:
        refusal = missing_dependency(error)
        if refusal is None:
            raise
        raise refusal from error


def draw_matplotlib(description: PlotDescription, options: FigureOptions) -> MatplotlibFigure:
    """The description as a matplotlib figure: a panel under another, the intervals of a flag shaded over the x axis, a
    band shaded over the y axis."""
    import_optional("matplotlib")
    from matplotlib import style as mpl_style
    from matplotlib.figure import Figure

    with mpl_style.context(options.style) if options.style else mpl_style.context({}):
        figure = Figure(figsize=options.size, layout="constrained")
        axes = figure.subplots(len(description.panels), 1, squeeze=False)[:, 0]
        for axis, panel in zip(axes, description.panels, strict=True):
            _panel(axis, panel, options)
        cast(Any, figure).suptitle(options.title if options.title is not None else description.title)
    return figure


def _panel(axis: Any, panel: Panel, options: FigureOptions) -> None:
    for band in panel.bands:
        axis.axhspan(band.low, band.high, color="0.5", alpha=0.12, label=band.label, zorder=0)
    seen: set[str] = set()
    for span in panel.spans:
        axis.axvspan(span.start, span.end, color="C3", alpha=0.18, zorder=0,
                     label=None if span.flag in seen else f"{span.flag} raised")
        seen.add(span.flag)
    for trace in panel.traces:
        if trace.style == "bars":
            axis.bar(trace.x, trace.y, width=trace.width, label=trace.label)
        elif trace.style == "points":
            axis.plot(trace.x, trace.y, "o", label=trace.label)
        else:
            line = axis.plot(trace.x, trace.y, "-o" if trace.style == "line+points" else "-", label=trace.label,
                             markersize=4)[0]
            if trace.marked is not None and trace.marked.any():
                axis.plot(trace.x[trace.marked], trace.y[trace.marked], "o", markerfacecolor="none",
                          markeredgecolor=line.get_color(), markersize=8)
            if trace.amount is not None:
                axis.fill_between(trace.x, trace.y - trace.amount, trace.y + trace.amount, color=line.get_color(),
                                  alpha=0.2, linewidth=0, label=f"{trace.label}: {_BAND[trace.uncertainty or 'bound']}")
    axis.set_xlabel(options.x_label if options.x_label is not None else panel.x_label)
    axis.set_ylabel(options.y_label if options.y_label is not None else panel.y_label)
    if options.legend != "off" and (axis.get_legend_handles_labels()[0]):
        axis.legend(loc=options.legend, fontsize="small")


def draw_plotly(description: PlotDescription, options: FigureOptions) -> PlotlyFigure:
    """The description as a plotly figure, with the same content: a row for each panel, the flags as shaded rectangles
    over the x axis, the bands over the y axis."""
    go = import_optional("plotly.graph_objects")
    subplots = import_optional("plotly.subplots")
    rows = len(description.panels)
    figure = subplots.make_subplots(rows=rows, cols=1, shared_xaxes=False, vertical_spacing=0.12 if rows > 1 else 0.0)
    uid = 0
    for row, panel in enumerate(description.panels, start=1):
        for trace in panel.traces:
            x, y = trace.x.tolist(), trace.y.tolist()
            common = {"name": trace.label, "uid": f"mime-{uid}", "legendgroup": f"panel-{row}", "showlegend": True}
            uid += 1
            if trace.style == "bars":
                figure.add_trace(go.Bar(x=x, y=y, width=trace.width, **common), row=row, col=1)
                continue
            if trace.amount is not None:
                low, high = (trace.y - trace.amount).tolist(), (trace.y + trace.amount).tolist()
                figure.add_trace(go.Scatter(x=x, y=low, mode="lines", line={"width": 0}, showlegend=False, hoverinfo="skip",
                                            uid=f"mime-{uid}"), row=row, col=1)
                figure.add_trace(go.Scatter(
                    x=x, y=high, mode="lines", line={"width": 0}, fill="tonexty", fillcolor="rgba(120,120,120,0.25)",
                    name=f"{trace.label}: {_BAND[trace.uncertainty or 'bound']}", uid=f"mime-{uid + 1}", hoverinfo="skip"),
                    row=row, col=1)
                uid += 2
            mode = {"line": "lines", "points": "markers", "line+points": "lines+markers"}[trace.style]
            scatter = go.Scatter(x=x, y=y, mode=mode, **common)
            figure.add_trace(scatter, row=row, col=1)
            if trace.marked is not None and trace.marked.any():
                figure.add_trace(go.Scatter(
                    x=trace.x[trace.marked].tolist(), y=trace.y[trace.marked].tolist(), mode="markers",
                    marker={"symbol": "circle-open", "size": 11}, name=f"{trace.label}: a flag raised",
                    uid=f"mime-{uid}", showlegend=False), row=row, col=1)
                uid += 1
        for band in panel.bands:
            figure.add_hrect(y0=band.low, y1=band.high, fillcolor="gray", opacity=0.12, line_width=0, row=row, col=1,
                             annotation_text=band.label, annotation_position="top left")
        for span in panel.spans:
            figure.add_vrect(x0=span.start, x1=span.end, fillcolor="red", opacity=0.15, line_width=0, row=row, col=1,
                             annotation_text=span.flag, annotation_position="top left")
        figure.update_xaxes(title_text=options.x_label if options.x_label is not None else panel.x_label, row=row, col=1)
        figure.update_yaxes(title_text=options.y_label if options.y_label is not None else panel.y_label, row=row, col=1)
    width, height = (round(v * 96) for v in options.size)
    legend: dict[str, Any] = {"visible": options.legend != "off"}
    figure.update_layout(title_text=options.title if options.title is not None else description.title, width=width,
                         height=height * rows, showlegend=options.legend != "off", legend=legend,
                         **({"template": options.style} if options.style else {}))
    return cast("PlotlyFigure", figure)


def plotly_json(figure: PlotlyFigure) -> str:
    """The figure's own JSON (plotly's data and layout, safe to share and to edit), the same bytes for the same figure."""
    return str(cast(Any, figure).to_json(pretty=True, remove_uids=False)) + "\n"


__all__ = ["LEGENDS", "OPTION_KEYS", "FigureOptions", "draw_matplotlib", "draw_plotly", "import_optional", "plotly_json"]
