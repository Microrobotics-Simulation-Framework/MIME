"""The node page: what the physics does. The equations have one home, a node's ``governing_equations``, and this page
prints them; the symbol table joins each symbol to the code. A node with no ``NodeMeta`` gets a page that says it is
undocumented and is counted, never an empty one.

Not rendered at this phase: the state, parameter, input and output tables, which come from a built node, and the
performance profile and the claims that cite the node, whose records do not exist yet. The symbol table is the checked
join of names, units and code in the meantime."""

from __future__ import annotations

from maddening.core.compliance.metadata import NodeMeta

from mime.core import Symbol
from mime.docs._context import Context, entry_path, node_path
from mime.docs._markdown import Page, Writer, code, link
from mime.docs._symbols import meta_of, rows_of

NONE_DECLARED = "none declared"
UNDOCUMENTED = "undocumented: this node class declares no `NodeMeta`, so there is nothing to print but where it is built."


def _meta_text(w: Writer, meta: NodeMeta, name: str) -> None:
    w.paragraph(f"Algorithm {code(meta.algorithm_id or 'unset')}.")
    if meta.description:
        w.paragraph(w.prose("description", meta.description))
    w.heading(2, "Governing equations")
    if meta.governing_equations.strip():
        w.add("$$", meta.governing_equations.strip(), "$$", "")
    else:
        w.paragraph(NONE_DECLARED)
        w.hatch(NONE_DECLARED, f"node {name} equations")
    w.heading(2, "Discretisation")
    if meta.discretization:
        w.paragraph(w.prose("discretisation", meta.discretization))
    order = meta.discretization_order
    if order is not None:
        parts = [f"{label} {value:g}" for label, value in (("spatial", order.spatial), ("temporal", order.temporal))
                 if value is not None]
        w.paragraph("Claimed order: " + (", ".join(parts) or "none") + (
            f". {w.prose('order notes', order.notes)}" if order.notes else "") + " (a claim, not a measurement).")
    if not meta.discretization and order is None:
        w.paragraph("none declared; a node whose order depends on its construction declares it with the "
                    "`discretization_order()` hook, which is read from a built node.")
    for title, field_name, held in (("Assumptions", "assumptions", meta.assumptions),
                                    ("Limitations", "limitations", meta.limitations)):
        w.heading(2, title)
        if held:
            w.bullets(w.prose(f"{field_name} {number}", text) for number, text in enumerate(held, start=1))
        else:
            w.paragraph(NONE_DECLARED)
            if field_name == "assumptions":
                w.hatch(NONE_DECLARED, f"node {name} assumptions")
    w.heading(2, "Validated regimes")
    if meta.validated_regimes:
        w.table(["Parameter", "Lowest", "Highest", "Units", "Notes"],
                [[code(regime.parameter), _bound(regime.min_value), _bound(regime.max_value), regime.units,
                  w.prose(f"regime {regime.parameter}", regime.notes)] for regime in meta.validated_regimes])
    else:
        w.paragraph(NONE_DECLARED)
    w.heading(2, "References")
    if meta.references:
        w.paragraph(", ".join(f"[@{ref.key}]" for ref in meta.references))
    else:
        w.paragraph(NONE_DECLARED)
        w.hatch(NONE_DECLARED, f"node {name} references")


def _bound(value: float | None) -> str:
    return "unbounded" if value is None else f"{value:g}"


def _symbols(w: Writer, rows: tuple[Symbol, ...]) -> None:
    w.heading(2, "Symbols")
    if not rows:
        w.paragraph("No symbol table: the page cannot say which code each symbol is.")
        return
    w.table(["Symbol", "Kind", "Name", "Units", "Implemented by"],
            [[f"${row.latex}$", row.kind.value, code(row.name), row.units, code(row.implemented_by)]
             for row in rows])


def node_page(node: type, dotted: str, context: Context) -> Page:
    """The page of the node class ``node`` (dotted path ``dotted``): its meta and symbol table, and the entries that
    build it."""
    name = node.__name__
    path = node_path(name)
    w = Writer()
    w.heading(1, f"Node {name}")
    w.paragraph(f"Defined in {code(dotted)}.")
    meta = meta_of(node)
    if meta is None:
        w.paragraph(UNDOCUMENTED)
        w.hatch("undocumented", f"node {name}")
    else:
        _meta_text(w, meta, name)
        _symbols(w, rows_of(node))
    w.heading(2, "Built by")
    builders = context.builders(dotted)
    if builders:
        w.bullets(f"{link(path, entry_path(record), code(record['name']))} {record['version']}"
                  + (f", rung {code(rung)}" if rung else "") for record, rung in builders)
    else:
        w.paragraph("No registered entry builds it.")
    hand = context.hand.get(path)
    if hand is not None:
        w.heading(2, "Derivation notes")
        w.add(hand.strip("\n"), "")
    return w.page(path)


__all__ = ["NONE_DECLARED", "UNDOCUMENTED", "node_page"]
