"""The checks on a node's ``NodeMeta`` and symbol table. They bound names, units and existence: that each row's
symbol is in the equations and each equation symbol has a row, that each row's name and units are what the code
declares, and that each row names a function that exists where the row says. Whether the code implements the equation
is the oracle's to show, not theirs.

Tokens are read from LaTeX macros first, then single letters, outside ``\\text{}``, ``\\mathrm{}`` and the other
macros whose argument is words; membership is by token, never by substring. Letters inside a braced subscript
(``x_{i}``) are indices and need no row; an index outside one (``x_i``, a bare ``i`` in a sum) is listed with its
reason, as is everything the checks would otherwise fail on."""

from __future__ import annotations

import importlib
import re
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import yaml
from maddening.compliance._validate import resolve_dotted_name
from maddening.core.compliance.metadata import NodeMeta
from maddening.core.node import SimulationNode

from mime.core import Symbol, SymbolKind
from mime.docs._lists import RatchetList, floor_problems, list_problems
from mime.registry import DimensionedSetting
from mime.units import UnitError, parse_unit, unit_symbol

#: Macros whose argument is words, not symbols.
_WORDS = frozenset({"text", "mathrm", "operatorname", "textrm", "textit", "textbf", "mbox", "begin", "end", "label"})
#: Macros that shape or join symbols, and ``\\pi``, which no row binds: none is a symbol. A macro not here is one.
STRUCTURE = frozenset({
    "frac", "dfrac", "tfrac", "sqrt", "mathbf", "boldsymbol", "bm", "mathit", "mathcal", "mathbb", "mathsf",
    "mathscr", "vec", "hat", "dot", "ddot", "tilde", "bar", "overline", "underline", "widehat", "widetilde",
    "left", "right", "big", "Big", "bigg", "Bigg", "bigl", "bigr", "Bigl", "Bigr", "middle", "cdot", "cdots",
    "ldots", "dots", "vdots", "ddots", "times", "otimes", "oplus", "partial", "nabla", "sum", "prod", "int", "iint",
    "oint", "lim", "exp", "log", "ln", "sin", "cos", "tan", "sinh", "cosh", "tanh", "arcsin", "arccos", "arctan",
    "min", "max", "sup", "inf", "det", "tr", "sgn", "quad", "qquad", "approx", "sim", "simeq", "propto", "equiv",
    "le", "leq", "ge", "geq", "neq", "ne", "ll", "gg", "pm", "mp", "to", "rightarrow", "leftarrow", "Rightarrow",
    "Leftarrow", "leftrightarrow", "mapsto", "in", "notin", "subset", "subseteq", "cup", "cap", "infty", "top",
    "intercal", "dagger", "prime", "circ", "langle", "rangle", "lVert", "rVert", "lvert", "rvert", "Vert", "vert",
    "mid", "forall", "exists", "displaystyle", "textstyle", "nonumber", "notag", "hspace", "vspace", "underbrace",
    "overbrace", "stackrel", "pi",
})
_MACRO = re.compile(r"\\([A-Za-z]+)")
_NODE_ID = re.compile(r"MIME-NODE-\d{3}")


@dataclass(frozen=True)
class Token:
    """A symbol-like token: its text (``\\mu``, ``x``), whether it lies inside a braced subscript, and whether it is
    the single token an unbraced ``_`` takes (the ``i`` of ``x_i``)."""

    text: str
    subscript: bool
    unbraced: bool

    @property
    def symbol(self) -> bool:
        return not self.text.startswith("\\") or self.text[1:] not in STRUCTURE


def _skip_group(latex: str, at: int) -> int:
    while at < len(latex) and latex[at].isspace():
        at += 1
    if at >= len(latex) or latex[at] != "{":
        return at
    depth = 0
    while at < len(latex):
        depth += {"{": 1, "}": -1}.get(latex[at], 0)
        at += 1
        if depth == 0:
            break
    return at


def tokens(latex: str) -> list[Token]:
    """The tokens of ``latex``, in order: macros and single letters, words removed."""
    found: list[Token] = []
    braces: list[bool] = []
    script: str | None = None
    at = 0
    while at < len(latex):
        char = latex[at]
        inside = bool(braces) and braces[-1]
        if char == "\\":
            macro = _MACRO.match(latex, at)
            if macro is None:
                at += 2
                continue
            at = macro.end()
            if macro.group(1) in _WORDS:
                at, script = _skip_group(latex, at), None
                continue
            found.append(Token(macro.group(0), inside, script == "_"))
            script = None
        elif char == "{":
            braces.append(script == "_" or inside)
            script = None
            at += 1
        elif char == "}":
            if braces:
                braces.pop()
            at += 1
        elif char in "_^":
            script = char
            at += 1
        elif char.isascii() and char.isalpha():
            found.append(Token(char, inside, script == "_"))
            script = None
            at += 1
        else:
            if not char.isspace():
                script = None
            at += 1
    return found


def symbols_of(latex: str) -> list[Token]:
    return [token for token in tokens(latex) if token.symbol]


def row_key(node: type, row: Symbol) -> str:
    """A row's item in the lists: the node class's name and the row's LaTeX, without whitespace."""
    return f"{node.__name__} {''.join(row.latex.split())}"


def resolve_dotted(dotted: str) -> object:
    parts = dotted.split(".")
    for end in range(len(parts), 0, -1):
        try:
            held: object = importlib.import_module(".".join(parts[:end]))
        except ImportError:
            continue
        for name in parts[end:]:
            held = getattr(held, name)
        return held
    raise ImportError(f"no module in {dotted!r} imports")


def path_problem(node: type, path: str) -> str | None:
    """Why ``implemented_by`` is not a function, method or property defined where ``path`` names it: in a module, or
    in the own ``__dict__`` of a class that is ``node`` or one of its bases. ``None`` when it is."""
    found = resolve_dotted_name(path, require_own=True, require_routine=True)
    if not found.ok:
        return found.reason
    owner = resolve_dotted(path.rsplit(".", 1)[0])
    if isinstance(owner, type) and owner not in node.__mro__:
        return (f"{path!r} names a method of {owner.__qualname__}, which is neither {node.__name__} nor one of its "
                f"bases")
    if not isinstance(owner, (type, ModuleType)):
        return f"{path!r} names an attribute of a {type(owner).__name__}, not of a module or a class"
    return None


def meta_of(node: type) -> NodeMeta | None:
    held = getattr(node, "meta", None)
    return held if isinstance(held, NodeMeta) else None


def rows_of(node: type) -> tuple[Symbol, ...]:
    return cast(tuple[Symbol, ...], getattr(node, "symbols", ()))


@dataclass(frozen=True)
class FrozenId:
    """A 0.2 algorithm id's verdict: the node class that held it, and the dotted path of the 0.3 class that keeps it,
    ``None`` where it is retired (its node changed, or has no 0.3 counterpart); ``why`` is what changed, or the
    evidence that the keeping class is the same node."""

    node: str
    kept_by: str | None
    why: str


def read_frozen_ids(path: Path) -> dict[str, FrozenId]:
    """The 0.2 line's algorithm ids and their verdicts. A row is ``retired`` with what changed, or ``kept_by`` a
    class's dotted path with its ``evidence``; a row with both, or neither, raises ``ValueError``."""
    held = cast(dict[str, Any], yaml.safe_load(path.read_text(encoding="utf-8")))
    frozen: dict[str, FrozenId] = {}
    for row in cast(list[dict[str, Any]], held["node_ids"]):
        retired, kept_by, evidence = row.get("retired"), row.get("kept_by"), row.get("evidence")
        if (retired is None) == (kept_by is None) or not str(evidence if kept_by is not None else retired).strip():
            raise ValueError(f"{path}: {row.get('id')} is either retired, saying what changed, or kept_by a class, "
                             f"with its evidence")
        frozen[str(row["id"])] = FrozenId(str(row["node"]), None if kept_by is None else str(kept_by),
                                          str(retired if kept_by is None else evidence))
    return frozen


def _keeps(path: str, node: type) -> bool:
    try:
        return resolve_dotted(path) is node
    except (ImportError, AttributeError):
        return False


def next_id(frozen: Collection[str], used: Collection[str]) -> str:
    """The next free algorithm id: one past every id of the frozen list and every id in ``used``."""
    numbers = [int(i.rsplit("-", 1)[1]) for i in [*frozen, *used] if _NODE_ID.fullmatch(i)]
    return f"MIME-NODE-{max(numbers, default=0) + 1:03d}"


def table_problems(node: type, frozen: Mapping[str, FrozenId], used: Collection[str] = ()) -> list[str]:
    """What is wrong with ``node``'s ``NodeMeta`` and its table, read from the class: rows whose symbols are not
    in the equations, rows whose ``implemented_by`` is not where it says, a missing algorithm id, a 0.2 id taken by
    any class but the one that keeps it, ``hazard_hints`` set, and an order declared twice. The next id it names is
    past the frozen list's and the ids in ``used``, the walked nodes'."""
    meta, rows, name = meta_of(node), rows_of(node), node.__name__
    if meta is None:
        return [f"{name}: it has a symbol table and no NodeMeta, so no equations for the rows to name"] if rows else []
    found: list[str] = []
    if not meta.algorithm_id:
        found.append(f"{name}: NodeMeta.algorithm_id is empty; give it the next id, "
                     f"{next_id(frozen, used)}")
    elif node.__module__.split(".")[0] == "mime" and not _NODE_ID.fullmatch(meta.algorithm_id):
        found.append(f"{name}: algorithm_id {meta.algorithm_id!r} is not a MIME-NODE id")
    elif meta.algorithm_id in frozen:
        held = frozen[meta.algorithm_id]
        if held.kept_by is None:
            found.append(f"{name}: algorithm_id {meta.algorithm_id} is retired (the 0.2 line's {held.node}: "
                         f"{held.why}); an id names one node for good, so give it the next id, {next_id(frozen, used)}")
        elif not _keeps(held.kept_by, node):
            found.append(f"{name}: algorithm_id {meta.algorithm_id} is kept by {held.kept_by} (the 0.2 line's "
                         f"{held.node}, found the same), and {node.__module__}.{node.__qualname__} is not that class; "
                         f"give it the next id, {next_id(frozen, used)}")
    if meta.hazard_hints:
        found.append(f"{name}: NodeMeta.hazard_hints is set ({list(meta.hazard_hints)}); MIME's nodes leave it empty")
    if meta.discretization_order is not None and callable(getattr(node, "discretization_order", None)):
        found.append(f"{name}: the order is declared by its discretization_order() and again in NodeMeta; keep the "
                     f"method's, and leave NodeMeta.discretization_order None")
    equation = {token.text for token in symbols_of(meta.governing_equations)}
    for row in rows:
        own = symbols_of(row.latex)
        if not own:
            found.append(f"{name}: the row {row.latex!r} holds no symbol")
        for token in own:
            if token.text not in equation:
                found.append(f"{name}: the row {row.latex!r}: {token.text} does not occur in governing_equations")
        problem = path_problem(node, row.implemented_by)
        if problem is not None:
            found.append(f"{name}: the row {row.latex!r}'s implemented_by: {problem}")
    return found


def unrowed(node: type) -> set[str]:
    """Each symbol of ``node``'s equations, outside a braced subscript, that no row's LaTeX holds outside one."""
    meta = meta_of(node)
    if meta is None:
        return set()
    heads = {token.text for row in rows_of(node) for token in symbols_of(row.latex) if not token.subscript}
    return {f"{node.__name__} {token.text}" for token in symbols_of(meta.governing_equations)
            if not token.subscript and token.text not in heads}


@dataclass(frozen=True)
class TableReport:
    problems: tuple[str, ...]
    documented: tuple[str, ...]
    undocumented: tuple[str, ...]


def static_problems(nodes: Iterable[type], *, rows: RatchetList, indices: RatchetList,
                    frozen: Mapping[str, FrozenId]) -> TableReport:
    """The checks a node's class answers, over ``nodes``: each one's table, its unrowed symbols against ``indices``,
    every algorithm id unique, and the pinned rows of ``rows`` (the floor) still in their tables. A node with a table
    and no pinned row fails: every row is pinned when its table first lands."""
    walked = sorted(set(nodes), key=lambda cls: (cls.__name__, cls.__module__))
    found: list[str] = []
    names: dict[str, type] = {}
    ids: dict[str, str] = {}
    documented: list[str] = []
    undocumented: list[str] = []
    present: set[str] = set()
    hits: set[str] = set()
    used = {meta.algorithm_id for meta in map(meta_of, walked) if meta is not None and meta.algorithm_id}
    for node in walked:
        held = names.setdefault(node.__name__, node)
        if held is not node:
            found.append(f"{node.__name__}: two walked node classes have this name ({held.__module__} and "
                         f"{node.__module__}); the lists key a node by its name")
        meta = meta_of(node)
        if meta is None and not rows_of(node):
            undocumented.append(node.__name__)
            continue
        documented.append(node.__name__)
        found += table_problems(node, frozen, used)
        if meta is not None and meta.algorithm_id:
            other = ids.setdefault(meta.algorithm_id, node.__name__)
            if other != node.__name__:
                found.append(f"{node.__name__}: algorithm_id {meta.algorithm_id} is {other}'s too")
        keys = {row_key(node, row) for row in rows_of(node)}
        present |= keys
        if keys and not keys & rows.items():
            found.append(f"{node.__name__}: none of its symbol rows is pinned in {rows.path}; pin every row when its "
                         f"table first lands ('{sorted(keys)[0]}  # pinned at landing', one line each)")
        hits |= unrowed(node)
    found += floor_problems(rows, present, "symbol row")
    found += list_problems(indices, hits, "an equation symbol with no row (an index belongs in a braced subscript, "
                                          "x_{i})")
    return TableReport(tuple(found), tuple(documented), tuple(undocumented))


def _same_units(stated: str, declared: str) -> bool:
    try:
        return parse_unit(stated) == parse_unit(declared)
    except UnitError:
        return stated == declared


def instance_problems(node: SimulationNode, *, settings: Mapping[str, DimensionedSetting],
                      published: Mapping[str, str], units: RatchetList) -> list[str]:
    """The checks a built node answers: each row's ``name`` declared where its kind says, and its units what the code
    declares there. A row whose units the code does not declare is listed in ``units`` (declared, not checked); only
    that node's lines are judged here, and the list's own form by the docs check."""
    cls = type(node)
    held = cast(Any, node)
    specs: dict[SymbolKind, Mapping[str, Any]] = {
        SymbolKind.PARAMETER: cast(Mapping[str, Any], held.param_specs()),
        SymbolKind.STATE: cast(Mapping[str, Any], held.initial_state()),
        SymbolKind.INPUT: cast(Mapping[str, Any], held.boundary_input_spec()),
        SymbolKind.OUTPUT: cast(Mapping[str, Any], held.boundary_flux_spec()),
        SymbolKind.SETTING: settings,
    }
    found: list[str] = []
    unchecked: set[str] = set()
    for row in rows_of(cls):
        where = f"{cls.__name__}: the row {row.latex!r}"
        if row.kind is SymbolKind.CONSTANT:
            try:
                value = resolve_dotted(row.name)
            except (ImportError, AttributeError):
                found.append(f"{where}: the constant {row.name!r} does not resolve (a constant is named by its dotted "
                             f"path)")
                continue
            if isinstance(value, (type, ModuleType)) or callable(value):
                found.append(f"{where}: {row.name!r} is not a constant")
            unchecked.add(row_key(cls, row))
            continue
        if row.name not in specs[row.kind]:
            found.append(f"{where}: {row.kind.value} {row.name!r} is not declared; the node's are "
                         f"{sorted(specs[row.kind])}")
            continue
        declared = _declared_units(row, specs[row.kind][row.name], published)
        if declared is None:
            unchecked.add(row_key(cls, row))
        elif not _same_units(row.units, declared):
            found.append(f"{where}: units {row.units!r}, and the code declares {declared!r}")
    listed = RatchetList(units.path, tuple(line for line in units.lines if line.item.split(" ", 1)[0] == cls.__name__),
                         ())
    return found + list_problems(listed, unchecked, "a row whose units the code does not declare (declared, not "
                                                    "checked)")


def _declared_units(row: Symbol, spec: object, published: Mapping[str, str]) -> str | None:
    if row.kind is SymbolKind.PARAMETER:
        return cast(Any, spec).units or None
    if row.kind is SymbolKind.STATE:
        return published.get(row.name)
    if row.kind is SymbolKind.INPUT:
        return cast(Any, spec).expected_units or None
    if row.kind is SymbolKind.OUTPUT:
        return cast(Any, spec).output_units or None
    held = cast(DimensionedSetting, spec)
    return None if held.dimension is None else unit_symbol(held.dimension)


__all__ = ["STRUCTURE", "FrozenId", "TableReport", "Token", "instance_problems", "meta_of", "next_id",
           "path_problem", "read_frozen_ids", "resolve_dotted", "row_key", "rows_of", "static_problems", "symbols_of",
           "table_problems", "tokens", "unrowed"]
