"""The refusal codes MIME's source constructs, read from it without importing it: the scan the refusal table's rows are
generated from.

A code is found three ways:

- **at a construction call**: the first argument of ``Refusal``, ``Problem`` or a refusing helper, as a literal; the
  call names its class (``RefusalClass.USER``), or the helper implies one;
- **in a tuple** ``(code, message, action)``, the form the scene's, the materials' and the parts' problems travel in to
  the build, which refuses each as the user's;
- **by the cross-check**: any other string shaped like a code (``word.word-word``). One that is not a code is a name
  (an entry's, an interface's, a file's, a module's, or one on the names list); one that is a code reaches a
  construction through a variable, and takes the class of the construction that passes a variable in its own
  function, or failing that its module.

A code that matches no rule is a failure, so a code the rules miss is caught rather than left out of the table; so is a
code built where it is used (an f-string or a concatenation, ``f"material.{member}"``), which no rule can read."""

from __future__ import annotations

import ast
import importlib.util
import re
from collections.abc import Callable, Collection, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import override

from mime.build import RefusalClass

CODE_SHAPE = re.compile(r"[a-z][a-z0-9_-]*(?:\.[a-z0-9_-]+)+")
#: The shape the design's count of tuple codes used, which leaves out a hyphen in the first word.
CENSUS_SHAPE = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z0-9_-]+)+")
CONSTRUCTORS = frozenset({"Refusal", "refuse", "_refuse", "refused", "refuse_problems", "Problem"})
#: The class of a construction that names none: a helper that fixes it, by the file it is called in and its name. A
#: problem an entry's plan returns is refused as the user's.
IMPLIED: Mapping[tuple[str, str], RefusalClass] = {
    ("build/_checks.py", "_refuse"): RefusalClass.INTERNAL,
    ("saved/_reconstruct.py", "refuse"): RefusalClass.USER,
    ("materials/_read.py", "refused"): RefusalClass.USER,
}
FILE_SUFFIXES = frozenset({"json", "npz", "npy", "yaml", "yml", "lock", "usda", "usd", "usdc", "txt", "md", "py"})
#: The start of a string built into a code: its first word and the dot after it.
_CODE_HEAD = re.compile(r"[a-z][a-z0-9_-]*\.")


@dataclass(frozen=True)
class Site:
    """Where a code is constructed. ``code`` is ``None`` where it is passed by a variable; ``classes`` is empty where
    the scan cannot tell the class."""

    path: str
    line: int
    form: str
    callee: str
    code: str | None
    classes: frozenset[RefusalClass]

    @property
    def where(self) -> str:
        return f"{self.path}:{self.line}"


@dataclass(frozen=True)
class Literal:
    text: str
    path: str
    line: int
    function: str | None


@dataclass
class Scan:
    """Every construction site and code-shaped literal of a package's source."""

    calls: list[Site] = field(default_factory=list[Site])
    tuples: list[Site] = field(default_factory=list[Site])
    variables: list[Site] = field(default_factory=list[Site])
    #: Constructions whose code is built where it is used, at a call or in a ``(code, message, action)`` tuple.
    built: list[Site] = field(default_factory=list[Site])
    literals: list[Literal] = field(default_factory=list[Literal])
    functions: dict[tuple[str, int], str | None] = field(default_factory=dict[tuple[str, int], str | None])

    def sites(self) -> Iterator[Site]:
        yield from self.calls
        yield from self.tuples


def _callee(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _classes(call: ast.Call) -> frozenset[RefusalClass]:
    named = [value for value in [*call.args, *(k.value for k in call.keywords)]
             if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name)
             and value.value.id == "RefusalClass" and value.attr in RefusalClass.__members__]
    return frozenset(RefusalClass[value.attr] for value in named)


def _code(node: ast.expr | None) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _head(node: ast.expr) -> str:
    """The literal text a string built by an f-string or a concatenation starts with."""
    if isinstance(node, ast.JoinedStr):
        return _head(node.values[0]) if node.values else ""
    if isinstance(node, ast.BinOp):
        return _head(node.left)
    return _code(node) or ""


def _built(node: ast.expr) -> bool:
    return isinstance(node, (ast.JoinedStr, ast.BinOp))


class _Reader(ast.NodeVisitor):
    def __init__(self, path: str, scan: Scan) -> None:
        self.path, self.scan = path, scan
        self.function: list[str] = []
        self.at_sites: set[tuple[int, int]] = set()

    def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.function.append(node.name)
        self.generic_visit(node)
        self.function.pop()

    @override
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802 - ast's visitor names
        self._function(node)

    @override
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._function(node)

    def _inside(self) -> str | None:
        return ".".join(self.function) or None

    @override
    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        callee = _callee(node)
        if callee in CONSTRUCTORS:
            first = node.args[0] if node.args else next((k.value for k in node.keywords if k.arg == "code"), None)
            if first is not None:
                classes = _classes(node)
                if not classes and callee == "Problem":
                    classes = frozenset({RefusalClass.USER})
                if not classes and (self.path, callee) in IMPLIED:
                    classes = frozenset({IMPLIED[(self.path, callee)]})
                code = _code(first)
                site = Site(self.path, node.lineno, "call" if code is not None else "variable", callee, code, classes)
                if code is not None:
                    self.scan.calls.append(site)
                    self.at_sites.add((first.lineno, first.col_offset))
                else:
                    self.scan.variables.append(site)
                    self.scan.functions[(self.path, node.lineno)] = self._inside()
                if _built(first):
                    self.scan.built.append(site)
        self.generic_visit(node)

    @override
    def visit_Tuple(self, node: ast.Tuple) -> None:  # noqa: N802
        code = _code(node.elts[0]) if len(node.elts) == 3 else None
        if code is not None and CODE_SHAPE.fullmatch(code):
            self.scan.tuples.append(Site(self.path, node.lineno, "tuple", "", code, frozenset({RefusalClass.USER})))
            self.at_sites.add((node.elts[0].lineno, node.elts[0].col_offset))
        elif len(node.elts) == 3 and _built(node.elts[0]) and _CODE_HEAD.match(_head(node.elts[0])):
            self.scan.built.append(Site(self.path, node.lineno, "tuple", "", None, frozenset({RefusalClass.USER})))
        self.generic_visit(node)

    @override
    def visit_Constant(self, node: ast.Constant) -> None:  # noqa: N802
        if isinstance(node.value, str) and CODE_SHAPE.fullmatch(node.value) and \
                (node.lineno, node.col_offset) not in self.at_sites:
            self.scan.literals.append(Literal(node.value, self.path, node.lineno, self._inside()))


def scan_package(package: Path) -> Scan:
    """Every construction site and code-shaped literal under ``package`` (``src/mime``), paths relative to it."""
    scan = Scan()
    for source in sorted(package.rglob("*.py")):
        path = source.relative_to(package).as_posix()
        _Reader(path, scan).visit(ast.parse(source.read_text(encoding="utf-8"), filename=str(source)))
    return scan


@dataclass(frozen=True)
class Derived:
    """What the scan makes of a package: each code with its classes and where it is constructed; the code-shaped
    literals that are on the names list, and those that are neither a code nor a name; and every construction whose
    class the scan cannot tell."""

    codes: Mapping[str, frozenset[RefusalClass]]
    where: Mapping[str, tuple[str, ...]]
    listed: frozenset[str]
    unclassified: tuple[Literal, ...]
    unknown_class: tuple[Site, ...]


def derive(scan: Scan, is_name: Callable[[str], bool], listed: Collection[str] = ()) -> Derived:
    """The codes ``scan`` finds. ``is_name`` says which code-shaped literals are names by rule, and ``listed`` holds
    those the names list says are names."""
    codes: dict[str, set[RefusalClass]] = {}
    where: dict[str, list[str]] = {}
    unknown: list[Site] = []
    for site in scan.sites():
        assert site.code is not None
        if not site.classes:
            unknown.append(site)
        codes.setdefault(site.code, set()).update(site.classes)
        where.setdefault(site.code, []).append(site.where)
    by_function: dict[tuple[str, str | None], set[RefusalClass]] = {}
    by_module: dict[str, set[RefusalClass]] = {}
    for site in scan.variables:
        function = scan.functions.get((site.path, site.line))
        by_function.setdefault((site.path, function), set()).update(site.classes)
        by_module.setdefault(site.path, set()).update(site.classes)
    names: set[str] = set()
    unclassified: list[Literal] = []
    for literal in scan.literals:
        if literal.text in codes or is_name(literal.text):
            continue
        if literal.text in listed:
            names.add(literal.text)
            continue
        classes = by_module.get(literal.path) if literal.function is None else \
            by_function.get((literal.path, literal.function))
        if not classes:
            unclassified.append(literal)
            continue
        codes.setdefault(literal.text, set()).update(classes)
        where.setdefault(literal.text, []).append(f"{literal.path}:{literal.line}")
    return Derived({code: frozenset(classes) for code, classes in sorted(codes.items())},
                   {code: tuple(sorted(set(places))) for code, places in sorted(where.items())},
                   frozenset(names), tuple(unclassified), tuple(unknown))


#: The packages whose modules, and the attributes of those modules, are names by rule: a module's, or an origin the
#: switch scan resolves a call to (``jax.numpy.sqrt``).
NAMED_PACKAGES = ("mime", "maddening", "jax", "builtins")


def _is_module(text: str) -> bool:
    if text.split(".", 1)[0] not in NAMED_PACKAGES:
        return False
    try:
        if importlib.util.find_spec(text) is not None:
            return True
    except (ImportError, ValueError):
        pass
    module, _, attribute = text.rpartition(".")
    try:
        return hasattr(importlib.import_module(module), attribute)
    except ImportError:
        return False


def name_rule(entries: Collection[str], interfaces: Collection[str],
              owners: Collection[str] = ()) -> Callable[[str], bool]:
    """Which code-shaped literals are names by rule: a registered entry's, a vocabulary interface's, a declared switch
    owner (``contact.band``), a file's (by its suffix), and a module's, or a module's attribute's, under one of
    ``NAMED_PACKAGES``."""

    def is_name(text: str) -> bool:
        return (text in entries or text in interfaces or text in owners or text.rsplit(".", 1)[-1] in FILE_SUFFIXES
                or _is_module(text))

    return is_name


@dataclass(frozen=True)
class Census:
    """The scan's counts as the design counted them: distinct literal codes at construction calls and the files they
    are in, calls passing a code by a variable, and codes in ``(code, message, action)`` tuples of plain parts that no
    construction call names."""

    literal_codes: int
    files: int
    variable_calls: int
    tuple_codes: int


_PLAIN = (ast.Constant, ast.JoinedStr, ast.Call, ast.BinOp, ast.Name, ast.Attribute)


def census(package: Path) -> Census:
    scan = scan_package(package)
    called = {site.code for site in scan.calls}
    plain: set[str] = set()
    for source in sorted(package.rglob("*.py")):
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Tuple) and len(node.elts) == 3 and all(isinstance(e, _PLAIN) for e in node.elts):
                code = _code(node.elts[0])
                if code is not None and CENSUS_SHAPE.fullmatch(code) and code not in called:
                    plain.add(code)
    return Census(len(called), len({site.path for site in scan.calls}), len(scan.variables), len(plain))
