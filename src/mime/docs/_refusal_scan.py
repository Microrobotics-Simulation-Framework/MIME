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
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
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
    #: The action's text as the source writes it, each alternative an ``if`` or ``or`` gives, every formatted value
    #: written ``{}``; ``None`` where the lint cannot read it, and empty where the construction takes no action.
    actions: tuple[str, ...] | None = ()
    #: The message's text, read as the action is.
    messages: tuple[str, ...] | None = ()

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


type Locals = Mapping[str, Sequence[ast.expr]]


def _locals(function: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, list[ast.expr]]:
    """Every value a function assigns to each plain name in its own body, nested functions left out, and each
    parameter's default."""
    found: dict[str, list[ast.expr]] = {}
    arguments = function.args
    positional = [*arguments.posonlyargs, *arguments.args]
    for argument, default in zip(positional[len(positional) - len(arguments.defaults):], arguments.defaults,
                                 strict=True):
        found[argument.arg] = [default]
    for argument, default in zip(arguments.kwonlyargs, arguments.kw_defaults, strict=True):
        if default is not None:
            found[argument.arg] = [default]
    pending: list[ast.AST] = list(function.body)
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    found.setdefault(target.id, []).append(node.value)
                elif isinstance(target, ast.Tuple) and isinstance(node.value, ast.Tuple) and \
                        len(target.elts) == len(node.value.elts):
                    for name, value in zip(target.elts, node.value.elts, strict=True):
                        if isinstance(name, ast.Name):
                            found.setdefault(name.id, []).append(value)
        pending.extend(ast.iter_child_nodes(node))
    return found


#: The suffix of a function of the package that returns a command line (``fetch_command``): a value it makes in an
#: f-string is a command, whatever builds its text.
COMMAND_SUFFIX = "_command"


def _commanded(value: ast.expr) -> bool:
    held = value.value if isinstance(value, ast.FormattedValue) else value
    return isinstance(held, ast.Call) and isinstance(held.func, ast.Name) and held.func.id.endswith(COMMAND_SUFFIX)


#: The package's functions, each with every value it returns and its own names' values: an action made by calling one
#: reads as each of its returns. Set while a package is scanned.
_RETURNS: dict[str, tuple[list[ast.expr], Locals]] = {}


def _returns(trees: Mapping[str, ast.Module]) -> dict[str, tuple[list[ast.expr], Locals]]:
    """Each module-level function of the package by its name, where one function has the name, with its returns."""
    found: dict[str, list[tuple[list[ast.expr], Locals]]] = {}
    for tree in trees.values():
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                returns = [held.value for held in ast.walk(node) if isinstance(held, ast.Return)
                           and held.value is not None]
                found.setdefault(node.name, []).append((returns, _locals(node)))
    return {name: held[0] for name, held in found.items() if len(held) == 1}


def _texts(node: ast.expr, constants: Mapping[str, str], local: Locals | None = None,
           calling: frozenset[str] = frozenset()) -> list[str] | None:
    """Every text ``node`` can be, each formatted value or call written ``{}``: a literal, an f-string, a
    concatenation, a module's string constant, a name its function assigns only such texts to (a parameter, its
    default), a call of a function of the package that returns only such texts, or each branch of an ``if`` or an
    ``or``; ``None`` when it is none of these (another call, a parameter with no default)."""
    local = local or {}
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _RETURNS and \
            node.func.id not in calling:
        returns, held_locals = _RETURNS[node.func.id]
        found_texts: list[str] = []
        for value in returns:
            held = _texts(value, constants, held_locals, calling | {node.func.id})
            if held is None:
                return None
            found_texts += held
        return found_texts or None
    if isinstance(node, ast.Name) and node.id in local and node.id not in constants:
        found: list[str] = []
        for value in local[node.id]:
            held = _texts(value, constants, {k: v for k, v in local.items() if k != node.id})
            if held is None:
                return None
            found += held
        return found
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.JoinedStr):
        return ["".join(v.value if isinstance(v, ast.Constant) and isinstance(v.value, str) else
                        "`{}`" if _commanded(v) else "{}" for v in node.values)]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _texts(node.left, constants, local) or ["{}"], _texts(node.right, constants, local) or ["{}"]
        return [a + b for a in left for b in right]
    if isinstance(node, ast.Name) and node.id in constants:
        return [constants[node.id]]
    if isinstance(node, ast.IfExp):
        body, orelse = _texts(node.body, constants, local), _texts(node.orelse, constants, local)
        return None if body is None or orelse is None else body + orelse
    if isinstance(node, ast.BoolOp):
        found: list[str] = []
        for value in node.values:
            held = _texts(value, constants, local)
            if held is None:
                return None
            found += held
        return found
    return None


def _argument_node(call: ast.Call, name: str, after_class: int) -> ast.expr | None:
    """The argument ``name`` a construction call passes: by its keyword, else at its place, ``after_class`` past the
    code and the class (where one is passed)."""
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    classed = len(call.args) > 1 and isinstance(call.args[1], ast.Attribute) and \
        isinstance(call.args[1].value, ast.Name) and call.args[1].value.id == "RefusalClass"
    at = (2 if classed else 1) + after_class
    return call.args[at] if len(call.args) > at else None


def _action_node(call: ast.Call, callee: str) -> ast.expr | None:
    """The action a construction call passes: by its keyword, else after the code, the class (where one is passed)
    and the message."""
    return _argument_node(call, "action", 1)


def _message_node(call: ast.Call) -> ast.expr | None:
    return _argument_node(call, "message", 0)


class _Reader(ast.NodeVisitor):
    def __init__(self, path: str, scan: Scan, constants: Mapping[str, str] | None = None) -> None:
        self.path, self.scan = path, scan
        self.constants = constants or {}
        self.local: list[Locals] = []
        self.function: list[str] = []
        self.at_sites: set[tuple[int, int]] = set()

    def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.function.append(node.name)
        self.local.append(_locals(node))
        self.generic_visit(node)
        self.local.pop()
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
                action = _action_node(node, callee)
                actions = () if action is None else _texts(action, self.constants,
                                                           self.local[-1] if self.local else None)
                held = tuple(actions) if actions is not None else None
                message = _message_node(node)
                said = None if message is None else _texts(message, self.constants,
                                                           self.local[-1] if self.local else None)
                site = Site(self.path, node.lineno, "call" if code is not None else "variable", callee, code, classes,
                            held, tuple(said) if said is not None else None)
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
            local = self.local[-1] if self.local else None
            actions, said = _texts(node.elts[2], self.constants, local), _texts(node.elts[1], self.constants, local)
            self.scan.tuples.append(Site(self.path, node.lineno, "tuple", "", code, frozenset({RefusalClass.USER}),
                                         tuple(actions) if actions is not None else None,
                                         tuple(said) if said is not None else None))
            self.at_sites.add((node.elts[0].lineno, node.elts[0].col_offset))
        elif len(node.elts) == 3 and _built(node.elts[0]) and _CODE_HEAD.match(_head(node.elts[0])):
            self.scan.built.append(Site(self.path, node.lineno, "tuple", "", None, frozenset({RefusalClass.USER})))
        self.generic_visit(node)

    @override
    def visit_Constant(self, node: ast.Constant) -> None:  # noqa: N802
        if isinstance(node.value, str) and CODE_SHAPE.fullmatch(node.value) and \
                (node.lineno, node.col_offset) not in self.at_sites:
            self.scan.literals.append(Literal(node.value, self.path, node.lineno, self._inside()))


def _constants(trees: Mapping[str, ast.Module]) -> dict[str, str]:
    """The package's module-level string constants by name, where the name holds one text across every module that
    assigns it: what an action passed by its name reads as."""
    found: dict[str, set[str]] = {}
    for tree in trees.values():
        for node in tree.body:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(
                node, ast.AnnAssign) and node.value is not None else []
            value = node.value if isinstance(node, (ast.Assign, ast.AnnAssign)) else None
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                for target in targets:
                    if isinstance(target, ast.Name):
                        found.setdefault(target.id, set()).add(value.value)
    return {name: next(iter(texts)) for name, texts in found.items() if len(texts) == 1}


def scan_package(package: Path) -> Scan:
    """Every construction site and code-shaped literal under ``package`` (``src/mime``), paths relative to it."""
    scan = Scan()
    trees = {source.relative_to(package).as_posix(): ast.parse(source.read_text(encoding="utf-8"),
                                                                filename=str(source))
             for source in sorted(package.rglob("*.py"))}
    constants = _constants(trees)
    _RETURNS.clear()
    _RETURNS.update(_returns(trees))
    for path, tree in trees.items():
        _Reader(path, scan, constants).visit(tree)
    return scan


#: What an action must hold to name what to change: a keyword argument (``spacing=``), a call or an index
#: (``exp.declare(``, ``mm(``, ``scene["..."]``), a command or flag (``mime assets fetch``, ``--into``,
#: ``pip install``), a variable of MIME's (``MIME_ASSETS_DIR``), or code written in backticks.
NAMES_WHAT_TO_CHANGE = re.compile(r"[\w}\]]=|[\w}][(\[]|`[^`]+`|\bmime [a-z]|(?<![\w-])--[a-z]|\bpip install|"
                                  r"\buv [a-z]|\bMIME_[A-Z_]+")


def action_problems(scan: Scan, table: Mapping[str, frozenset[RefusalClass]]) -> list[str]:
    """Every construction of a code the user or the environment can act on whose action names no MIME argument or
    command (:data:`NAMES_WHAT_TO_CHANGE`), or which the lint cannot read. A code refused only as MIME's defect is
    left out: its action is the report and the issue, which its display gives."""
    problems: list[str] = []
    for site in scan.sites():
        assert site.code is not None
        classes = table.get(site.code, site.classes)
        if not classes - {RefusalClass.INTERNAL}:
            continue
        if site.actions is None:
            problems.append(f"{site.where}: {site.code}'s action is neither text nor a string constant, so the lint "
                            f"cannot read it; write it as an f-string")
            continue
        for text in site.actions:
            if not NAMES_WHAT_TO_CHANGE.search(text):
                problems.append(f"{site.where}: {site.code}'s action names no MIME argument or command ({text!r}); "
                                f"write what to change as code: spacing=mm(...), exp.declare(...), `mime ...`")
    return problems


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
