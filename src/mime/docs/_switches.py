"""The switch scan: every non-smooth operation a registered node's step evaluates goes through ``mime.core.switch``,
under an owner some entry or shape declares, and none inside a transform.

**Its scope comes from traces.** Each registered node's step is traced in the configurations a caller builds, and
the source function JAX records for each equation, nested jaxprs included, names what the step runs: every function
of the scanned package on an equation's frames, outside ``mime.core.switch`` and ``mime.testing``. Code that produces
no equation (building, planning, constructors, scripts) is never scanned, and a function no configuration reaches is
out of scope until one does. A registered node class no configuration traces fails the scan, naming it.

**The families**, applied to those functions' source (their own bodies; a nested function or lambda is scanned only
if it is itself in scope, so a lazy indicator, evaluated only inside a collector, is not): a raw non-smooth operation
of ``jax.numpy`` or ``jax.lax`` (or the builtin ``abs`` or ``round``, or an array method of the same name), MIME's
``safe_norm``; a square root, or ``** 0.5``, not of a ``domain_guard`` output; a comparison, or a name bound once to
one or to a ``predicate``, cast to a number or multiplied in; ``stop_gradient``, ``custom_jvp`` and ``custom_vjp``.
Names are resolved through the module's imports and aliases, MIME's re-exports followed.

**Nesting**, read from the traces: an equation a switch function emits inside a transform's body (a ``cond``,
``switch``, ``scan``, ``fori_loop`` or ``while_loop`` body, a ``checkpoint``, a nested ``jit``), or under ``vmap``,
whose frames show it, called directly or through a helper.

**The owners**: every owner literal passed to a switch function, or to a helper's ``owner`` argument, is declared,
and every owner the package's entries and shapes declare is passed somewhere in scope. An owner that is neither a
literal nor the enclosing function's own parameter, passed on, cannot be read and fails.

Every hit not listed in ``docs/ratchets/raw_switches.txt`` fails, and so does a listed site that is no longer a hit.
A listed ``stop_gradient``, ``custom_jvp`` or ``custom_vjp`` names the status it caps its entry at, or the standing
test that shows the entry's status holds with it. JAX's source information is internal to it: the one function that
reads it is :func:`equation_frames`."""

from __future__ import annotations

import ast
import importlib.util
import inspect
import re
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any, cast

from jax.extend.core import Jaxpr, JaxprEqn
from maddening.core.node import SimulationNode

from mime.build._maddening import make_jaxpr
from mime.core import MimeNode, shape_named, shape_names, shape_switches
from mime.core.switch import SWITCH_FUNCTIONS, Declaration
from mime.docs._lists import RatchetList
from mime.gradients import GradientStatus
from mime.registry import Registered

#: The switch module, where every switch function's own equations are emitted.
SWITCH_MODULE = "mime.core.switch"
#: Where a scanned package's code is never scanned: the switch module, the test harnesses, this scan.
EXEMPT = ("core/switch.py", "testing/", "docs/_switches.py")
#: The raw non-smooth operations, by their origin; a call of one in scope is a hit named by its last part.
RAW_OPERATIONS = frozenset(
    {f"jax.numpy.{name}" for name in ("where", "maximum", "minimum", "abs", "absolute", "clip", "sign", "floor", "ceil",
                                      "round", "rint", "trunc", "argmax", "argmin", "argsort", "sort", "max",
                                      "min", "amax", "amin", "fmax", "fmin", "heaviside", "select", "piecewise")}
    | {"jax.numpy.linalg.norm"}
    | {f"jax.lax.{name}" for name in ("select", "select_n", "cond", "switch", "max", "min", "abs", "sign", "clamp",
                                      "floor", "ceil", "round")}
    | {"jax.nn.relu", "jax.nn.relu6", "jax.nn.hard_tanh"}
    | {"builtins.abs", "builtins.round"}
    | {"mime.core._quaternion.safe_norm", "mime.core.switch._safe_norm"})
#: Array methods that are raw non-smooth operations, called on a value the scan cannot resolve.
RAW_METHODS = frozenset({"clip", "max", "min", "argmax", "argmin", "argsort", "sort", "round", "__abs__"})
SQUARE_ROOTS = frozenset({"jax.numpy.sqrt", "jax.lax.sqrt"})
STOP_GRADIENTS = frozenset({"jax.lax.stop_gradient"})
CUSTOM_RULES = {"jax.custom_jvp": "custom_jvp", "jax.custom_vjp": "custom_vjp"}
CASTS = frozenset({"astype"})
#: The families whose listed site names the status it caps its entry at, or the test that clears it.
CAPPING = frozenset({"stop_gradient", "custom_jvp", "custom_vjp"})
#: The frames by which JAX's ``vmap`` calls the function it maps: rootward of a switch's frame, the switch is
#: evaluated under ``vmap``, which no sub-jaxpr shows.
VMAP_FRAMES = ("jax/_src/interpreters/batching.py", "jax/_src/api.py:vmap.")

_KINDS = r"(?:dynamics|interaction|observation|field_provider|controller|drive|geometry)"
#: A list item: the file (relative to the package), the function's qualified name, the family and its occurrence in
#: that function, and for a capping family the entry and status it caps, or the test that clears it.
ITEM = re.compile(rf"(?P<site>(?P<path>[\w./-]+\.py) (?P<function>[\w.<>]+) (?P<family>[a-z_]+) (?P<occurrence>[1-9]\d*))"
                  rf"(?: caps (?P<entry>{_KINDS} [a-z][a-z0-9_.]* \S+) (?P<status>cut|unverified)"
                  rf"| cleared_by (?P<test>\S+::\S+))?")


def item_form(item: str) -> str | None:
    if ITEM.fullmatch(item) is None:
        return ("an item is a site, '<file> <function> <family> <occurrence>', then for a capping family 'caps <kind> "
                "<name> <version> cut|unverified' or 'cleared_by <test id>'")
    return None


@dataclass(frozen=True)
class ListedSite:
    site: str
    family: str
    entry: str | None
    status: GradientStatus | None
    cleared_by: str | None


def listed_sites(held: RatchetList) -> dict[str, ListedSite]:
    """The list's well-formed lines, by their site."""
    found: dict[str, ListedSite] = {}
    for line in held.lines:
        matched = ITEM.fullmatch(line.item)
        if matched is not None:
            status = matched["status"]
            found[matched["site"]] = ListedSite(matched["site"], matched["family"], matched["entry"],
                                                GradientStatus(status) if status else None, matched["test"])
    return found


def caps(held: RatchetList) -> dict[str, GradientStatus]:
    """The status each capped entry (``kind name version``, as the lists name one) is capped at: the lowest its lines
    name."""
    found: dict[str, GradientStatus] = {}
    order = (GradientStatus.CUT, GradientStatus.UNVERIFIED)
    for listed in listed_sites(held).values():
        if listed.entry is not None and listed.status is not None:
            held_status = found.get(listed.entry)
            if held_status is None or order.index(listed.status) < order.index(held_status):
                found[listed.entry] = listed.status
    return found


# -- the traces ----------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Frame:
    """One frame of an equation's recorded traceback: the file, the function's qualified name, its first line (its
    first decorator's, if it has one) and the line executing."""

    file: str
    function: str
    start: int
    line: int


def equation_frames(equation: JaxprEqn) -> tuple[Frame, ...]:
    """The frames JAX recorded where ``equation`` was traced, innermost first. JAX records them in an internal object
    (``source_info.traceback``), read here and nowhere else; it cuts them at the function the enclosing trace began
    in (a step, a ``scan`` body, a jitted function)."""
    traceback = cast(Any, equation).source_info.traceback
    if traceback is None:
        return ()
    return tuple(Frame(str(f.file_name), str(f.function_name), int(f.function_start_line), int(f.line_num))
                 for f in traceback.frames)


def _open(jaxpr: object) -> Jaxpr | None:
    """A jaxpr, or a closed jaxpr's (a parameter of a ``jit``, ``cond`` or ``scan`` holds either); ``None`` for any
    other value."""
    held = cast(object, getattr(jaxpr, "jaxpr", jaxpr))
    return held if isinstance(held, Jaxpr) else None


def _inner_jaxprs(equation: JaxprEqn) -> Iterator[Jaxpr]:
    for value in equation.params.values():
        for held in cast(tuple[object, ...], value) if isinstance(value, (tuple, list)) else (value,):
            inner = _open(held)
            if inner is not None:
                yield inner


def equations(jaxpr: object, within: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], JaxprEqn]]:
    """Every equation of ``jaxpr`` (open or closed), nested ones included, with the primitives whose bodies it sits
    inside."""
    held = _open(jaxpr)
    for equation in held.eqns if held is not None else ():
        yield within, equation
        for inner in _inner_jaxprs(equation):
            yield from equations(inner, (*within, equation.primitive.name))


@dataclass(frozen=True)
class TracedStep:
    configuration: str
    node: str
    node_class: type[SimulationNode]
    jaxpr: Jaxpr


def step_function(node: SimulationNode) -> Callable[..., object]:
    """What a node's step is traced through: a MIME node's ``fresh_update``, whose top level is its step (``update``
    wraps it in the compiled call, a ``jit`` the scan would read as nesting), or another node's ``update``."""
    if isinstance(node, MimeNode):
        return node.fresh_update
    return cast(Callable[..., object], cast(Any, node).update)


def traced_step(configuration: str, name: str, node: SimulationNode, state: object, given: object,
                leaves: object) -> TracedStep:
    """One node's step traced with no switch context open, from the state, inputs and ``params`` leaves the graph
    gives it."""
    step = step_function(node)
    dt = cast(float, cast(Any, node).delta_t)

    def with_leaves(state: object, given: object, leaves: object) -> object:
        return step(state, given, dt, params=leaves)

    def plain(state: object, given: object) -> object:
        return step(state, given, dt)

    if "params" in inspect.signature(step).parameters:
        return TracedStep(configuration, name, type(node), make_jaxpr(with_leaves, state, given, leaves))
    return TracedStep(configuration, name, type(node), make_jaxpr(plain, state, given))


# -- the source ----------------------------------------------------------------------------------------------------

@dataclass(frozen=True, order=True)
class Function:
    """A function of the scanned package in scope: its file relative to the package, its qualified name and its
    first line."""

    path: str
    name: str
    start: int


type Definition = ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda


@cache
def _parsed(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _first_line(node: Definition) -> int:
    if isinstance(node, ast.Lambda):
        return node.lineno
    return min([node.lineno, *(decorator.lineno for decorator in node.decorator_list)])


def definitions(path: Path, function: Function) -> list[Definition]:
    """The definitions ``function`` can be in ``path``: those starting on its line with its last name."""
    last = function.name.rsplit(".", 1)[-1]
    found: list[Definition] = []
    for node in ast.walk(_parsed(path)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) and _first_line(node) == function.start:
            name = "<lambda>" if isinstance(node, ast.Lambda) else node.name
            if name == last:
                found.append(node)
    return found


_NESTED = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


def _own_nodes(definition: Definition) -> Iterator[ast.AST]:
    """The nodes of a definition's own code: its body, and a ``def``'s decorators; nested definitions excluded."""
    roots: list[ast.AST] = [definition.body] if isinstance(definition, ast.Lambda) else [*definition.body]
    if not isinstance(definition, ast.Lambda):
        roots += definition.decorator_list
        roots += [d for d in ast.walk(definition.args) if isinstance(d, ast.expr)]
    pending = [root for root in roots if not isinstance(root, _NESTED)]
    while pending:
        node = pending.pop()
        yield node
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, _NESTED):
                pending.append(child)


def _module_name(package: Path, path: Path) -> str:
    parts = list(path.relative_to(package.parent).with_suffix("").parts)
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _unwrapped(node: ast.expr) -> ast.expr:
    """``cast(T, x)`` as ``x``."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "cast" and len(node.args) == 2:
        return node.args[1]
    return node


def _dotted(node: ast.expr) -> str | None:
    node = _unwrapped(node)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return None if base is None else f"{base}.{node.attr}"
    return None


_reading: set[str] = set()


@cache
def module_aliases(module: str) -> Mapping[str, str]:
    """What each top-level name of ``module`` (a ``mime`` or lab module) stands for, as a dotted origin: its imports,
    and its assignments of a module or one of its attributes (``jnp: Any = jax.numpy``). A name imported from another
    such module is followed into it; one imported from a module whose own aliases are being read (a package importing
    its own submodule) is taken as that module's attribute."""
    spec = importlib.util.find_spec(module)
    if spec is None or spec.origin is None or not spec.origin.endswith(".py"):
        return {}
    _reading.add(module)
    try:
        return tree_aliases(_parsed(Path(spec.origin)), module)
    finally:
        _reading.discard(module)


def tree_aliases(tree: ast.Module, module: str) -> dict[str, str]:
    """What each top-level name of a module's source stands for (see :func:`module_aliases`); a function or class it
    defines stands for itself, ``<module>.<name>``."""
    found: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found[node.name] = f"{module}.{node.name}"
        found.update(_import_aliases(node, module, found))
    return found


def _import_aliases(node: ast.stmt, module: str, known: Mapping[str, str]) -> dict[str, str]:
    found: dict[str, str] = {}
    if isinstance(node, ast.Import):
        for alias in node.names:
            found[alias.asname or alias.name.split(".")[0]] = alias.name if alias.asname else alias.name.split(".")[0]
    elif isinstance(node, ast.ImportFrom) and node.module is not None:
        source = node.module if node.level == 0 else _relative(module, node.module, node.level)
        for alias in node.names:
            found[alias.asname or alias.name] = _origin_of(source, alias.name)
    elif isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        dotted = _dotted(node.value)
        if dotted is not None:
            head, _, rest = dotted.partition(".")
            if head in known:
                origin = known[head] + (f".{rest}" if rest else "")
                found.update({t.id: origin for t in targets if isinstance(t, ast.Name)})
    return found


def _relative(module: str, name: str, level: int) -> str:
    return ".".join([*module.split(".")[:-level], name])


def _origin_of(source: str, name: str) -> str:
    """``name`` as imported from ``source``: what the module binds it to, if it is a module the scan can read and
    binds it as an alias; a submodule; or ``source.name``."""
    top = source.split(".")[0]
    if top not in ("jax", "jaxtyping", "builtins") and source not in _reading and \
            importlib.util.find_spec(top) is not None:
        held = module_aliases(source)
        if name in held:
            return held[name]
    return f"{source}.{name}"


@dataclass
class Hit:
    site: str
    where: str


@dataclass
class OwnerUse:
    owner: str
    where: str
    clip: bool = False


@dataclass
class SourceFindings:
    hits: list[Hit] = field(default_factory=list[Hit])
    owners: list[OwnerUse] = field(default_factory=list[OwnerUse])
    unreadable_owners: list[str] = field(default_factory=list[str])


def _resolve(node: ast.expr, local: Mapping[str, str], aliases: Mapping[str, str]) -> str | None:
    dotted = _dotted(node)
    if dotted is None:
        return None
    head, _, rest = dotted.partition(".")
    origin = local.get(head) or aliases.get(head)
    if origin is None:
        origin = f"builtins.{head}" if head in ("abs", "round") else None
    if origin is None:
        return None
    return origin + (f".{rest}" if rest else "")


def _canonical(origin: str) -> str:
    """An origin through MIME's module aliases: ``mime.core.quaternion.safe_norm`` is ``mime.core._quaternion``'s."""
    module, _, name = origin.rpartition(".")
    if module.startswith("mime.") and name:
        held = module_aliases(module)
        if name in held and held[name] != origin:
            return _canonical(held[name])
    return origin


def _bound_once(definition: Definition, local: Mapping[str, str], aliases: Mapping[str, str]) -> dict[str, str]:
    """The names a definition binds exactly once, to what: ``comparison``, ``predicate`` or ``domain_guard``."""
    counts: dict[str, int] = {}
    kinds: dict[str, str] = {}
    for node in _own_nodes(definition):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            counts[name] = counts.get(name, 0) + 1
            kind = _value_kind(node.value, local, aliases)
            if kind is not None:
                kinds[name] = kind
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)) and isinstance(node.target, ast.Name):
            counts[node.target.id] = counts.get(node.target.id, 0) + 2
    return {name: kind for name, kind in kinds.items() if counts.get(name) == 1}


def _value_kind(node: ast.expr, local: Mapping[str, str], aliases: Mapping[str, str]) -> str | None:
    if isinstance(node, ast.Compare):
        return "comparison"
    if isinstance(node, ast.Call):
        origin = _resolve(node.func, local, aliases)
        if origin == f"{SWITCH_MODULE}.predicate":
            return "predicate"
        if origin == f"{SWITCH_MODULE}.domain_guard":
            return "domain_guard"
    return None


def _is(node: ast.expr, kinds: Collection[str], bound: Mapping[str, str], local: Mapping[str, str],
        aliases: Mapping[str, str]) -> bool:
    if isinstance(node, ast.Name) and bound.get(node.id) in kinds:
        return True
    return _value_kind(node, local, aliases) in kinds


def scan_definition(definition: Definition, aliases: Mapping[str, str], where: Callable[[ast.AST], str],
                    site: Callable[[str], str]) -> SourceFindings:
    """The families' hits and the owners passed in one definition's own code. ``site(family)`` names a hit's site,
    numbering each family's occurrences in source order."""
    findings = SourceFindings()
    local: dict[str, str] = {}
    nodes = sorted(_own_nodes(definition), key=lambda n: (getattr(n, "lineno", 0), getattr(n, "col_offset", 0)))
    for node in nodes:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            local.update(_import_aliases(node, "", {**aliases, **local}))
    bound = _bound_once(definition, local, aliases)
    arguments = definition.args
    parameters = {a.arg for a in [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]}
    for node in nodes:
        if isinstance(node, ast.Call):
            origin = _resolve(node.func, local, aliases)
            canonical = _canonical(origin) if origin is not None else None
            family = _call_family(node, canonical, bound, local, aliases)
            if family is not None:
                findings.hits.append(Hit(site(family), where(node)))
            for keyword in node.keywords:
                if keyword.arg != "owner":
                    continue
                clip = canonical == f"{SWITCH_MODULE}.clip"
                if isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
                    findings.owners.append(OwnerUse(keyword.value.value, where(node), clip))
                elif not (isinstance(keyword.value, ast.Name) and keyword.value.id in parameters):
                    findings.unreadable_owners.append(where(node))
        elif isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Pow) and isinstance(node.right, ast.Constant) and node.right.value == 0.5 \
                    and not _is(node.left, ("domain_guard",), bound, local, aliases):
                findings.hits.append(Hit(site("sqrt"), where(node)))
            if isinstance(node.op, ast.Mult) and any(_is(side, ("comparison", "predicate"), bound, local, aliases)
                                                     for side in (node.left, node.right)):
                findings.hits.append(Hit(site("comparison_cast"), where(node)))
    if not isinstance(definition, ast.Lambda):
        for decorator in definition.decorator_list:
            origin = _resolve(decorator.func if isinstance(decorator, ast.Call) else decorator, local, aliases)
            if origin in CUSTOM_RULES:
                findings.hits.append(Hit(site(CUSTOM_RULES[origin]), where(decorator)))
    return findings


def _call_family(node: ast.Call, origin: str | None, bound: Mapping[str, str], local: Mapping[str, str],
                 aliases: Mapping[str, str]) -> str | None:
    if origin is not None:
        if origin in RAW_OPERATIONS:
            return origin.rsplit(".", 1)[-1]
        if origin in SQUARE_ROOTS:
            guarded = bool(node.args) and _is(node.args[0], ("domain_guard",), bound, local, aliases)
            return None if guarded else "sqrt"
        if origin in STOP_GRADIENTS:
            return "stop_gradient"
        if origin in CUSTOM_RULES:
            return CUSTOM_RULES[origin]
        return None
    if isinstance(node.func, ast.Attribute):
        receiver, method = node.func.value, node.func.attr
        if method in CASTS and _is(receiver, ("comparison", "predicate"), bound, local, aliases):
            return "comparison_cast"
        if method in RAW_METHODS:
            return method.strip("_")
    return None


# -- the scan ------------------------------------------------------------------------------------------------------

@dataclass
class SwitchScan:
    """What the scan found over a package: the functions in scope, the families' hits by site, the owners passed,
    the nested switch evaluations, the node classes traced (with their configurations), and what it could not read."""

    package: Path
    scope: set[Function] = field(default_factory=set[Function])
    hits: dict[str, list[str]] = field(default_factory=dict[str, list[str]])
    owners: dict[str, list[str]] = field(default_factory=dict[str, list[str]])
    unreadable_owners: list[str] = field(default_factory=list[str])
    traced: dict[type[SimulationNode], set[str]] = field(default_factory=dict[type[SimulationNode], set[str]])
    unmatched: list[str] = field(default_factory=list[str])


def _in_package(package: Path, frame: Frame) -> str | None:
    return _relative_path(package, frame.file)


@cache
def _relative_path(package: Path, file: str) -> str | None:
    try:
        return Path(file).resolve().relative_to(package.resolve()).as_posix()
    except ValueError:
        return None


def _exempt(relative: str) -> bool:
    return any(relative == prefix or (prefix.endswith("/") and relative.startswith(prefix)) for prefix in EXEMPT)


def _is_switch_frame(frame: Frame) -> bool:
    return frame.file.replace("\\", "/").endswith("mime/core/switch.py")


def _under_vmap(frames: Sequence[Frame]) -> bool:
    """Whether a switch evaluated in these frames (innermost first) was called under ``vmap``: a batching frame
    rootward of its outermost switch frame."""
    outermost = max(i for i, frame in enumerate(frames) if _is_switch_frame(frame))
    return any(frame.file.replace("\\", "/").endswith(VMAP_FRAMES[0])
               or f"{frame.file.replace(chr(92), '/')}:{frame.function}".find(VMAP_FRAMES[1]) >= 0
               for frame in frames[outermost + 1:])


def scan_switches(steps: Sequence[TracedStep], package: Path) -> SwitchScan:
    """The scan of ``package`` (``src/mime``, or a lab's package directory) over the traced ``steps``."""
    scan = SwitchScan(package)
    nested: dict[tuple[str, str], int] = {}
    for step in steps:
        scan.traced.setdefault(step.node_class, set()).add(step.configuration)
        for within, equation in equations(step.jaxpr):
            frames = equation_frames(equation)
            for frame in frames:
                relative = _in_package(package, frame)
                if relative is not None and not _exempt(relative):
                    scan.scope.add(Function(relative, frame.function, frame.start))
            if not any(_is_switch_frame(frame) for frame in frames):
                continue
            transform = within[-1] if within else ("vmap" if _under_vmap(frames) else None)
            if transform is None:
                continue
            outermost = max(i for i, frame in enumerate(frames) if _is_switch_frame(frame))
            caller = next((frame for frame in frames[outermost + 1:] if _in_package(package, frame) is not None),
                          None)
            relative = _in_package(package, caller) if caller is not None else None
            if caller is not None and relative is not None and not _exempt(relative):
                nested[(f"{relative} {caller.function}", transform)] = 1
    _scan_source(scan)
    for (function, transform), _ in sorted(nested.items()):
        family = f"nested_{re.sub(r'[^a-z0-9]+', '_', transform.lower()).strip('_')}"
        count = sum(1 for site in scan.hits if site.startswith(f"{function} {family} "))
        scan.hits.setdefault(f"{function} {family} {count + 1}", []).append(f"{function.split()[0]}: under {transform}")
    return scan


def _scan_source(scan: SwitchScan) -> None:
    for function in sorted(scan.scope):
        path = scan.package / function.path
        found = definitions(path, function)
        if not found:
            scan.unmatched.append(f"{function.path}:{function.start} {function.name}")
            continue
        module = _module_name(scan.package, path)
        aliases = tree_aliases(_parsed(path), module)
        counts: dict[str, int] = {}

        def site(family: str, function: Function = function, counts: dict[str, int] = counts) -> str:
            counts[family] = counts.get(family, 0) + 1
            return f"{function.path} {function.name} {family} {counts[family]}"

        def where(node: ast.AST, function: Function = function) -> str:
            return f"{function.path}:{getattr(node, 'lineno', function.start)}"

        for definition in found[:1]:
            findings = scan_definition(definition, aliases, where, site)
            for hit in findings.hits:
                scan.hits.setdefault(hit.site, []).append(hit.where)
            for use in findings.owners:
                owners = [f"{use.owner}.low", f"{use.owner}.high"] if use.clip else [use.owner]
                for owner in owners:
                    scan.owners.setdefault(owner, []).append(use.where)
            scan.unreadable_owners += findings.unreadable_owners


def scan_problems(scan: SwitchScan, listed: RatchetList, declared: Mapping[str, str],
                  node_classes: Collection[type[SimulationNode]], tests: Path | None = None) -> list[str]:
    """Every failure of ``scan``: a node class no configuration traced; a function in scope whose source the scan
    cannot find; a hit not listed, and a listed site that is no longer one; a listed capping site naming neither a
    status nor a test that exists; an owner passed that no entry or shape declares (``declared``: each declared
    owner, with who declares it), one declared that nothing in scope passes, and one that cannot be read."""
    found = list(listed.problems)
    for cls in sorted(set(node_classes) - set(scan.traced), key=lambda c: f"{c.__module__}.{c.__qualname__}"):
        found.append(f"{cls.__module__}.{cls.__qualname__}: a registered node class that no configuration traces; "
                     f"add a configuration that builds it")
    found += [f"{where}: a function in scope whose source the scan cannot find, so JAX's source information did not "
              f"hold" for where in scan.unmatched]
    sites = listed_sites(listed)
    for site in sorted(set(scan.hits) - set(sites)):
        found.append(f"{', '.join(sorted(set(scan.hits[site])))}: {site.split()[2]} in {site.split()[1]}, which a "
                     f"node's step runs, is not written through mime.core.switch and is not on {listed.path}; write "
                     f"it through a switch function with its owner, or add the line '{site}  # <reason>'")
    for line in listed.lines:
        matched = ITEM.fullmatch(line.item)
        if matched is None:
            continue
        site = matched["site"]
        if site not in scan.hits:
            found.append(f"{listed.path}:{line.number}: {site!r} is no longer a raw site a node's step runs; delete "
                         f"this line")
        if matched["family"] in CAPPING and matched["status"] is None and matched["test"] is None:
            found.append(f"{listed.path}:{line.number}: a listed {matched['family']} names the status it caps its "
                         f"entry at ('caps <kind> <name> <version> cut|unverified') or the test that clears it "
                         f"('cleared_by <test id>')")
        if matched["family"] not in CAPPING and (matched["status"] or matched["test"]):
            found.append(f"{listed.path}:{line.number}: only a stop_gradient, custom_jvp or custom_vjp caps a status")
        if matched["test"] is not None and tests is not None and not _test_exists(tests, matched["test"]):
            found.append(f"{listed.path}:{line.number}: {matched['test']} is no test of {tests}")
    for owner in sorted(set(scan.owners) - set(declared)):
        found.append(f"{', '.join(sorted(set(scan.owners[owner])))}: the owner {owner!r} is passed and no entry or "
                     f"shape declares it; declare it in the gradient record of the entry whose law it is")
    for owner in sorted(set(declared) - set(scan.owners)):
        found.append(f"{declared[owner]} declares the switch {owner!r}, and nothing a node's step runs passes it")
    found += [f"{where}: an owner that is neither a string nor the function's own parameter passed on, which the scan "
              f"cannot read" for where in scan.unreadable_owners]
    return found


@dataclass(frozen=True)
class Declarer:
    """Who declares a switch: an entry (``kind name version``) or a shape (``shape <name>``), and its package."""

    label: str
    package: str


def declarations(records: Sequence[Registered]) -> list[tuple[Declarer, Declaration]]:
    """Every switch declared, with its declarer: each registered entry's own (a geometry kind's inherited ones are its
    shapes'), then each registered shape's."""
    found: list[tuple[Declarer, Declaration]] = []
    for record in records:
        gradient = record.cls.gradient
        for declared in gradient.switches if gradient is not None else ():
            found.append((Declarer(f"{record.kind.value} {record.name} {record.version}",
                                   record.cls.__module__.split(".")[0]), declared))
    for name in shape_names():
        package = type(shape_named(name)).__module__.split(".")[0]
        found += [(Declarer(f"shape {name}", package), declared) for declared in shape_switches(name)]
    return found


def declared_owners(records: Sequence[Registered], package: str = "mime") -> dict[str, str]:
    """Each owner the entries and shapes of ``package`` declare, with its declarer: what the scan of that package
    holds the owners passed against."""
    return {declared.owner: declarer.label for declarer, declared in declarations(records)
            if declarer.package == package}


def declaration_problems(records: Sequence[Registered]) -> list[str]:
    """Each owner declared more than once, and each owner declared outside MIME's namespace that does not carry its
    package's name as a prefix, so that a lab's owners cannot collide with MIME's."""
    found: list[str] = []
    seen: dict[str, tuple[Declarer, Declaration]] = {}
    for declarer, declared in declarations(records):
        held = seen.get(declared.owner)
        if held is not None and held[1] is not declared:
            found.append(f"{declarer.label}: the switch {declared.owner!r} is declared again, first by "
                         f"{held[0].label}; declare each owner once, by the entry or shape whose law it is")
        seen.setdefault(declared.owner, (declarer, declared))
        if declarer.package != "mime" and not declared.owner.startswith(f"{declarer.package}."):
            found.append(f"{declarer.label}: the switch {declared.owner!r} is declared outside MIME's namespace, and "
                         f"its owner does not start with its package's name, {declarer.package}.")
    return found


def _test_exists(tests: Path, test: str) -> bool:
    path, _, name = test.partition("::")
    file = tests.parent / path
    function = name.split("[", 1)[0].split("::")[-1]
    return file.is_file() and re.search(rf"^\s*def {re.escape(function)}\(", file.read_text(encoding="utf-8"),
                                        re.MULTILINE) is not None


__all__ = ["CAPPING", "EXEMPT", "ITEM", "RAW_OPERATIONS", "SWITCH_FUNCTIONS", "Declarer", "Frame", "Function",
           "ListedSite", "SwitchScan", "TracedStep", "caps", "declaration_problems", "declarations", "declared_owners",
           "equation_frames", "equations", "item_form", "listed_sites", "module_aliases", "scan_definition",
           "scan_problems", "scan_switches", "step_function", "traced_step", "tree_aliases"]
