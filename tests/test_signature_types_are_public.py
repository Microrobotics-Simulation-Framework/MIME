"""A user must be able to annotate what a public function hands them or takes (maintainer's rule, SURFACE-REDUCTION item
100: keep public every type a user would annotate), so no public signature names a class of ``mime.push``, ``mime.credentials``, ``mime.cli``, ``mime.scaffold``,
``mime.blocks``, ``mime.terms``, ``mime.errors``, ``mime.validity``, ``mime.walls``, ``mime.surrogates`` or
``mime.materials`` that no package exports.

The walk reads each name of the ``__all__`` of every public ``mime`` package: a function's parameters and return, a
class's fields and the signatures of its public methods and properties. A class of the tooling packages that an
annotation names must be in the ``__all__`` of some ``mime`` package (the other packages' own gaps wait for their
surface reduction, which extends ``TOOLING``). A type alias (``type Tier = Literal[...]``, or a name bound to a ``Literal`` or union) is a type a user annotates with, so it
counts as one: an alias a public signature names must be exported too. The controls plant a public function whose
return type is private, and one whose return is a private alias of each kind, and must find them."""

from __future__ import annotations

import ast
import functools
import importlib
import inspect
import pkgutil
import sys
import types
import typing
from collections.abc import Iterator
from typing import Any

import pytest

import mime
from mime.push._draft import Drafted

TOOLING = ("mime.push", "mime.credentials", "mime.cli", "mime.scaffold", "mime.blocks",
           "mime.terms", "mime.errors", "mime.validity", "mime.walls", "mime.surrogates", "mime.materials")


def exported() -> set[int]:
    """Identity of every object listed in the ``__all__`` of ``mime`` or of one of its public packages."""
    modules = [mime] + [importlib.import_module(f"mime.{each.name}") for each in pkgutil.iter_modules(mime.__path__)
                        if not each.name.startswith("_")]
    return {id(getattr(module, name)) for module in modules for name in getattr(module, "__all__", ())
            if hasattr(module, name)}


def names_in(text: str) -> Iterator[str]:
    """The dotted names an annotation written as text mentions."""
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:  # a string inside a Literal[...], not an annotation
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            yield node.id
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield from names_in(node.value)


@functools.cache
def imported_names(module: types.ModuleType) -> dict[str, tuple[Any, str]]:
    """What a module's source imports anywhere, including under ``TYPE_CHECKING`` (which its globals do not hold), each
    with the module it came from."""
    found: dict[str, tuple[Any, str]] = {}
    try:
        tree = ast.parse(inspect.getsource(module))
    except OSError:  # a module built in memory (the controls') has no source to read
        return found
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0 and node.module.startswith("mime"):
            for each in node.names:
                try:
                    found[each.asname or each.name] = (getattr(importlib.import_module(node.module), each.name),
                                                       node.module)
                except (ImportError, AttributeError):
                    continue
    return found


def is_alias(thing: object) -> bool:
    """A ``type X = ...`` alias, or a name bound to a subscripted ``Literal``, ``Union`` or ``X | Y`` (an alias a
    module writes as an assignment)."""
    return isinstance(thing, typing.TypeAliasType) or (
        typing.get_origin(thing) in (typing.Literal, typing.Union, types.UnionType))


def value_of(alias: typing.TypeAliasType) -> object:
    """What an alias names, or nothing where a name in it is imported only under ``TYPE_CHECKING``."""
    try:
        return alias.__value__
    except NameError:
        return None


def named(thing: object, module: str, name: str) -> tuple[str, object]:
    """(qualified name, object) of a class or alias an annotation names, ``module`` being where its name lives."""
    if isinstance(thing, type):
        return f"{thing.__module__}.{thing.__qualname__}", thing
    if isinstance(thing, typing.TypeAliasType):
        return f"{thing.__module__}.{thing.__name__}", thing
    return f"{module}.{name}", thing


def classes_named(annotation: object, module: types.ModuleType, seen: frozenset[int] = frozenset()
                  ) -> Iterator[tuple[str, object]]:
    """(qualified name, object) of each class and each alias (a type of the vocabulary a user annotates with) an
    annotation names; an alias is read through to what it names too."""
    if isinstance(annotation, str):
        scope = {**{key: value for key, value in imported_names(module).items()}, **{
            key: (value, module.__name__) for key, value in vars(module).items()}}
        for name in names_in(annotation):
            found = scope.get(name)
            if found is None:
                continue
            thing, home = found
            if isinstance(thing, type) or is_alias(thing):
                yield named(thing, home, name)
                if isinstance(thing, typing.TypeAliasType) and id(thing) not in seen:
                    yield from classes_named(value_of(thing), importlib.import_module(thing.__module__),
                                             seen | {id(thing)})
        return
    if isinstance(annotation, type):
        yield named(annotation, annotation.__module__, annotation.__name__)
    elif isinstance(annotation, typing.TypeAliasType) and id(annotation) not in seen:
        yield named(annotation, annotation.__module__, annotation.__name__)
        yield from classes_named(value_of(annotation), importlib.import_module(annotation.__module__),
                                 seen | {id(annotation)})
    for part in typing.get_args(annotation):
        yield from classes_named(part, module, seen)


def signatures(name: str, thing: object) -> Iterator[tuple[str, object, types.ModuleType]]:
    """(where, annotation, the module that wrote it) for each annotation of a public function or class."""
    if not (inspect.isclass(thing) or inspect.isfunction(thing)):
        return
    module = importlib.import_module(thing.__module__)
    if inspect.isclass(thing):
        for field, annotation in thing.__dict__.get("__annotations__", {}).items():
            if not field.startswith("_"):
                yield f"{name}.{field}", annotation, module
        for member, value in vars(thing).items():
            if not member.startswith("_") or member == "__init__":
                target = value.fget if isinstance(value, property) else value
                if inspect.isfunction(target):
                    yield from signatures(f"{name}.{member}", target)
    elif inspect.isfunction(thing):
        for parameter, annotation in thing.__annotations__.items():
            yield f"{name}({parameter})", annotation, module


#: Types a public signature names that no package exports and that wait for a decision. A line goes when its type is
#: exported or no signature names it: the test fails on a line that no longer applies, so the list only shrinks.
UNDECIDED = {
    "mime.push._sharing.Sharing": "decided with Manifest in the DUPLICATES work (rule 4)",
    "mime.validity._profiles.Judge": "decide() is named in one docs page's prose and no script calls it: its types follow its decision",
    "mime.validity._profiles.Finding": "as Judge: decide()'s parameter",
    "mime.validity._profiles.Action": "as Judge: decide()'s parameter",
    "mime.validity._profiles.Decision": "as Judge: decide()'s return",
    "mime.validity._profiles.FindingClasses": "using_tables() is named in one docs page's prose and no script calls it",
    "mime.validity._profiles.ProfileRules": "as FindingClasses: using_tables()'s parameter",
    "mime.validity._pending.Stamp": "results.Layout.stamp: plumbing of the results package, which waits for its own reduction",
    "mime.materials._values.ResolvedValue": "build.EntryPlan.values and ResolvedRegistration.values: plumbing of the build package, which waits for its reduction",
    "mime.materials._records.LoadedRecord": "saved.SavedRecord.records: plumbing of the saved package, which waits for its own reduction",
}


def private_types(public: set[int], names: dict[str, object]) -> list[str]:
    problems: list[str] = []
    for name, thing in names.items():
        for where, annotation, module in signatures(name, thing):
            for label, each in classes_named(annotation, module):
                if label.startswith("mime") and id(each) not in public:
                    problems.append(f"{where} names {label}, which no package exports")
    return problems


def public_names() -> dict[str, object]:
    """Every public name of every public package of ``mime``."""
    packages = [f"mime.{each.name}" for each in pkgutil.iter_modules(mime.__path__) if not each.name.startswith("_")]
    return {f"{package}.{name}": getattr(importlib.import_module(package), name)
            for package in packages for name in getattr(importlib.import_module(package), "__all__", ())}


def test_no_public_signature_names_a_type_of_the_tooling_packages_that_no_package_exports():
    names = public_names()
    assert sum(name.startswith(TOOLING) for name in names) > 5, "the walk found nothing to read"
    named = [each for each in private_types(exported(), names) if any(f"names {package}." in each for package in TOOLING)]
    unexpected = [each for each in named if not any(each.endswith(f"names {type_}, which no package exports")
                                                    for type_ in UNDECIDED)]
    assert unexpected == []
    stale = [type_ for type_ in UNDECIDED if not any(f"names {type_}," in each for each in named)]
    assert stale == [], f"no public signature names these any more: delete their lines {stale}"


def test_the_walk_finds_a_public_function_that_returns_a_private_type():
    """The control: ``Drafted`` is no package's export, so a function that returns it is found."""
    def planted() -> Drafted:  # a public function handing back an unexported type
        raise NotImplementedError

    found = private_types(exported(), {"planted": planted})
    assert found and "Drafted" in found[0], found
    assert not private_types(exported(), {"planted": lambda: None}), "the control found a defect where there is none"



def planted_module(monkeypatch: pytest.MonkeyPatch, source: str) -> types.ModuleType:
    """A module named like MIME's (``mime._planted``) holding ``source``, in ``sys.modules`` for the test."""
    module = types.ModuleType("mime._planted")
    module.__file__ = "mime/_planted.py"
    exec(source, module.__dict__)
    monkeypatch.setitem(sys.modules, "mime._planted", module)
    return module


def test_the_walk_finds_a_public_function_that_returns_a_private_alias(monkeypatch: pytest.MonkeyPatch):
    """The controls: a ``type`` statement's alias and a name bound to a ``Literal`` are no package's export, so a function
    that returns either is found, and one that returns a plain ``str`` is not."""
    module = planted_module(monkeypatch, "from __future__ import annotations\nfrom typing import Literal\n"
                            "type Tier = Literal['open', 'closed']\nKind = Literal['a', 'b']\n"
                            "def by_type() -> Tier: ...\ndef by_literal() -> Kind: ...\ndef plain() -> str: ...\n"
                            "def deep() -> tuple[Tier, ...]: ...\n")
    for function, name in (("by_type", "Tier"), ("by_literal", "Kind"), ("deep", "Tier")):
        found = private_types(exported(), {function: getattr(module, function)})
        assert found and f"mime._planted.{name}" in found[0], (function, found)
    assert not private_types(exported(), {"plain": module.plain}), "the control found a defect where there is none"
