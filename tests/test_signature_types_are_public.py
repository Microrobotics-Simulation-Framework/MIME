"""A user must be able to annotate what a public function hands them or takes (maintainer's rule, SURFACE-REDUCTION item
100: keep public every type a user would annotate), so no public signature names a class of ``mime.push``, ``mime.credentials``, ``mime.cli``, ``mime.scaffold`` or
``mime.blocks`` that no package exports.

The walk reads each name of the ``__all__`` of every public ``mime`` package: a function's parameters and return, a
class's fields and the signatures of its public methods and properties. A class of the tooling packages that an
annotation names must be in the ``__all__`` of some ``mime`` package (the other packages' own gaps wait for their
surface reduction, which extends ``TOOLING``). The control
plants a public function whose return type is private and must be found.

The paradigm packages ``mime.units``, ``mime.things``, ``mime.fields``, ``mime.dynamics``, ``mime.interactions`` and
``mime.drives`` are read the same way; a class a public submodule exports (``mime.dynamics.flow``) is exported."""

from __future__ import annotations

import ast
import functools
import importlib
import inspect
import pkgutil
import types
import typing
from collections.abc import Iterator
from typing import Any

import mime
from mime.push._draft import Drafted

TOOLING = ("mime.push", "mime.credentials", "mime.cli", "mime.scaffold", "mime.blocks")

#: The small paradigm packages (SURFACE-REDUCTION, PR 4). Their public subpackages (``mime.dynamics.flow``) count as
#: places a class is exported from, as they are paths a script writes.
PARADIGM = ("mime.units", "mime.things", "mime.fields", "mime.dynamics", "mime.interactions", "mime.drives")
GUARDED = TOOLING + PARADIGM


def public_submodules(package: str) -> Iterator[types.ModuleType]:
    """The modules of a package none of whose path parts starts with an underscore."""
    module = importlib.import_module(package)
    for each in pkgutil.walk_packages(getattr(module, "__path__", ()), f"{package}."):
        if not any(part.startswith("_") for part in each.name.split(".")):
            yield importlib.import_module(each.name)


def exported() -> set[int]:
    """Identity of every object listed in the ``__all__`` of ``mime``, of one of its public packages, or of a public
    module of a paradigm package."""
    modules = [mime] + [importlib.import_module(f"mime.{each.name}") for each in pkgutil.iter_modules(mime.__path__)
                        if not each.name.startswith("_")]
    modules += [module for package in PARADIGM for module in public_submodules(package)]
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
def imported_names(module: types.ModuleType) -> dict[str, Any]:
    """What a module's source imports anywhere, including under ``TYPE_CHECKING`` (which its globals do not hold)."""
    found: dict[str, Any] = {}
    for node in ast.walk(ast.parse(inspect.getsource(module))):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0 and node.module.startswith("mime"):
            for each in node.names:
                try:
                    found[each.asname or each.name] = getattr(importlib.import_module(node.module), each.name)
                except (ImportError, AttributeError):
                    continue
    return found


def classes_named(annotation: object, module: types.ModuleType) -> Iterator[type]:
    if isinstance(annotation, str):
        scope = {**imported_names(module), **vars(module)}
        for name in names_in(annotation):
            if isinstance(scope.get(name), type):
                yield scope[name]
        return
    if isinstance(annotation, type):
        yield annotation
    for part in typing.get_args(annotation):
        yield from classes_named(part, module)


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
}

#: The paradigm packages' own, kept apart from the tooling list above. Each is judged by one line: would a user write
#: it in an annotation? These are not exported today and no script, doc, example or plugin names one: they wait for the
#: package that hands them out.
UNDECIDED_PARADIGM = {
    "mime.fields._solved.ExchangeFrame": "plumbing of the flow seam (BuildContext.exchanges, FVM.exchange_frame): decided with the registry and the flow's reduction",
    "mime.fields._solved.SolvedField": "plumbing of the flow seam (FVM.field_for, Womersley.field): decided with the registry and the flow's reduction",
    "mime.things._parts.PartMass": "named only by materials.PartPlan, plumbing of the resolver that leaves mime.materials with the policy PR (#351): delete this line when both have merged",
    "mime.things._handles._Names": "what ApparatusHandle.commands and BodyHandle.initial hand back, underscore-named by design: the maintainer decides whether a user annotates the namespace of typed paths",
}


def private_types(public: set[int], names: dict[str, object]) -> list[str]:
    problems: list[str] = []
    for name, thing in names.items():
        for where, annotation, module in signatures(name, thing):
            for each in classes_named(annotation, module):
                if each.__module__.startswith("mime") and id(each) not in public:
                    problems.append(f"{where} names {each.__module__}.{each.__qualname__}, which no package exports")
    return problems


def public_names() -> dict[str, object]:
    """Every public name of every public package of ``mime``."""
    packages = [f"mime.{each.name}" for each in pkgutil.iter_modules(mime.__path__) if not each.name.startswith("_")]
    return {f"{package}.{name}": getattr(importlib.import_module(package), name)
            for package in packages for name in getattr(importlib.import_module(package), "__all__", ())}


def test_no_public_signature_names_a_type_of_the_tooling_packages_that_no_package_exports():
    names = public_names()
    assert sum(name.startswith(TOOLING) for name in names) > 5, "the walk found nothing to read"
    named = [each for each in private_types(exported(), names) if any(f"names {package}." in each for package in GUARDED)]
    waiting = UNDECIDED | UNDECIDED_PARADIGM
    unexpected = [each for each in named if not any(each.endswith(f"names {type_}, which no package exports")
                                                    for type_ in waiting)]
    assert unexpected == []
    stale = [type_ for type_ in waiting if not any(f"names {type_}," in each for each in named)]
    assert stale == [], f"no public signature names these any more: delete their lines {stale}"


def test_the_walk_finds_a_public_function_that_returns_a_private_type():
    """The control: ``Drafted`` is no package's export, so a function that returns it is found."""
    def planted() -> Drafted:  # a public function handing back an unexported type
        raise NotImplementedError

    found = private_types(exported(), {"planted": planted})
    assert found and "Drafted" in found[0], found
    assert not private_types(exported(), {"planted": lambda: None}), "the control found a defect where there is none"



def test_the_walk_reads_the_paradigm_packages_and_counts_a_public_submodule_as_an_export():
    """The control for ``PARADIGM``: a function that returns ``PartMass`` (``mime.things``, exported by nothing) is found,
    and one returning ``IMEXEulerScheme`` (exported by ``mime.dynamics.flow`` alone) is not."""
    from mime.dynamics.flow import IMEXEulerScheme
    from mime.things._parts import PartMass

    def planted_private() -> PartMass:
        raise NotImplementedError

    def planted_exported_by_a_submodule() -> IMEXEulerScheme:
        raise NotImplementedError

    found = private_types(exported(), {"planted": planted_private})
    assert found and any(f"names {package}." in found[0] for package in GUARDED), found
    assert not private_types(exported(), {"planted": planted_exported_by_a_submodule})
