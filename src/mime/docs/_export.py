"""The registry export, ``mime registry --json``: every registered entry, MIME's and those of each installed
``mime.entries`` package, with every ``ClassVar`` and dataclass field it declares, each with its type, its default
and its value. It is what the pages are generated from, so it imports nothing from ``mime.testing`` and runs on a
base install: a namespace that needs the ``verify`` extra is listed as not loaded.

A process lists what its imports registered, so the export is taken in a fresh process (:func:`registry_export`), where
no entry a test registered on its own is found."""

from __future__ import annotations

import dataclasses
import importlib.metadata
import inspect
import json
import math
import re
import subprocess
import sys
import types
import typing
from collections.abc import Mapping
from enum import Enum
from types import FunctionType
from typing import Any, cast

from mime._extras import VERIFY_EXTRA_MISSING
from mime.registry import Entry, NotEstablished, Registered, SettingGroup, registered
from mime.units import Dimension, Quantity, unit_symbol
from mime.vocabulary import versions

ENTRY_POINT_GROUP = "mime.entries"
_CLASS_VAR = re.compile(r"(?:typing\.)?ClassVar\b")

type JSON = None | bool | int | float | str | list[JSON] | dict[str, JSON]


class Unencodable(TypeError):
    """A value of a type the export has no rule for."""


def load_namespaces() -> dict[str, str]:
    """Import MIME and each installed ``mime.entries`` namespace, one by one: the namespaces not loaded because the
    ``verify`` extra is missing, each with that message. Any other failure to import one is raised."""
    importlib.import_module("mime")
    not_loaded: dict[str, str] = {}
    for point in sorted(importlib.metadata.entry_points(group=ENTRY_POINT_GROUP), key=lambda p: p.name):
        try:
            point.load()
        except ImportError as error:
            if str(error) != VERIFY_EXTRA_MISSING:
                raise
            not_loaded[point.name] = str(error)
    return not_loaded


def encode(value: object) -> JSON:
    """``value`` as JSON, by a rule for each kind of value an entry declares; :class:`Unencodable` for any other."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, NotEstablished):
        return {"not_established": True}
    if isinstance(value, Enum):
        return encode(cast(object, value.value))
    if isinstance(value, Dimension):
        return {"dimension": str(value), "unit": unit_symbol(value)}
    if isinstance(value, Quantity):
        return {"value": encode(value.value), "unit": unit_symbol(value.dimension)}
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, (tuple, list)):
        items = cast(tuple[object, ...], value)
        fields = cast(tuple[str, ...] | None, getattr(items, "_fields", None))
        if fields is not None:
            return {name: encode(cast(object, getattr(items, name))) for name in fields}
        return [encode(item) for item in items]
    if dataclasses.is_dataclass(value):
        instance = cast(Any, value)
        return {f.name: encode(cast(object, getattr(instance, f.name))) for f in dataclasses.fields(instance)}
    if isinstance(value, Mapping):
        pairs = cast(Mapping[object, object], value).items()
        return {str(encode(key)): encode(item) for key, item in pairs}
    if isinstance(value, (set, frozenset)):
        return sorted((encode(item) for item in cast(frozenset[object], value)), key=json.dumps)
    raise Unencodable(f"{type(value).__module__}.{type(value).__qualname__}")


def _encoded(value: object) -> JSON:
    try:
        return encode(value)
    except Unencodable as error:
        return {"unencodable": str(error)}


def _annotations(cls: type) -> dict[str, str]:
    """The annotations ``cls`` itself declares, as written."""
    if sys.version_info >= (3, 14):
        import annotationlib

        found = annotationlib.get_annotations(cls, format=annotationlib.Format.STRING)
    else:
        found = inspect.get_annotations(cls)
    return {name: hint if isinstance(hint, str) else inspect.formatannotation(hint) for name, hint in found.items()}


def class_vars(cls: type[Entry]) -> dict[str, tuple[type, str]]:
    """Every ``ClassVar`` ``cls`` declares or inherits, with the class that first declares it and its type."""
    found: dict[str, tuple[type, str]] = {}
    for held in reversed(cls.__mro__):
        for name, hint in _annotations(held).items():
            if _CLASS_VAR.match(hint) and name not in found:
                found[name] = (held, hint)
    return found


def _members(hint: object) -> tuple[object, ...]:
    if typing.get_origin(hint) in (typing.Union, types.UnionType):
        return typing.get_args(hint)
    return (hint,)


def overrides_plan(cls: type[Entry]) -> bool:
    """Whether ``cls`` writes its own ``plan``, which can add to what its class variables declare: the export reads
    declarations, so it cannot say what such a ``plan`` adds."""
    return any("plan" in vars(held) for held in cls.__mro__ if held.__module__ != Entry.__module__)


def entry_record(record: Registered) -> dict[str, JSON]:
    """One entry as the export writes it."""
    cls = record.cls
    held: dict[str, JSON] = {}
    for name, (declared_by, hint) in sorted(class_vars(cls).items()):
        given = cast(object, vars(declared_by).get(name, dataclasses.MISSING))
        held[name] = {"type": hint, "declared_by": encode(declared_by),
                      "default": {"unset": True} if given is dataclasses.MISSING else _encoded(given),
                      "value": _encoded(getattr(cls, name)) if hasattr(cls, name) else {"unset": True}}
    fields: dict[str, JSON] = {}
    hints = typing.get_type_hints(cls)
    for item in dataclasses.fields(cls):
        if item.default is not dataclasses.MISSING:
            default: JSON = _encoded(item.default)
        elif item.default_factory is not dataclasses.MISSING:
            default = _encoded(item.default_factory())
        else:
            default = {"required": True}
        groups = [member.group for member in _members(hints.get(item.name))
                  if isinstance(member, type) and issubclass(member, SettingGroup)]
        fields[item.name] = {"type": item.type if isinstance(item.type, str) else inspect.formatannotation(item.type),
                             "default": default, "group": groups[0] if groups else None}
    settings: dict[str, JSON] = {
        name: {"dimension": None if declared.dimension is None else encode(declared.dimension),
               "kind": None if declared.kind is None else declared.kind.name, "optional": declared.optional}
        for name, declared in sorted(record.settings.items())}
    return {"kind": record.kind.value, "name": record.name, "version": record.version,
            "namespace": record.namespace, "defined_in": record.defined_in, "class_vars": held, "fields": fields,
            "settings": settings, "overrides_plan": overrides_plan(cls)}


def export(not_loaded: Mapping[str, str]) -> dict[str, JSON]:
    """The export of what this process has registered."""
    entries: list[JSON] = [entry_record(record) for record in registered()]
    return {"entries": entries, "not_loaded": dict(sorted(not_loaded.items())),
            "namespaces": dict(versions().items())}


def export_json() -> str:
    """The export, as ``mime registry --json`` prints it: MIME and every installed namespace loaded first."""
    return json.dumps(export(load_namespaces()), indent=2, sort_keys=True)


def registry_export() -> dict[str, Any]:
    """The export taken in a fresh process, as the CLI takes it."""
    done = subprocess.run([sys.executable, "-m", "mime.cli", "registry", "--json"], capture_output=True, text=True,
                          check=True)
    return cast(dict[str, Any], json.loads(done.stdout))


def _is_data(value: object) -> bool:
    return not isinstance(value, (FunctionType, classmethod, staticmethod, property)) and \
        not inspect.isdatadescriptor(value) and not inspect.ismethoddescriptor(value)


def completeness_problems(cls: type[Entry], record: Mapping[str, Any]) -> list[str]:
    """Every value an entry class holds at class level that the export ``record`` of it does not carry: an attribute
    that is not a declared ``ClassVar`` (the export reads declarations, so it cannot see one), and a ``ClassVar`` whose
    value the export cannot encode. Read from the class itself, not from what the export read."""
    carried = cast(Mapping[str, Any], record.get("class_vars", {}))
    fields = {item.name for item in dataclasses.fields(cls)} if dataclasses.is_dataclass(cls) else set[str]()
    label = f"{record.get('kind')} {record.get('name')} {record.get('version')} ({cls.__module__}.{cls.__qualname__})"
    found: list[str] = []
    seen: set[str] = set()
    for held in cls.__mro__:
        if not issubclass(held, Entry):
            continue
        for name, value in vars(held).items():
            if name.startswith("_") or name in fields or name in seen or not _is_data(value):
                continue
            seen.add(name)
            if name not in carried:
                found.append(f"{label}: {held.__qualname__}.{name} is held at class level and is not a declared "
                             f"ClassVar, so the registry export does not carry it: declare it {name}: "
                             f"ClassVar[...] where it is first set")
    for name, item in carried.items():
        value = cast(Mapping[str, Any], item).get("value")
        if isinstance(value, Mapping) and "unencodable" in value:
            found.append(f"{label}: ClassVar {name} holds a {cast(Mapping[str, str], value)['unencodable']}, which "
                         f"the registry export has no rule for")
    return found


__all__ = ["ENTRY_POINT_GROUP", "Unencodable", "class_vars", "completeness_problems", "encode", "entry_record",
           "export", "export_json", "load_namespaces", "overrides_plan", "registry_export"]
