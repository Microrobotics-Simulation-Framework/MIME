"""The ``mime:`` USD schema's reference: what each attribute and applied schema in MIME's namespace means, held to
what MIME writes on a stage and what it reads from one.

The meanings are hand-written (``docs/reference/mime_usd.md``, one list item per name, ``- `mime:<name>`: <what it
means>``). The rest is generated: the constant that names each attribute in the code, and each registered geometry
kind's settings written as ``mime:geometry:<setting>``. The names are partly built where they are written
(``mime:{name}``, ``mime:geometry:{setting}``), so no scan of the code finds them all; the check reads them where they
land instead, in both directions:

- every name a fixture's stage authors, and every constant ``scene/_compose.py`` and ``scene/_read.py`` read a stage
  with, is documented (a ``mime:geometry:<setting>`` is documented by the pattern where it is a setting the kind on its
  prim writes);
- every documented name is authored on some fixture's stage, or is on ``docs/ratchets/usd_read_only.txt``, the names
  MIME reads from a stage and never writes, each with its reason.

``mime docs check`` runs the second half of the first direction and the list's form; the fixtures' stages are the
round-trip cases' (``tests/saved/test_usd_reference.py``), which plan them and write their stages, no graph built, and
need the fixtures' assets. Its edges: ``scene/_derived.py`` and ``viz/_visuals.py`` also read names from a stage and
are not among :data:`READERS`; the keys MIME writes in a layer's ``customLayerData`` (``mime:recording``,
``mime:visuals``) are not prims' names, and are outside the list."""

from __future__ import annotations

import ast
import importlib
import re
import typing
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mime.docs._lists import RatchetList, list_problems

SOURCE = Path("docs/reference/mime_usd.md")
#: The modules that read a stage: the constants they read it with must be documented.
READERS = (Path("src/mime/scene/_compose.py"), Path("src/mime/scene/_read.py"))
#: The modules of ``mime.scene`` that name what the stage writer writes, for the page's column of constants.
NAMERS = ("_compose", "_derived", "_stage")
NAME = re.compile(r"mime:[a-z_][a-z0-9_]*(?::[a-z_][a-z0-9_]*)*(?::<[a-z]+>)?|Mime[A-Za-z]+API")
SETTING_PATTERN = "mime:geometry:<setting>"
_ITEM = re.compile(r"- `(?P<name>[^`]+)`: (?P<meaning>.+)")


@dataclass(frozen=True)
class Authored:
    """A name authored on a prim of a stage, with the geometry kind that prim names (``mime:geometry`` and its version),
    where it names one."""

    name: str
    kind: tuple[str, str] | None = None


def documented(text: str, path: Path = SOURCE) -> tuple[dict[str, str], list[str]]:
    """The names the reference's source documents, each with its meaning (its continuation lines joined), and every
    item whose name is not a ``mime:`` name or is given twice."""
    found: dict[str, str] = {}
    problems: list[str] = []
    current: str | None = None
    for number, line in enumerate(text.splitlines(), start=1):
        match = _ITEM.fullmatch(line)
        if match is not None:
            name = match.group("name")
            if not NAME.fullmatch(name):
                problems.append(f"{path}:{number}: {name!r} is not a name of the mime: schema")
            elif name in found:
                problems.append(f"{path}:{number}: {name!r} is documented twice")
            found[name] = match.group("meaning").strip()
            current = name
        elif current is not None and line.startswith("  ") and line.strip():
            found[current] += " " + line.strip()
        else:
            current = None
    return found, problems


def _assigned(tree: ast.Module) -> dict[str, str]:
    """The module-level constants ``tree`` assigns a ``mime:`` name to, one by one or unpacked from a tuple."""
    found: dict[str, str] = {}
    for statement in tree.body:
        pairs: list[tuple[ast.expr, ast.expr]] = []
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Tuple) and isinstance(statement.value, ast.Tuple):
                    pairs += list(zip(target.elts, statement.value.elts, strict=False))
                else:
                    pairs.append((target, statement.value))
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            pairs.append((statement.target, statement.value))
        for target, value in pairs:
            if isinstance(target, ast.Name) and isinstance(value, ast.Constant) and isinstance(value.value, str) \
                    and NAME.fullmatch(value.value):
                found[target.id] = value.value
    return found


def read_constants(path: Path) -> dict[str, str]:
    """The ``mime:`` names the module at ``path`` reads with: each module-level constant it assigns a name to, and each
    it imports from MIME, by the constant's name."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = _assigned(tree)
    for statement in tree.body:
        if isinstance(statement, ast.ImportFrom) and (statement.module or "").startswith("mime."):
            module = importlib.import_module(statement.module or "")
            for alias in statement.names:
                held = getattr(module, alias.name, None)
                if isinstance(held, str) and NAME.fullmatch(held):
                    found[alias.asname or alias.name] = held
    return found


def constant_names() -> dict[str, list[str]]:
    """Each ``mime:`` name the code names by a constant, with the dotted constants that define it."""
    found: dict[str, list[str]] = {}
    for stem in NAMERS:
        dotted = f"mime.scene.{stem}"
        module = importlib.import_module(dotted)
        tree = ast.parse(Path(str(module.__file__)).read_text(encoding="utf-8"))
        for attribute, value in sorted(_assigned(tree).items()):
            found.setdefault(value, []).append(f"{dotted}.{attribute}")
    return {name: sorted(held) for name, held in found.items()}


def kind_settings() -> dict[tuple[str, str], tuple[str | None, dict[str, str]]]:
    """Each registered geometry kind (its name and version) with its native USD primitive (``None`` for none) and the
    settings it writes as ``mime:geometry:<setting>`` where it has none, each with its unit: its dimensioned settings
    that do not place it, and the whole numbers it holds, as counts."""
    import dataclasses

    from mime.registry import EntryKind, registered

    found: dict[tuple[str, str], tuple[str | None, dict[str, str]]] = {}
    for record in registered():
        if record.kind is not EntryKind.GEOMETRY:
            continue
        cls: Any = record.cls
        native = getattr(cls, "native", None)
        units: dict[str, str] = {}
        if native is None:
            from mime.units import unit_symbol

            for setting, declared in record.settings.items():
                if setting in cls.placement:
                    continue
                units[setting] = "1" if declared.dimension is None else unit_symbol(declared.dimension)
            hints = typing.get_type_hints(cls)
            units |= {field.name: "1" for field in dataclasses.fields(cls)
                      if field.name not in record.settings and hints.get(field.name) is int}
        found[(record.name, record.version)] = (getattr(native, "primitive", None), dict(sorted(units.items())))
    return found


def _accepted(item: Authored, names: Mapping[str, str],
              kinds: Mapping[tuple[str, str], tuple[str | None, Mapping[str, str]]]) -> str | None:
    """The documented name that documents ``item``: itself, or the setting pattern where it is a setting its prim's
    kind writes."""
    if item.name in names:
        return item.name
    prefix = SETTING_PATTERN.removesuffix("<setting>")
    held = kinds.get(item.kind) if item.kind is not None else None
    if SETTING_PATTERN in names and item.name.startswith(prefix) and held is not None:
        if item.name.removeprefix(prefix) in held[1]:
            return SETTING_PATTERN
    return None


def reference_problems(names: Mapping[str, str], read: Mapping[str, Mapping[str, str]], read_only: RatchetList,
                       kinds: Mapping[tuple[str, str], tuple[str | None, Mapping[str, str]]],
                       authored: Iterable[Authored] | None = None) -> list[str]:
    """Both directions over the documented ``names``: each constant of ``read`` (by reading module) and each name in
    ``authored`` documented, and each documented name authored or listed read only. Without ``authored`` (no fixture's
    stage at hand), the constants alone and the list's form."""
    problems = [f"{module}: {constant} = {value!r} is read from a stage and the mime: reference does not document "
                f"it; add '- `{value}`: <what it means>' to {SOURCE}"
                for module, held in sorted(read.items()) for constant, value in sorted(held.items())
                if value not in names]
    if authored is None:
        return problems + list(read_only.problems)
    seen: set[str] = set()
    for item in sorted(set(authored), key=lambda a: (a.name, a.kind or ("", ""))):
        documents = _accepted(item, names, kinds)
        if documents is None:
            on = f" on a prim of the geometry kind {' '.join(item.kind)}" if item.kind else ""
            problems.append(f"{item.name} is authored on a fixture's stage{on} and the mime: reference does not "
                            f"document it; add '- `{item.name}`: <what it means>' to {SOURCE}")
        else:
            seen.add(documents)
    absent = {name for name in names if name not in seen}
    problems += list_problems(read_only, absent, "documented in the mime: reference and authored on no fixture's "
                                                 "stage (a name MIME only reads)")
    return problems


def authored_on(text: str) -> set[Authored]:
    """Every ``mime:`` attribute authored on a prim of the stage whose layer is ``text``, and every ``mime:`` applied
    schema in a prim's ``apiSchemas``, each with its prim's geometry kind."""
    from mime.scene._pxr import Sdf, Usd

    layer = Sdf.Layer.CreateAnonymous(".usda")
    layer.ImportFromString(text)
    stage = Usd.Stage.Open(layer)
    found: set[Authored] = set()
    for prim in stage.Traverse():
        kind_name, kind_version = prim.GetAttribute("mime:geometry"), prim.GetAttribute("mime:geometry:version")
        kind = (str(kind_name.Get()), str(kind_version.Get())) if kind_name.HasAuthoredValue() and \
            kind_version.HasAuthoredValue() else None
        for attribute in prim.GetAuthoredAttributes():
            name = str(attribute.GetName())
            if name.startswith("mime:"):
                found.add(Authored(name, kind))
        schemas: Any = prim.GetMetadata("apiSchemas")
        applied: list[object] = list(schemas.ApplyOperations([])) if schemas is not None else []
        found |= {Authored(str(schema), kind) for schema in applied if str(schema).startswith("Mime")}
    return found


def reference_page_rows(names: Mapping[str, str]) -> tuple[list[tuple[str, ...]], list[tuple[str, ...]]]:
    """The page's two tables: each documented name with the constants that name it and its meaning; and each geometry
    kind with its native form or the settings it writes, with their units."""
    constants = constant_names()
    rows = [(f"`{name}`", ", ".join(f"`{c}`" for c in constants.get(name, [])) or "built where it is written",
             meaning) for name, meaning in sorted(names.items())]
    kinds = [(name, version, f"its native USD primitive, {native}" if native else
              ", ".join(f"`mime:geometry:{setting}` ({unit})" for setting, unit in settings.items()) or "none")
             for (name, version), (native, settings) in sorted(kind_settings().items())]
    return rows, kinds


__all__ = ["NAME", "READERS", "SETTING_PATTERN", "SOURCE", "Authored", "authored_on", "constant_names",
           "documented", "kind_settings", "read_constants", "reference_page_rows", "reference_problems"]
