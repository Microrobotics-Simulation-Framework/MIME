"""A declaration made a second time: what ``exp.add_*``, ``attach``, ``observe``, ``declare``, ``source``,
``set_controller``, ``licence`` and ``permission`` do when the experiment already holds one of the same key (a thing's
name, a law's or an observation's id, a value's or a source's path; the controller and the licence are one each).

- **The same as before** (``mime._content.same_content``: its type, its settings with their units and reasons, what
  it binds), in a script or a notebook: it is kept once, with a note.
- **Not the same, in a notebook** (``mime._notebook.in_notebook``): the last one run wins, with a note, where no other
  declaration names the earlier one. Where one does (a law, an observation, a value or a source naming a thing, a
  thing's setting naming another's direction, a source or the controller naming a law's or an observation's id),
  both are kept, with a note, and the build refuses the two as in a script, saying a cell has probably run twice:
  replacing it would bind what names it to a declaration it was not written against, unseen.
- **Not the same, in a script**: added, and the build refuses the two (``thing.duplicate``, ``id.duplicate``,
  ``value.duplicate``, ``source.duplicate``, ``controller.more-than-one``, ``terms.licence``), as before.

A permission has no key (an experiment holds one for each grant), so only the same one again is kept once, and another
is another permission."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, cast

from mime._content import same_content
from mime._notebook import in_notebook
from mime.build import (
    Attachment,
    ControllerDeclaration,
    Declaration,
    LicenceDeclaration,
    ObservationDeclaration,
    PermissionDeclaration,
    SourceDeclaration,
    ThingDeclaration,
    ValueDeclaration,
    entry_id,
)
from mime.errors._frames import warn_at_script
from mime.things import Handle, PrimPath, TypedPath

#: How many of the declarations that name an earlier one a note lists before it counts the rest.
LISTED = 3
#: The call that adds a thing of each kind, as a note names it.
ADDS = {"structure": "exp.add_structure(...)", "medium": "exp.add_medium(...)", "body": "exp.add_body(...)",
        "apparatus": "exp.add_apparatus(...)"}


@dataclass(frozen=True)
class Known:
    """How a note speaks of a declaration: the call that makes it and its noun. ``key`` is what a second one shares
    with it, ``None`` where there is none (a permission); ``named`` whether another declaration can name it, by the
    key's name."""

    call: str
    noun: str
    key: tuple[str, str] | None
    named: bool = False


def _path(written: object) -> str:
    return str(written) if isinstance(written, (TypedPath, str)) else repr(written)


def known_as(declaration: Declaration) -> Known | None:
    """``None`` for a declaration no call of a script makes (a loaded document's terms)."""
    if isinstance(declaration, ThingDeclaration):
        thing = declaration.thing
        name = cast(object, thing.name)  # a script is not type-checked
        kind = thing.kind.value
        if not isinstance(name, str):
            return Known(ADDS[kind], f"the {kind} named {name!r}", None)
        return Known(ADDS[kind], f"the {kind} {name!r}", ("thing", name), named=True)
    if isinstance(declaration, Attachment):
        owner = entry_id(type(declaration.law), declaration.bindings)
        return Known("exp.attach(...)", owner, ("attach", owner), named=True)
    if isinstance(declaration, ObservationDeclaration):
        observation = declaration.observation
        owner = entry_id(type(observation), observation.bindings())
        return Known("exp.observe(...)", owner, ("observe", owner), named=True)
    if isinstance(declaration, ValueDeclaration):
        path = _path(declaration.path)
        return Known("exp.declare(...)", f"the value {path}", ("value", path))
    if isinstance(declaration, SourceDeclaration):
        path = _path(declaration.path)
        return Known("exp.source(...)", f"the source of {path}", ("source", path))
    if isinstance(declaration, ControllerDeclaration):
        return Known("exp.set_controller(...)", "the controller", ("controller", ""))
    if isinstance(declaration, LicenceDeclaration):
        return Known("exp.licence(...)", "the experiment's licence", ("licence", ""))
    if isinstance(declaration, PermissionDeclaration):
        return Known("exp.permission(...)", "a permission", None)
    return None


def names_in(value: object, _open: frozenset[int] = frozenset()) -> Iterator[str]:
    """Every name ``value`` holds that can name a thing, a law or an observation: each string (a binding, a path, a
    setting written as a path), a handle's name and a typed path. A mapping's keys are its own labels and a reason is
    prose, so neither is read. A string that only happens to equal a thing's name is read as naming it: that keeps a
    declaration that could have been replaced, and never replaces one that could not."""
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, Handle):
        yield value.name
        return
    if isinstance(value, TypedPath):
        yield str(value)
        return
    if isinstance(value, PrimPath) or id(value) in _open:
        return
    inside = _open | {id(value)}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            if field.name != "reason":
                yield from names_in(getattr(value, field.name), inside)
    elif isinstance(value, Mapping):
        for held in cast(Mapping[object, object], value).values():
            yield from names_in(held, inside)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for held in cast(Any, value):
            yield from names_in(held, inside)


def _key_of(declaration: Declaration) -> tuple[str, str] | None:
    known = known_as(declaration)
    return known.key if known is not None else None


def _naming(held: list[Declaration], earlier: list[int], name: str) -> list[str]:
    """The nouns of the declarations of ``held``, other than ``earlier``, that name ``name``."""
    found: list[str] = []
    for index, other in enumerate(held):
        if index in earlier or not any(text == name or text.startswith(f"{name}.") for text in names_in(other)):
            continue
        known = known_as(other)
        found.append(known.noun if known is not None else type(other).__name__)
    return found


def _listed(nouns: list[str]) -> str:
    shown = ", ".join(nouns[:LISTED])
    return shown if len(nouns) <= LISTED else f"{shown} and {len(nouns) - LISTED} more"


def add(held: list[Declaration], declaration: Declaration) -> None:
    """Add ``declaration`` to ``held`` by the rule of this module, warning with its note at the script's line."""
    known = known_as(declaration)
    if known is None:
        held.append(declaration)
        return
    earlier = [index for index, other in enumerate(held)
               if type(other) is type(declaration) and (known.key is None or _key_of(other) == known.key)]
    if any(same_content(held[index], declaration) for index in earlier):
        warn_at_script(f"{known.call} declares {known.noun} again, the same as before: it is kept once", UserWarning)
        return
    if known.key is None or not earlier or not in_notebook():
        held.append(declaration)
        return
    naming = _naming(held, earlier, known.key[1]) if known.named else []
    if naming:
        held.append(declaration)
        warn_at_script(f"{known.call} declares {known.noun} again, not the same as before, and {_listed(naming)} "
                       f"name{'s' if len(naming) == 1 else ''} the earlier one: in a notebook the last one run replaces "
                       f"the earlier one only where nothing names it, so both are kept and the build will refuse them; "
                       f"run the cells again from the one that makes the experiment, or run them all", UserWarning)
        return
    held[earlier[0]] = declaration
    for index in reversed(earlier[1:]):
        del held[index]
    warn_at_script(f"{known.call} declares {known.noun} again, not the same as before: in a notebook the last one run "
                   f"wins, so it replaces the earlier one", UserWarning)


__all__ = ["Known", "add", "known_as", "names_in"]
