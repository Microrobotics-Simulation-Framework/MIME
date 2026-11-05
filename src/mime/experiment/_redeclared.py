"""A declaration made a second time: what ``exp.add_*``, ``attach``, ``observe``, ``declare``, ``source``,
``set_controller``, ``licence`` and ``permission`` do when the experiment already holds one of the same key (a thing's
name, a law's or an observation's id, a value's or a source's path; the controller and the licence are one each).

- **The same as before** (``mime._content.same_content``: its type, its settings with their units and reasons, what
  it binds), in a script or a notebook: it is kept once, with a note.
- **Not the same, in a notebook or IPython** (``mime._notebook.in_notebook``): the last one run wins, with a note.
  - *A structure, medium, body or apparatus* (the same name and kind) replaces the earlier one in its place, and
    whatever names it follows it to the new one. Nothing is rewritten: every declaration refers to a thing by its
    name (a law's binding, a handle, a typed path, an observation's binding, the controller's reads and writes, a
    value's or a source's path, a medium's ``fills``, a setting that names a direction), and the build resolves a
    name against the things the experiment holds, so there is only the new one to find. The one note names the thing
    replaced and every declaration that holds its name, by what it is (a law, an observation, the controller, a
    value, a source, another thing). A declaration that holds the earlier *object* instead of its name would keep the
    old parameters under the new name, so then both are kept and the build refuses the two; so it does for a name
    of another kind (a body declared again as an apparatus), which no replacement can serve. A value declared of the
    thing still applies to the new one and overrides what the new declaration brings, and the note says so.
  - *A law or an observation*, where no other declaration names its id, replaces the earlier one with a note; where
    one does (a value, a source, the controller), both are kept and the build refuses the two, saying a line or cell
    has probably run twice.
  - *A value, a source, the controller or the licence* replaces the earlier one; nothing names them.
  - *The note is also kept.* A warning is all a warnings filter removes, so each replacement is also recorded on the
    experiment (``replaced``), which ``plan()`` and the build report show (``PlanSet.replaced``,
    ``BuildReport.replaced``). The saved form is not changed: it holds the final declarations. The note is raised
    before anything is changed, so a filter that turns a warning into an error leaves the experiment as it was.
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
from mime._notebook import WHERE, in_notebook
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
from mime.things import Apparatus, Body, Handle, Medium, PrimPath, Structure, TypedPath

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
    #: What a note calls it where it names this declaration as one that holds another's name, where its noun alone
    #: would not say what it is (a law or an observation is only its id).
    holder: str = ""


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
        return Known("exp.attach(...)", owner, ("attach", owner), named=True, holder=f"the law {owner}")
    if isinstance(declaration, ObservationDeclaration):
        observation = declaration.observation
        owner = entry_id(type(observation), observation.bindings())
        return Known("exp.observe(...)", owner, ("observe", owner), named=True, holder=f"the observation {owner}")
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
    """How a note calls each declaration of ``held``, other than ``earlier``, that holds ``name``."""
    found: list[str] = []
    for index, other in enumerate(held):
        if index in earlier or not any(text == name or text.startswith(f"{name}.") for text in names_in(other)):
            continue
        known = known_as(other)
        found.append(known.holder or known.noun if known is not None else type(other).__name__)
    return found


def things_in(value: object, _open: frozenset[int] = frozenset()) -> Iterator[Structure | Medium | Body | Apparatus]:
    """Every thing (the object itself, not its name) that ``value`` holds."""
    if isinstance(value, (Structure, Medium, Body, Apparatus)):
        yield value
        return
    if isinstance(value, (str, bytes, Handle, TypedPath, PrimPath)) or id(value) in _open:
        return
    inside = _open | {id(value)}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            yield from things_in(getattr(value, field.name), inside)
    elif isinstance(value, Mapping):
        for held in cast(Mapping[object, object], value).values():
            yield from things_in(held, inside)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for held in cast(Any, value):
            yield from things_in(held, inside)


def _other_kinds(held: list[Declaration], earlier: list[int], declaration: ThingDeclaration) -> list[str]:
    """The kinds, other than ``declaration``'s, of the things ``earlier`` holds under its name."""
    kinds = {cast(ThingDeclaration, held[index]).thing.kind for index in earlier}
    return sorted(kind.value for kind in kinds - {declaration.thing.kind})


def _holding_the_object(held: list[Declaration], earlier: list[int], name: str) -> list[str]:
    """How a note calls each declaration of ``held``, other than ``earlier``, that holds the thing named ``name`` itself."""
    found: list[str] = []
    for index, other in enumerate(held):
        if index not in earlier and any(thing.name == name for thing in things_in(other)):
            known = known_as(other)
            found.append(known.holder or known.noun if known is not None else type(other).__name__)
    return found


def _listed(nouns: list[str]) -> str:
    shown = ", ".join(nouns[:LISTED])
    return shown if len(nouns) <= LISTED else f"{shown} and {len(nouns) - LISTED} more"


def _kept_beside(known: Known, why: str, held: list[Declaration], declaration: Declaration) -> None:
    warn_at_script(f"{known.call} declares {known.noun} again, not the same as before, and {why}: both are kept and "
                   f"the build will refuse them; run the lines or cells again from the one that makes the experiment, "
                   f"or run them all", UserWarning)
    held.append(declaration)  # after the note: a warning raised as an error leaves the experiment as it was


def _declared_values(held: list[Declaration], earlier: list[int], name: str) -> list[str]:
    """Each value declared of ``name`` (``water.density``), as ``the value water.density (1000.0 kg/m^3)``: one that
    still applies to a thing declared again under the name, over whatever the new declaration brings."""
    found: list[str] = []
    for index, other in enumerate(held):
        if index in earlier or not isinstance(other, ValueDeclaration):
            continue
        path = _path(other.path)
        if path == name or path.startswith(f"{name}."):
            found.append(f"the value {path} ({other.value})")
    return found


def _still_applies(values: list[str], kind: str) -> str:
    """The clause of a note that says a declared value of the thing replaced still applies, and wins."""
    if not values:
        return ""
    verbs = "still apply and override" if len(values) > 1 else "still applies and overrides"
    return f"; of these, {_listed(values)} {verbs} what the new {kind} brings"


def _replace_thing(held: list[Declaration], earlier: list[int], declaration: ThingDeclaration, known: Known,
                   name: str, replaced: list[str]) -> None:
    """A thing declared again, changed, in a notebook: it takes the earlier one's place, and what names it follows
    it (by its name, which is all a declaration holds of a thing), with one note that names each. The note is raised
    first and recorded after the change."""
    apart = _other_kinds(held, earlier, declaration)
    if apart:
        warn_at_script(f"{known.call} declares {known.noun} again, but an earlier {name!r} is of another kind "
                       f"({' and '.join(apart)}), and a name has one kind: nothing is replaced, both are kept and the "
                       f"build will refuse them; run the lines or cells again from the one that makes the experiment, "
                       f"or run them all", UserWarning)
        held.append(declaration)
        return
    objects = _holding_the_object(held, earlier, name)
    if objects:
        _kept_beside(known, f"{', '.join(objects)} hold{'s' if len(objects) == 1 else ''} the earlier one itself and "
                            f"not its name, so replacing it would leave them its old settings under the new name", held,
                     declaration)
        return
    naming = _naming(held, earlier, name)
    follows = (f"; what holds its name now finds the new one: {', '.join(naming)}" if naming else
               "; nothing else holds its name")
    note = (f"{known.call} declares {known.noun} again, not the same as before: in {WHERE} the last one run wins, so "
            f"it replaces the earlier one{follows}"
            f"{_still_applies(_declared_values(held, earlier, name), declaration.thing.kind.value)}")
    _replace(held, earlier, declaration, note, replaced)


def _replace(held: list[Declaration], earlier: list[int], declaration: Declaration, note: str,
             replaced: list[str]) -> None:
    """``declaration`` takes the first of ``earlier``'s place and the rest go, with ``note`` raised first (a filter that
    makes it an error leaves ``held`` and ``replaced`` as they were) and then recorded in ``replaced``."""
    warn_at_script(note, UserWarning)
    held[earlier[0]] = declaration
    for index in reversed(earlier[1:]):
        del held[index]
    replaced.append(note)


def add(held: list[Declaration], declaration: Declaration, replaced: list[str]) -> None:
    """Add ``declaration`` to ``held`` by the rule of this module, warning with its note at the script's line, and
    record in ``replaced`` each note of a declaration that replaced an earlier one."""
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
    if isinstance(declaration, ThingDeclaration):
        _replace_thing(held, earlier, declaration, known, known.key[1], replaced)
        return
    naming = _naming(held, earlier, known.key[1]) if known.named else []
    if naming:
        _kept_beside(known, f"{_listed(naming)} name{'s' if len(naming) == 1 else ''} the earlier one, and in {WHERE} "
                            f"a law or an observation is replaced only where nothing names it", held, declaration)
        return
    _replace(held, earlier, declaration, f"{known.call} declares {known.noun} again, not the same as before: in "
                                         f"{WHERE} the last one run wins, so it replaces the earlier one", replaced)


__all__ = ["Known", "add", "known_as", "names_in", "things_in"]
