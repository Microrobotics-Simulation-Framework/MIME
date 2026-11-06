"""``exp.remove(...)``: a declaration taken out of the experiment, by hand, in a script or a notebook.

What it removes, by the word ``kind=`` names it: a ``thing`` (by its name or handle), a ``law`` (by its id, which
``attach`` returned), an ``observation`` (by its id or handle), a declared ``value`` or a ``source`` (by the path), or
the ``controller`` (which has no name: ``exp.remove(kind="controller")``). A name that is two kinds at once (a path
with a declared value and a source) is refused until ``kind=`` says which; one that is none is refused with the
candidates.

**Refused while something still names it.** A thing is named by the laws bound to it, the observations of it, the
values and sources declared of it, the controller's reads and writes, a medium that fills it, and another thing that
holds it; a law by the sources written under its id and the controller; an observation by the controller's reads. The
refusal lists each, with the cell or line that declared it (``exp.remove``'s own call, for a declaration made another
way, says the document), and the call that removes it. Nothing is taken out with the thing: it is the reader's to
remove, or to change.

**Recorded like a replacement** (``mime.experiment._redeclared``): a note, raised before anything changes, and kept on
the experiment (``removed``), which ``plan()`` and the build report show whatever the warnings filter. The saved form
holds what was left.

Removing and declaring again under the name is how a thing's kind is changed (a body that becomes an apparatus), which a
second declaration of a name of another kind never does."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from mime.build import Declaration
from mime.build._refusals import ArgumentRefused, Refusal, RefusalClass
from mime.experiment._redeclared import (
    called,
    holding_the_object_at,
    named_by,
    known_as,
)
from mime.errors._frames import warn_at_script
from mime.things import did_you_mean

#: The kinds ``kind=`` names, with the key ``known_as`` gives the declaration of each.
KEYS = {"thing": "thing", "law": "attach", "observation": "observe", "value": "value", "source": "source",
        "controller": "controller"}
_WORDS = {key: word for word, key in KEYS.items()}


@dataclass(frozen=True)
class Found:
    """The declarations ``exp.remove`` was asked for: their kind, their name and where they are held."""

    kind: str
    name: str
    places: list[int]


def _removable(held: list[Declaration], kind: str | None) -> list[Found]:
    """Every removable name of ``held`` (of ``kind``, where given), each with its places."""
    found: dict[tuple[str, str], list[int]] = {}
    for index, declaration in enumerate(held):
        known = known_as(declaration)
        if known is None or known.key is None or known.key[0] not in _WORDS:
            continue
        word = _WORDS[known.key[0]]
        if kind is None or word == kind:
            found.setdefault((word, known.key[1]), []).append(index)
    return [Found(word, name, places) for (word, name), places in found.items()]


def _how_to_remove(found: Found) -> str:
    if found.kind == "controller":
        return "exp.remove(kind='controller')"
    if found.kind in ("value", "source"):
        return f"exp.remove({found.name!r}, kind={found.kind!r})"
    return f"exp.remove({found.name!r})"


def referrers(held: list[Declaration], found: Found) -> list[int]:
    """The places of the declarations, other than ``found``'s, that name it: by its name or id, or (a thing) by
    holding the thing itself."""
    if found.kind not in ("thing", "law", "observation"):
        return []
    named = named_by(held, found.places, found.name)
    if found.kind == "thing":
        named += [index for index in holding_the_object_at(held, found.places, found.name) if index not in named]
    return sorted(index for index in named if _kind_and_name(held[index]) is not None)  # a licence and a permission
    # hold prose and file hashes, and are never listed as naming anything


def remove(held: list[Declaration], target: str | None, kind: str | None, place_of: Callable[[Declaration], str],
           removed: list[str]) -> tuple[str, str]:
    """Remove from ``held`` the declarations ``target`` names (``None``: the controller), of ``kind`` where given.
    Raises ``ArgumentRefused`` when none is named, when it is two kinds, or when others still name it. Returns what was
    removed (as a note names it) and the note, which is also appended to ``removed`` once the removal is made."""
    if kind is not None and kind not in KEYS:
        raise ArgumentRefused([Refusal(
            "remove.unknown", RefusalClass.USER, f"remove(kind={kind!r}) names no kind of declaration",
            f"name one: exp.remove('<name>', kind='law'), kind= one of {', '.join(repr(word) for word in KEYS)}")])
    if target is None:
        if kind not in (None, "controller"):
            raise ArgumentRefused([Refusal(
                "remove.unknown", RefusalClass.USER, f"remove(kind={kind!r}) needs the name of what to remove",
                f"name it: exp.remove('<name>', kind={kind!r}); only the controller has no name")])
        kind = "controller"
    matches = [found for found in _removable(held, kind) if kind == "controller" or found.name == target]
    if not matches:
        pool = [found.name for found in _removable(held, kind) if found.kind != "controller"]
        controller = any(found.kind == "controller" for found in _removable(held, None))
        hint = ("; the controller is removed by kind: exp.remove(kind='controller')"
                if controller and kind is None else "")
        what = "no controller is declared" if target is None else \
            f"nothing is declared under {target!r}{f' as a {kind}' if kind else ''}"
        raise ArgumentRefused([Refusal(
            "remove.unknown", RefusalClass.USER, f"{what}, so nothing is removed{hint}",
            "declare one first, exp.set_controller(...), or drop the call" if target is None else
            f"name what is declared, exp.remove('<name>'). {did_you_mean(target, pool)}",
            why="exp.remove takes out a declaration by its name, its id (a law's, an observation's) or its path")])
    if len(matches) > 1:
        raise ArgumentRefused([Refusal(
            "remove.ambiguous", RefusalClass.USER,
            f"{target!r} is declared as a {' and as a '.join(m.kind for m in matches)}, and a name removes one kind at "
            f"a time", f"say which with kind=, exp.remove('<path>', kind='value'): {' or '.join(_how_to_remove(m) for m in matches)}")])
    (found,) = matches
    holding = referrers(held, found)
    known = known_as(held[found.places[0]])
    assert known is not None
    noun = known.holder or known.noun
    if holding:
        lines = [f"{called(held[index])} ({place_of(held[index])})" for index in holding]
        calls = [_how_to_remove(Found(*cast(tuple[str, str], _kind_and_name(held[index])), [index]))
                 for index in holding]
        raise ArgumentRefused([Refusal(
            "remove.named", RefusalClass.USER,
            f"{noun} cannot be removed while {len(holding)} declaration{'s' if len(holding) > 1 else ''} still "
            f"name{'' if len(holding) > 1 else 's'} it: {'; '.join(lines)}",
            f"remove or change each first, at the cell or line named, with exp.remove(...) ({'; '.join(calls)}), then "
            f"remove it",
            why="a removal takes nothing else with it: what names it would be left with no one to resolve it")])
    note = (f"exp.remove(...) removes {noun}" + (f" ({len(found.places)} declarations)" if len(found.places) > 1
                                                      else "") + "; nothing else names it")
    warn_at_script(note, UserWarning)  # before the change: a warning raised as an error leaves the experiment as it was
    for index in reversed(found.places):
        del held[index]
    removed.append(note)
    return noun, note


def _kind_and_name(declaration: Declaration) -> tuple[str, str] | None:
    """The kind (as ``kind=`` says it) and name of a declaration ``exp.remove`` can remove, or ``None``."""
    known = known_as(declaration)
    if known is None or known.key is None or known.key[0] not in _WORDS:
        return None
    return _WORDS[known.key[0]], known.key[1]


__all__ = ["Found", "KEYS", "remove"]
