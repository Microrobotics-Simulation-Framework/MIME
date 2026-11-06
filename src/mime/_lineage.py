"""Which declarations a build, a result or a view came from, and whether the experiment has changed since.

An experiment changes under a build in two ways that a plain ``add_*`` does not: a declaration is *replaced* (a notebook's
cell run again with another number) or *removed* (``exp.remove``). A built experiment, the result of its run and the view
of that run each remember the declarations they came from (an :class:`Origin`), and when they are used or shown after a
replacement or a removal they say so, in words that name what changed: they would otherwise pass for the new physics.

An origin is stale when a change was recorded after it was taken **and** the declarations now differ from the ones it
holds, compared by content (``mime._content.same_content``): a declaration replaced and then put back is not stale. What
it does not call stale: a declaration only added (a new law, an observation) is no replacement of anything the build
held, so the build is still what it was built from.

The origin of a run folder ``run_to`` wrote is also kept here by the folder's path (:func:`origin_of_folder`), since
``mime.viz.display`` is given a path and not the build."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from mime._content import same_content

#: How many changes a sentence lists before it counts the rest.
LISTED = 3


@dataclass(frozen=True)
class Change:
    """One replacement or removal made in this session: what it was of (``the apparatus 'field'``) and the note
    raised for it."""

    how: Literal["replacement", "removal"]
    subject: str
    note: str


def _listing(changes: Sequence[Change]) -> str:
    """The changes in a sentence, each once however often it was made (a cell run again and again)."""
    every = list(dict.fromkeys(f"the {change.how} of {change.subject}" for change in changes))
    shown = every[:LISTED]
    text = " and ".join(shown) if len(shown) < 3 else f"{', '.join(shown[:-1])} and {shown[-1]}"
    return text if len(every) <= LISTED else f"{text} and {len(every) - LISTED} more"


@dataclass(frozen=True)
class Origin:
    """The declarations something was made from. ``changes`` is the experiment's own list of replacements and removals
    (live, so a later one is seen), ``seen`` how many it held at the time, ``state`` the declarations then, and
    ``current`` reads the declarations now."""

    changes: list[Change]
    seen: int
    state: tuple[object, ...]
    current: Callable[[], Sequence[object]]

    def since(self) -> tuple[Change, ...]:
        """The replacements and removals made since, or none where the declarations are the same as then."""
        later = tuple(self.changes[self.seen:])
        if not later:
            return ()
        now = self.current()
        if len(now) == len(self.state) and all(a is b or same_content(a, b) for a, b in zip(self.state, now)):
            return ()
        return later

    def stale(self, what: str, action: str) -> str | None:
        """``what`` (``this build``) predates the changes made since, with the ``action`` to take; ``None`` where there
        are none."""
        later = self.since()
        return f"{what} predates {_listing(later)}: {action}" if later else None


#: The origin of each run folder ``run_to`` wrote in this process, by its resolved path: the latest run into it.
_FOLDERS: dict[Path, Origin] = {}


def remember_folder(folder: Path, origin: Origin | None) -> None:
    key = folder.resolve()
    if origin is None:
        _FOLDERS.pop(key, None)
    else:
        _FOLDERS[key] = origin


def origin_of_folder(folder: str | Path) -> Origin | None:
    """The origin of the run ``run_to`` wrote into ``folder`` in this process, or ``None`` (a folder from another
    process, or one a command line wrote)."""
    try:
        return _FOLDERS.get(Path(folder).resolve())
    except OSError:
        return None
