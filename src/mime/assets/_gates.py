"""The download gates: the licence a download needs accepted, and the disk space it needs. Both live here, in the library,
so that the command line, a script and a notebook meet the same two questions, and nothing downloads round them.

**The licence gate.** An asset under a licence with conditions beyond credit (non-commercial, share-alike, no
redistribution, a custom licence: whatever ``mime.terms`` does not judge open) is downloaded only once its conditions
are accepted. Conditions are short lines (a custom licence's ``conditions``, in its manifest; the Creative Commons
elements of a conditional licence), then the licence's title and link, and the record's notice. Where a person can answer
(a terminal, a notebook) the gate asks ``Do you accept these conditions? [y/N]``, defaulting to no, and remembers a
yes in the user's config directory (:mod:`mime.assets._acceptances`) so that it asks once per licence and version; where
nobody can, it refuses, naming the licence and ``--accept-licence <id>``. **No generic switch accepts a licence**: not
``yes``, not ``force``; only ``accept_licences``, which names the ids it accepts and nothing else, and which accepts for
the call without being remembered. An open licence (CC0, CC-BY, MIT, ...) asks nothing. ``NOASSERTION`` (nobody stated
the terms) is gated: its panel says the terms are not stated and that checking them is up to the person.

**When it asks.** Before a download, and also at the **first use** of a cached file: wherever its bytes are read (a bake,
the overlay, a pull's unpack, a fetch that finds it cached), a licence with conditions that has no recorded acceptance
is asked about once. A cache filled before the gates existed is covered. Acceptances are kept by the licence's id and
version (its title), so a licence read from a manifest and the same one read from a saved lock are one acceptance.
The lines of conditions are looked up by the licence's id (:func:`stated_licence`): in the record in hand, else in the
manifests installed packages ship, else in those a bake has cached.

**The disk gate.** Before a download MIME adds up the bytes it needs (the files not in the cache, and for a bake an
allowance for what the bake writes), reads the free space of the drive that holds the cache (``shutil.disk_usage`` on
the cache's own folder, so it follows ``MIME_ASSETS_DIR`` and works on every platform), says both in one line where
there is a terminal, and then **asks** when the download is over :data:`CONFIRM_ABOVE_BYTES` or over
:data:`CONFIRM_ABOVE_FRACTION` of the free space (``yes`` answers it), and **stops** when it would leave under
:data:`STOP_BELOW_BYTES` free or under :data:`STOP_BELOW_FRACTION` of the drive (``force`` overrides the stop, and
nothing else). Without a terminal the question refuses unless ``yes`` is given, and the stop refuses unless ``force`` is.

**Who asks.** The library's own questions are plain text on standard output and ``input()``, where there is a terminal
or a notebook (:func:`can_ask`). The command line puts its own asker in place (:func:`asking`) that draws the licence in
a panel. A batch (:func:`approved`) asks once for all its files, and the resolver then asks nothing more of them; a file
resolved alone is gated by the resolver itself.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from collections.abc import Callable, Collection, Generator, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, TextIO, cast

from mime._notebook import in_notebook
from mime._said import QUESTION_CONFIRM, Said
from mime._terminal import for_terminal, plain
from mime.assets._acceptances import Licence, accepted, remember
from mime.assets._cache import cache_dir
from mime.assets._reference import AssetRef
from mime.build import BuildRefused, Refusal, RefusalClass
from mime.terms import (
    CONDITIONAL,
    NO_ASSERTION,
    NON_COMMERCIAL,
    NO_DERIVATIVES,
    OPEN,
    SHARE_ALIKE,
    CustomLicence,
    ids_of,
    licence_verdict,
)

#: A download over this many bytes, or over :data:`CONFIRM_ABOVE_FRACTION` of the free space, is asked about first. A
#: choice: about what a session on a laptop's connection should not start without being told.
CONFIRM_ABOVE_BYTES = 1_000_000_000
CONFIRM_ABOVE_FRACTION = 0.10
#: A download that would leave under this many bytes free, or under this fraction of the drive, is stopped. A choice:
#: a drive that full is one the system, the swap and the next download are already short of.
STOP_BELOW_BYTES = 2_000_000_000
STOP_BELOW_FRACTION = 0.05
#: What baking a set of meshes writes, as a share of the meshes it reads (:func:`mime.viz.bake`): the AR4's set (18
#: meshes, 49.7 MB) wrote 0.019 of its input in all, and its smallest meshes the most for their size, 0.28 at worst
#: (a mesh the budget leaves whole is written whole, and a layer and its record have a size of their own). A bound
#: over what was measured, not a model.
BAKE_OUTPUT_FACTOR = 0.30

#: What a licence's condition is said as, for each condition ``mime.terms`` names.
_CONDITION_WORDS = {NON_COMMERCIAL: "non-commercial use only",
                    SHARE_ALIKE: "anything made from it must be shared under the same licence",
                    NO_DERIVATIVES: "modified copies may not be shared"}
_UNLISTED_CONDITION = ("its conditions beyond credit are not ones MIME lists: read the licence at its link before "
                       "accepting")
_UNDESCRIBED_CONDITION = ("the record does not describe this licence: read the licence at its link, if it has one, "
                          "before accepting")
_NOT_REDISTRIBUTABLE = "it may not be redistributed: MIME downloads the file for you, from its owner, into your cache only"
_SPDX_PAGE = "https://spdx.org/licenses/{}.html"


#: What a licence nobody stated is said as: the terms are not stated, and checking them is the person's.
_NOT_STATED_TITLE = "terms not stated"
_NOT_STATED = ("the terms of use are not stated: you are responsible for checking them before you use it",)
#: The folder of the cache where a bake keeps its indexes (``mime.viz.visuals_dir``); each holds the licences of the
#: manifest it baked, so that a later use finds their condition lines by id.
_BAKED_FOLDER = "visuals"
LICENCES_KEY = "licences"


def _installed_licences() -> Iterator[CustomLicence]:
    """The custom licences of the manifests installed packages ship (and MIME's own)."""
    from mime.assets._sources import shared_manifests

    try:
        manifests = shared_manifests()
    except BuildRefused:  # a broken source: the lines are not found there, and the licence is asked about without them
        return
    for _, manifest in manifests:
        for ref in manifest.assets:
            yield from ref.licences


def _cached_licences() -> Iterator[CustomLicence]:
    """The custom licences of the manifests a bake cached: the ``licences`` its index holds."""
    folder = cache_dir() / _BAKED_FOLDER
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else ():
        try:
            held: object = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        listed = cast("dict[str, object]", held).get(LICENCES_KEY) if isinstance(held, dict) else None
        if not isinstance(listed, dict):
            continue
        for identifier, entry in cast("dict[object, object]", listed).items():
            row = cast("dict[str, object]", entry) if isinstance(entry, dict) else {}
            lines = row.get("conditions")
            if isinstance(identifier, str) and isinstance(row.get("title"), str) and isinstance(row.get("url"), str) \
                    and isinstance(lines, list):
                yield CustomLicence(identifier, str(row["title"]), str(row["url"]), "not_permitted",
                                    tuple(str(each) for each in cast("list[object]", lines)))


def stated_licence(identifier: str) -> CustomLicence | None:
    """The custom licence ``identifier`` as a manifest states it, with condition lines: from the manifests installed
    packages ship, else from those a bake has cached (:data:`LICENCES_KEY` in its index). ``None`` where none gives it
    lines. This is how a licence read from a saved lock, whose form has no key for lines, gets its lines: by its id."""
    for found in (*_installed_licences(), *_cached_licences()):
        if found.id == identifier and found.conditions:
            return found
    return None


def licences_named(expression: str, custom: Mapping[str, CustomLicence] | None = None) -> tuple[Licence, ...]:
    """The licences of ``expression`` that MIME does not judge open, each with its title, link and condition lines:
    ``custom`` gives a custom licence's title and link (and lines, where it states them), and any lines it lacks are
    looked up by the licence's id (:func:`stated_licence`). ``NOASSERTION`` is one of them. None for a licence that is
    open as a whole (``MIT OR CC-BY-NC-4.0`` is the freer of the two)."""
    held = custom or {}
    if licence_verdict(expression, held).tier == "open":
        return ()
    found: list[Licence] = []
    for each in ids_of(expression):
        if each in OPEN:
            continue
        if each == NO_ASSERTION:
            found.append(Licence(each, _NOT_STATED_TITLE, "", _NOT_STATED))
            continue
        own = held.get(each)
        elsewhere = None if own is not None and own.conditions else stated_licence(each)
        if own is not None:
            lines = own.conditions or (elsewhere.conditions if elsewhere is not None else ())
            stated = lines or (_UNDESCRIBED_CONDITION,)
            if own.redistribution == "not_permitted" and not lines:
                stated = (_NOT_REDISTRIBUTABLE, *stated)
            found.append(Licence(each, own.title, own.url, stated))
        elif each in CONDITIONAL:
            found.append(Licence(each, each, _SPDX_PAGE.format(each),
                                 tuple(_CONDITION_WORDS[word] for word in CONDITIONAL[each])))
        elif each.startswith("LicenseRef-"):
            found.append(Licence(each, elsewhere.title if elsewhere is not None else each,
                                 elsewhere.url if elsewhere is not None else "",
                                 elsewhere.conditions if elsewhere is not None else (_UNDESCRIBED_CONDITION,)))
        else:
            found.append(Licence(each, each, _SPDX_PAGE.format(each), (_UNLISTED_CONDITION,)))
    return tuple(found)


def licences_read(expression: str) -> tuple[Licence, ...]:
    """:func:`licences_named` for a licence expression read from a record that is not an :class:`AssetRef` (a baked mesh's
    own), failing closed: an expression MIME cannot read is one licence, undescribed, to be accepted."""
    try:
        return licences_named(expression)
    except ValueError:
        return (Licence(expression, expression, "", (_UNDESCRIBED_CONDITION,)),)


def licences_of(ref: AssetRef) -> tuple[Licence, ...]:
    """The licences ``ref`` cannot be used without accepting (:func:`licences_named` of its licence and its custom
    licences): none for an open licence."""
    return licences_named(ref.licence, ref.custom)


def bytes_text(size: float) -> str:
    """A size as the gates say it: decimal, to a tenth below a hundred (``1.2 GB``, ``49.7 MB``, ``250 GB``, ``12.3 kB``,
    ``800 bytes``)."""
    for unit, scale in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if size >= scale:
            value = size / scale
            return f"{value:.0f} {unit}" if value >= 100 else f"{value:.1f} {unit}"
    return f"{int(size)} bytes"


def free_space(folder: Path) -> tuple[int, int]:
    """The free and the total bytes of the drive that holds ``folder``, or the nearest folder above it that exists."""
    probe = folder
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    return int(usage.free), int(usage.total)


@dataclass(frozen=True)
class Plan:
    """What a download needs and what its drive has: ``files`` and ``download`` bytes to fetch, ``extra`` bytes more for
    what a bake writes, into ``folder``, whose drive has ``free`` of ``total`` bytes. With ``bounded`` the ``download``
    is the most a download of unstated size may take (its cap), not a size: the stop counts it, the question does not."""

    files: int
    download: int
    extra: int
    folder: Path
    free: int
    total: int
    bounded: bool = False

    @property
    def needed(self) -> int:
        return self.download + self.extra

    @property
    def left(self) -> int:
        return self.free - self.needed

    @property
    def asks(self) -> bool:
        """Whether the download is over the size MIME asks about: :data:`CONFIRM_ABOVE_BYTES` or
        :data:`CONFIRM_ABOVE_FRACTION` of the free space. Never for a bounded download, whose size nobody stated."""
        return not self.bounded and (self.needed > CONFIRM_ABOVE_BYTES or self.needed > CONFIRM_ABOVE_FRACTION * self.free)

    @property
    def stops(self) -> bool:
        """Whether the download would leave under :data:`STOP_BELOW_BYTES` free or under :data:`STOP_BELOW_FRACTION` of
        the drive."""
        return self.left < STOP_BELOW_BYTES or self.left < STOP_BELOW_FRACTION * self.total

    def line(self) -> str:
        more = f", and bake about {bytes_text(self.extra)} more," if self.extra else ""
        size = f"up to {bytes_text(self.download)}" if self.bounded else bytes_text(self.download)
        return (f"Download {size} ({self.files} file{'s' if self.files != 1 else ''}){more} into "
                f"{plain(str(self.folder))}: {bytes_text(self.free)} free of {bytes_text(self.total)}.")


class Asker(Protocol):
    """Whoever asks the gates' questions where a person can answer: the licence (``licences``, with the record's
    ``notice`` to show under it), a yes-or-no question (``confirm``), and the one line that says what the download
    takes (``line``). The library's own is plain text (:class:`PlainAsker`); the command line draws a panel."""

    def licences(self, licences: Sequence[Licence], notice: str | None) -> bool: ...

    def confirm(self, question: str) -> bool: ...

    def line(self, text: str) -> None: ...


class NoTerminal(Exception):
    """An asker found there is nobody to answer after all (a notebook kernel that takes no input, or standard input that
    ends before an answer): the gate refuses as it does without a terminal, naming the flag, and not as a person's no."""


#: What the licence prompt is headed with, and asks.
LICENCE_HEADING = "Licence conditions: your acceptance is needed"
LICENCE_QUESTION = "Do you accept these conditions?"


def licence_lines(licences: Sequence[Licence], notice: str | None) -> list[str]:
    """The licence prompt as plain lines: its heading, each licence's conditions, title and link, then the notice. Every
    word of a record reaches the terminal as plain characters (:func:`mime._terminal.plain`; the notice keeps its line
    breaks): a manifest's or a lock's text is a stranger's, and an escape sequence in it could erase the conditions this
    prompt exists to show."""
    lines = [LICENCE_HEADING]
    for each in licences:
        lines += [f"  - {plain(line)}" for line in each.conditions]
        lines.append(f"  {plain(each.title)} ({plain(each.id)})")
        if each.url:
            lines.append(f"  {plain(each.url)}")
    if notice:
        lines += ["", for_terminal(notice)]
    return lines


def _yes(answer: str) -> bool:
    return answer.strip().lower() in ("y", "yes")


class PlainAsker:
    """The library's asker: plain lines and ``input()``, as a script or a notebook meets them. The lines and the prompt
    go to the terminal the person answers at: standard output where that is a terminal, else standard error (a pipe on
    standard output would swallow the question); in a notebook, to the cell's output and ``input()``'s own box."""

    def licences(self, licences: Sequence[Licence], notice: str | None) -> bool:
        out = self._terminal()
        for line in licence_lines(licences, notice):
            print(line, file=out)
        return self._answer(out, f"{LICENCE_QUESTION} [y/N] ")

    def confirm(self, question: str) -> bool:
        return self._answer(self._terminal(), f"{question} [y/N] ")

    def line(self, text: str) -> None:
        """What the download takes, said only where standard output is a terminal: a pipe gets what it always got."""
        if _is_terminal(sys.stdout):
            print(text)

    @staticmethod
    def _terminal() -> TextIO:
        return next((each for each in (sys.stdout, sys.stderr) if _is_terminal(each)), sys.stdout)

    @staticmethod
    def _answer(out: TextIO, prompt: str) -> bool:
        try:
            if in_notebook() or not _is_terminal(out):
                return _yes(input(prompt))
            out.write(prompt)
            out.flush()
            return _yes(input())
        except EOFError as error:  # standard input ends at once: no one is there (a null device reads as a terminal on Windows)
            raise NoTerminal from error
        except RuntimeError as error:  # a notebook kernel that takes no input (ipykernel's StdinNotImplementedError)
            raise NoTerminal from error


def _is_terminal(stream: object) -> bool:
    try:
        return bool(stream.isatty())  # type: ignore[attr-defined]
    except (AttributeError, ValueError, OSError):
        return False


def can_ask() -> bool:
    """Whether a person can answer here: standard input is a terminal and the question can be shown on one (standard output
    or standard error), or this is a notebook (Jupyter, Colab) or IPython, where ``input()`` asks in the page."""
    if in_notebook():
        return True
    stdin = getattr(sys, "stdin", None)
    # And a screen to show the question on: with standard output and standard error both redirected a question would go
    # into a file unseen (and Windows reads its null device as a terminal, which a program run with it must not be asked).
    return (stdin is not None and bool(stdin.isatty())
            and (_is_terminal(sys.stdout) or _is_terminal(sys.stderr)))


_UNSET: object = object()
_asker: ContextVar[object] = ContextVar("mime_asset_asker", default=_UNSET)


@contextmanager
def asking(asker: Asker | None) -> Generator[None]:
    """Inside this block ``asker`` asks the gates' questions, in place of the library's own; ``None`` says nobody can be
    asked, whatever the terminal."""
    token = _asker.set(asker)
    try:
        yield
    finally:
        _asker.reset(token)


def current_asker() -> Asker | None:
    """The asker in force: the one :func:`asking` set, else the library's plain one where a person can answer
    (:func:`can_ask`), else none."""
    held = _asker.get()
    if held is _UNSET:
        return PlainAsker() if can_ask() else None
    return held if held is not None else None  # type: ignore[return-value]


def _refuse(code: str, category: RefusalClass, message: str, action: str) -> BuildRefused:
    return BuildRefused([Refusal(code, category, message, action)])


def _flags(licences: Sequence[Licence]) -> str:
    return " ".join(f"--accept-licence {each.id}" for each in licences)


def _licence_refusal(licences: Sequence[Licence], needs: str, why: str) -> BuildRefused:
    """The refusal of a download whose licences are not accepted and cannot be asked about."""
    said = "; ".join(f"{plain(each.title)} ({plain(each.id)}): {'; '.join(plain(line) for line in each.conditions)}"
                     for each in licences)
    ids = ", ".join(repr(each.id) for each in licences)
    return _refuse(
        "asset.licence-not-accepted", RefusalClass.USER,
        f"{needs} is under {'a licence' if len(licences) == 1 else 'licences'} with conditions beyond credit, which "
        f"need your acceptance, and {why}. {said}",
        f"read the conditions, then accept them by name: add `{_flags(licences)}` to the command (in Python, "
        f"accept_licences={{{ids}}}); no other flag accepts a licence")


@dataclass
class Approval:
    """What a gate decided: the ``plan`` of the download (``None`` when nothing needed downloading), the ``licences``
    asked about and accepted this time, and whether a licence prompt showed the record's notice (``notice_shown``)."""

    plan: Plan | None = None
    licences: tuple[Licence, ...] = ()
    notice_shown: bool = False
    #: The hashes of the files this approval cleared, which the resolver then downloads without asking again.
    cleared: frozenset[str] = field(default_factory=frozenset[str])


_cleared: ContextVar[frozenset[str]] = ContextVar("mime_asset_cleared", default=frozenset())
#: The ids of the licences accepted inside the batch in force (:func:`approved`): named in ``accept_licences`` or answered
#: yes. The resolver, which a batch calls once per file, asks nothing of them again.
_accepted_here: ContextVar[frozenset[str]] = ContextVar("mime_asset_accepted_here", default=frozenset())


def is_cleared(ref: AssetRef) -> bool:
    """Whether a batch (:func:`approved`) has already cleared ``ref``'s download in this context."""
    return ref.sha256 in _cleared.get()


def _unaccepted(licences: Sequence[Licence], accept_licences: Collection[str]) -> list[Licence]:
    """The licences, each once, that nothing has accepted: not named in ``accept_licences``, not accepted in the batch in
    force, not remembered (by id and version)."""
    named = {*accept_licences, *_accepted_here.get()}
    found: dict[str, Licence] = {}
    for each in licences:
        if each.id not in named and not accepted(each):
            found.setdefault(each.id, each)
    return list(found.values())


def _accept(needed: Sequence[Licence], names: str, asker: Asker | None, notice: str | None) -> bool:
    """Ask for ``needed`` to be accepted, or refuse; remember a yes. Whether the notice was shown with the question."""
    if asker is None:
        raise _licence_refusal(needed, names, "there is no terminal to ask at")
    try:
        said_yes = asker.licences(needed, notice)
    except NoTerminal:
        raise _licence_refusal(needed, names, "there is no terminal to ask at") from None
    if not said_yes:
        raise _refuse("asset.licence-declined", RefusalClass.USER,
                      f"the conditions of {', '.join(plain(each.title) for each in needed)} were not accepted, and "
                      f"nothing was downloaded or used",
                      f"read them again and answer y to accept, or accept them by name: `{_flags(needed)}`")
    remember(needed)
    return bool(notice)


def _what(refs: Sequence[AssetRef], needed: Sequence[Licence], fallback: str) -> str:
    affected = [ref for ref in refs if {each.id for each in licences_of(ref)} & {each.id for each in needed}]
    return (affected[0].name if len(affected) == 1 else f"{len(affected)} assets" if affected else fallback)


def require_accepted(refs: Iterable[AssetRef], *, accept_licences: Collection[str] = (),
                     notice: str | None = None) -> Approval:
    """The **first use** of files already in the cache: wherever their bytes are about to be read (the resolver finding
    them, a bake, a pull's unpack), a licence with conditions that has no recorded acceptance is asked about once, or
    refused where nobody can be asked (``asset.licence-not-accepted``, naming ``--accept-licence <id>``), or declined
    (``asset.licence-declined``). Nothing else is gated: no disk check, no size question. A cache filled before the
    gates existed is covered. What was accepted by name or by a yes holds for the rest of an enclosing batch."""
    held = list({ref.sha256: ref for ref in refs}.values())
    needed = _unaccepted([licence for ref in held for licence in licences_of(ref)], accept_licences)
    if not needed:
        return Approval()
    shown = _accept(needed, _what(held, needed, "the file"), current_asker(),
                    notice or next((ref.notice for ref in held if ref.notice), None))
    return Approval(None, tuple(needed), shown)


def require_licences(licences: Sequence[Licence], *, accept_licences: Collection[str] = (),
                     notice: str | None = None, what: str = "the file") -> Approval:
    """:func:`require_accepted` for licences read from somewhere other than a record in hand (a baked mesh's own record,
    for the overlay): ``what`` names the files in a refusal."""
    needed = _unaccepted(licences, accept_licences)
    if not needed:
        return Approval()
    return Approval(None, tuple(needed), _accept(needed, what, current_asker(), notice))


def approve(refs: Iterable[AssetRef], *, accept_licences: Collection[str] = (), yes: bool = False,
            force: bool = False, extra_bytes: int = 0, notice: str | None = None,
            confirm: Callable[[str], bool] | None = None, bundle: tuple[str, int] | None = None,
            downloading: bool = False, bounded: bool = False) -> Approval:
    """Decide whether the download of ``refs`` may go ahead, asking where a person can answer and refusing where not.

    The disk counts only the files that would be downloaded (:func:`~mime.assets._fetch.needs_download`: not in the
    cache, not offline, not copied in by hand; with ``downloading`` every one of ``refs``, for a caller that is about to
    store them whatever the cache holds, as the resolver is when it repairs a file); nothing to download is nothing to
    ask about the disk. The licences count every one of ``refs``, cached or not: the first use of a cached file is
    gated too (:func:`require_accepted`). In this order: the **stop** (refused unless ``force``), the **licences** not
    yet accepted (asked about, or accepted by name in ``accept_licences``, or refused), the one **line** saying what the
    download takes (to whoever can be asked), and the **question** when the download is over the size MIME asks about
    (answered by ``yes``, else asked; refused where nobody can be). ``confirm`` answers the size question in place of
    the asker's. ``extra_bytes`` is what a bake will write; ``notice`` the text shown under a licence prompt;
    ``bundle`` a download that is not an asset, by its name and size, which with ``bounded`` is only the most it may take
    (its cap): the stop counts it and the question does not.

    Raises ``asset.disk-low``, ``asset.licence-not-accepted``, ``asset.licence-declined``, ``asset.download-large``
    or ``asset.download-declined``."""
    from mime.assets._fetch import needs_download

    wanted = list({ref.sha256: ref for ref in refs}.values())
    absent = [ref for ref in wanted if downloading or needs_download(ref)]
    asker = current_asker()
    notice = notice or next((ref.notice for ref in wanted if ref.notice), None)
    plan: Plan | None = None
    if absent or bundle is not None:
        folder = cache_dir()
        sizes = [ref.size for ref in absent] + ([bundle[1]] if bundle is not None else [])
        free, total = free_space(folder)
        plan = Plan(len(sizes), sum(sizes), extra_bytes if absent else 0, folder, free, total, bounded)
        if plan.stops and not force:
            raise _refuse(
                "asset.disk-low", RefusalClass.USER,
                f"the download needs {'up to ' if bounded else ''}{bytes_text(plan.needed)} and {plain(str(folder))} has "
                f"{bytes_text(free)} free of {bytes_text(total)}: it would leave {bytes_text(max(plan.left, 0))}, under "
                f"the {bytes_text(STOP_BELOW_BYTES)} or {STOP_BELOW_FRACTION:.0%} of the drive MIME keeps free",
                "free some space, or set MIME_ASSETS_DIR to a folder on a drive with room; `--force` (in Python, "
                "force=True) downloads anyway")
    needed = _unaccepted([licence for ref in wanted for licence in licences_of(ref)], accept_licences)
    shown = False
    if needed:
        shown = _accept(needed, _what(wanted, needed, "the download"), asker, notice)
    if plan is None:
        return Approval(None, tuple(needed), shown)
    if asker is not None:
        asker.line(plan.line())
    if plan.asks and not yes:
        question = f"Download {bytes_text(plan.needed)} now?"
        if confirm is not None:
            answered = confirm(Said(question, QUESTION_CONFIRM))
        elif asker is not None:
            try:
                answered = asker.confirm(question)
            except NoTerminal:
                answered = None
        else:
            answered = None
        if answered is None:
            raise _refuse(
                "asset.download-large", RefusalClass.USER,
                f"the download is {bytes_text(plan.needed)}, over the {bytes_text(CONFIRM_ABOVE_BYTES)} or "
                f"{CONFIRM_ABOVE_FRACTION:.0%} of the free space ({bytes_text(plan.free)}) MIME asks about, and this is "
                f"not a terminal to ask in",
                "add `--yes` to the command to download it (in Python, yes=True)")
        if not answered:
            raise _refuse("asset.download-declined", RefusalClass.USER,
                          f"the download of {bytes_text(plan.needed)} was not confirmed, and nothing was downloaded",
                          "answer y when asked, or add `--yes` to the command")
    return Approval(plan, tuple(needed), shown, frozenset(ref.sha256 for ref in absent))


@contextmanager
def approved(refs: Iterable[AssetRef], *, accept_licences: Collection[str] = (), yes: bool = False,
             force: bool = False, extra_bytes: int = 0, notice: str | None = None,
             confirm: Callable[[str], bool] | None = None) -> Generator[Approval]:
    """:func:`approve` for ``refs``, and inside the block the resolver downloads, and reads, them without asking again."""
    approval = approve(refs, accept_licences=accept_licences, yes=yes, force=force, extra_bytes=extra_bytes,
                       notice=notice, confirm=confirm)
    cleared = _cleared.set(_cleared.get() | approval.cleared)
    here = _accepted_here.set(_accepted_here.get() | {*accept_licences, *(each.id for each in approval.licences)})
    try:
        yield approval
    finally:
        _accepted_here.reset(here)
        _cleared.reset(cleared)


def bake_bytes(refs: Iterable[AssetRef]) -> int:
    """What baking the meshes ``refs`` writes, at most: :data:`BAKE_OUTPUT_FACTOR` of their size."""
    return math.ceil(BAKE_OUTPUT_FACTOR * sum(ref.size for ref in refs))


__all__ = ["BAKE_OUTPUT_FACTOR", "CONFIRM_ABOVE_BYTES", "CONFIRM_ABOVE_FRACTION", "LICENCE_HEADING",
           "LICENCE_QUESTION", "STOP_BELOW_BYTES", "STOP_BELOW_FRACTION", "Approval", "Asker", "Licence", "NoTerminal",
           "PlainAsker", "Plan", "approve", "approved", "asking", "bake_bytes", "bytes_text", "can_ask",
           "current_asker", "free_space", "is_cleared", "licence_lines", "licences_named", "licences_of", "licences_read",
           "require_accepted", "require_licences", "stated_licence"]
