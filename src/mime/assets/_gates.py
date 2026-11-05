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
the call without being remembered. An open licence (CC0, CC-BY, MIT, ...) asks nothing, and ``NOASSERTION`` (nobody
stated a licence) states no condition to accept.

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

import math
import shutil
import sys
from collections.abc import Callable, Collection, Generator, Iterable, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, TextIO

from mime._notebook import in_notebook
from mime._said import QUESTION_CONFIRM, Said
from mime.assets._acceptances import Licence, accepted, remember
from mime.assets._cache import cache_dir
from mime.assets._reference import AssetRef
from mime.build import BuildRefused, Refusal, RefusalClass
from mime.terms import CONDITIONAL, NO_ASSERTION, NON_COMMERCIAL, NO_DERIVATIVES, OPEN, SHARE_ALIKE, ids_of

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


def licences_of(ref: AssetRef) -> tuple[Licence, ...]:
    """The licences ``ref`` cannot be downloaded without accepting: the ids of its licence that MIME does not judge open,
    unless its verdict as a whole is open (``MIT OR CC-BY-NC-4.0`` is the freer of the two). None for an open licence
    and for ``NOASSERTION``, which states nothing to accept."""
    if ref.verdict.tier == "open":
        return ()
    found: list[Licence] = []
    for each in ids_of(ref.licence):
        if each in OPEN or each == NO_ASSERTION:
            continue
        custom = ref.custom.get(each)
        if custom is not None:
            stated = custom.conditions or (_UNDESCRIBED_CONDITION,)
            if custom.redistribution == "not_permitted" and not custom.conditions:
                stated = (_NOT_REDISTRIBUTABLE, *stated)
            found.append(Licence(each, custom.title, custom.url, stated))
        elif each in CONDITIONAL:
            found.append(Licence(each, each, _SPDX_PAGE.format(each),
                                 tuple(_CONDITION_WORDS[word] for word in CONDITIONAL[each])))
        elif each.startswith("LicenseRef-"):
            found.append(Licence(each, each, "", (_UNDESCRIBED_CONDITION,)))
        else:
            found.append(Licence(each, each, _SPDX_PAGE.format(each), (_UNLISTED_CONDITION,)))
    return tuple(found)


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
    what a bake writes, into ``folder``, whose drive has ``free`` of ``total`` bytes."""

    files: int
    download: int
    extra: int
    folder: Path
    free: int
    total: int

    @property
    def needed(self) -> int:
        return self.download + self.extra

    @property
    def left(self) -> int:
        return self.free - self.needed

    @property
    def asks(self) -> bool:
        """Whether the download is over the size MIME asks about: :data:`CONFIRM_ABOVE_BYTES` or
        :data:`CONFIRM_ABOVE_FRACTION` of the free space."""
        return self.needed > CONFIRM_ABOVE_BYTES or self.needed > CONFIRM_ABOVE_FRACTION * self.free

    @property
    def stops(self) -> bool:
        """Whether the download would leave under :data:`STOP_BELOW_BYTES` free or under :data:`STOP_BELOW_FRACTION` of
        the drive."""
        return self.left < STOP_BELOW_BYTES or self.left < STOP_BELOW_FRACTION * self.total

    def line(self) -> str:
        more = f", and bake about {bytes_text(self.extra)} more," if self.extra else ""
        return (f"Download {bytes_text(self.download)} ({self.files} file{'s' if self.files != 1 else ''}){more} into "
                f"{self.folder}: {bytes_text(self.free)} free of {bytes_text(self.total)}.")


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
    """The licence prompt as plain lines: its heading, each licence's conditions, title and link, then the notice."""
    lines = [LICENCE_HEADING]
    for each in licences:
        lines += [f"  - {line}" for line in each.conditions]
        lines.append(f"  {each.title} ({each.id})")
        if each.url:
            lines.append(f"  {each.url}")
    if notice:
        lines += ["", notice]
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
    said = "; ".join(f"{each.title} ({each.id}): {'; '.join(each.conditions)}" for each in licences)
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


def is_cleared(ref: AssetRef) -> bool:
    """Whether a batch (:func:`approved`) has already cleared ``ref``'s download in this context."""
    return ref.sha256 in _cleared.get()


def approve(refs: Iterable[AssetRef], *, accept_licences: Collection[str] = (), yes: bool = False,
            force: bool = False, extra_bytes: int = 0, notice: str | None = None,
            confirm: Callable[[str], bool] | None = None, bundle: tuple[str, int] | None = None) -> Approval:
    """Decide whether the download of ``refs`` may go ahead, asking where a person can answer and refusing where not.

    Only the files that would be downloaded count (:func:`~mime.assets._fetch.needs_download`: not in the cache, not
    offline, not copied in by hand); nothing to download is nothing to ask. In this order: the **stop** (refused
    unless ``force``), the **licences** not yet accepted (asked about, or accepted by name in ``accept_licences``, or
    refused), the one **line** saying what the download takes (to whoever can be asked), and the **question** when the
    download is over the size MIME asks about (answered by ``yes``, else asked; refused where nobody can be). ``confirm``
    answers the size question in place of the asker's. ``extra_bytes`` is what a bake will write; ``notice`` the text
    shown under a licence prompt; ``bundle`` a download that is not an asset, by its name and size.

    Raises ``asset.disk-low``, ``asset.licence-not-accepted``, ``asset.licence-declined``, ``asset.download-large``
    or ``asset.download-declined``."""
    from mime.assets._fetch import needs_download

    wanted = list(refs)
    absent = list({ref.sha256: ref for ref in wanted if needs_download(ref)}.values())
    if not absent and bundle is None:
        return Approval()
    asker = current_asker()
    notice = notice or next((ref.notice for ref in absent if ref.notice), None)
    folder = cache_dir()
    sizes = [ref.size for ref in absent] + ([bundle[1]] if bundle is not None else [])
    free, total = free_space(folder)
    plan = Plan(len(sizes), sum(sizes), extra_bytes if absent else 0, folder, free, total)
    if plan.stops and not force:
        raise _refuse(
            "asset.disk-low", RefusalClass.USER,
            f"the download needs {bytes_text(plan.needed)} and {folder} has {bytes_text(free)} free of "
            f"{bytes_text(total)}: it would leave {bytes_text(max(plan.left, 0))}, under the "
            f"{bytes_text(STOP_BELOW_BYTES)} or {STOP_BELOW_FRACTION:.0%} of the drive MIME keeps free",
            "free some space, or set MIME_ASSETS_DIR to a folder on a drive with room; `--force` (in Python, "
            "force=True) downloads anyway")
    shown = False
    needed = _unaccepted([licence for ref in absent for licence in licences_of(ref)], accept_licences)
    if needed:
        name = (absent[0].name if len(absent) == 1 else f"{len(absent)} assets" if absent else "the download")
        if asker is None:
            raise _licence_refusal(needed, name, "there is no terminal to ask at")
        try:
            said_yes = asker.licences(needed, notice)
        except NoTerminal:
            raise _licence_refusal(needed, name, "there is no terminal to ask at") from None
        if not said_yes:
            raise _refuse("asset.licence-declined", RefusalClass.USER,
                          f"the conditions of {', '.join(each.title for each in needed)} were not accepted, and "
                          f"nothing was downloaded",
                          f"read them again and answer y to accept, or accept them by name: `{_flags(needed)}`")
        remember(needed)
        shown = True
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
                f"{CONFIRM_ABOVE_FRACTION:.0%} of the free space ({bytes_text(free)}) MIME asks about, and this is "
                f"not a terminal to ask in",
                "add `--yes` to the command to download it (in Python, yes=True)")
        if not answered:
            raise _refuse("asset.download-declined", RefusalClass.USER,
                          f"the download of {bytes_text(plan.needed)} was not confirmed, and nothing was downloaded",
                          "answer y when asked, or add `--yes` to the command")
    return Approval(plan, tuple(needed), shown and bool(notice), frozenset(ref.sha256 for ref in absent))


def _unaccepted(licences: Sequence[Licence], accept_licences: Collection[str]) -> list[Licence]:
    """The licences, each once, that nothing has accepted: not named in ``accept_licences``, not remembered."""
    named = set(accept_licences)
    found: dict[str, Licence] = {}
    for each in licences:
        if each.id not in named and not accepted(each):
            found.setdefault(each.id, each)
    return list(found.values())


@contextmanager
def approved(refs: Iterable[AssetRef], *, accept_licences: Collection[str] = (), yes: bool = False,
             force: bool = False, extra_bytes: int = 0, notice: str | None = None,
             confirm: Callable[[str], bool] | None = None) -> Generator[Approval]:
    """:func:`approve` for ``refs``, and inside the block the resolver downloads them without asking again."""
    approval = approve(refs, accept_licences=accept_licences, yes=yes, force=force, extra_bytes=extra_bytes,
                       notice=notice, confirm=confirm)
    token = _cleared.set(_cleared.get() | approval.cleared)
    try:
        yield approval
    finally:
        _cleared.reset(token)


def bake_bytes(refs: Iterable[AssetRef]) -> int:
    """What baking the meshes ``refs`` writes, at most: :data:`BAKE_OUTPUT_FACTOR` of their size."""
    return math.ceil(BAKE_OUTPUT_FACTOR * sum(ref.size for ref in refs))


__all__ = ["BAKE_OUTPUT_FACTOR", "CONFIRM_ABOVE_BYTES", "CONFIRM_ABOVE_FRACTION", "LICENCE_HEADING",
           "LICENCE_QUESTION", "STOP_BELOW_BYTES", "STOP_BELOW_FRACTION", "Approval", "Asker", "Licence", "NoTerminal",
           "PlainAsker", "Plan", "approve", "approved", "asking", "bake_bytes", "bytes_text", "can_ask",
           "current_asker", "free_space", "is_cleared", "licence_lines", "licences_of"]
