from __future__ import annotations

import functools
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from importlib.resources import files
from types import FunctionType, MappingProxyType
from typing import TYPE_CHECKING, Any, cast, override

from mime.vocabulary import MIME_NAMESPACE, namespace_of

if TYPE_CHECKING:
    from mime.build._report import BuildReport

MIME_DEFECT = "MIME defect: report it with this build report."


class RefusalClass(Enum):
    """Who can act on a refusal. ``USER``: the script can change it. ``ENVIRONMENT``: something the build needs is
    missing. ``INTERNAL``: a check on MIME's own construction, which no supported script should reach: a MIME defect.
    ``UNRECOGNISED``: an error from below (JAX, MADDENING, another library, or the script's own code) that MIME does
    not recognise, which is never called MIME's defect."""

    USER = "user"
    ENVIRONMENT = "environment"
    INTERNAL = "internal"
    UNRECOGNISED = "unrecognised"


#: The file of MIME's refusal codes, each with the classes it is refused in, beside this module.
TABLE_FILE = "refusal_table.txt"


def read_table(text: str) -> dict[str, frozenset[RefusalClass]]:
    """The refusal table's rows, ``code  class [class]``; a line starting with ``#`` is a comment. A code on two rows,
    like a class that is none of ``RefusalClass``'s, raises ``ValueError``."""
    table: dict[str, frozenset[RefusalClass]] = {}
    first: dict[str, int] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        if line.strip() and not line.lstrip().startswith("#"):
            code, *classes = line.split()
            if code in first:
                raise ValueError(f"line {number}: {code} has a row already, at line {first[code]}; keep one row")
            first[code] = number
            table[code] = frozenset(RefusalClass(name) for name in classes)
    return table


#: Every code MIME constructs, with the classes it is refused in.
REFUSAL_TABLE: Mapping[str, frozenset[RefusalClass]] = MappingProxyType(
    read_table(files("mime.build").joinpath(TABLE_FILE).read_text(encoding="utf-8")))


class UncataloguedRefusal(ValueError):
    """A refusal constructed with a code of MIME's that its refusal table does not hold, or in a class the table does
    not give it: a MIME defect, raised where it is constructed."""


def catalogue_problem(code: str, category: RefusalClass) -> str | None:
    """Why ``code`` cannot be refused in ``category``, or ``None``. A code whose first word is a registered namespace
    other than MIME's (``labx.too-far``) is that namespace's own and is not catalogued here."""
    if namespace_of(code) != MIME_NAMESPACE:
        return None
    classes = REFUSAL_TABLE.get(code)
    if classes is None:
        return (f"refusal code {code!r} is not in MIME's refusal table (mime/build/{TABLE_FILE}): a MIME code is "
                f"entered there, and a lab's is prefixed with its own namespace")
    if category not in classes:
        given = " and ".join(sorted(c.value for c in classes))
        return f"refusal code {code!r} is refused as {given} in MIME's refusal table, not as {category.value}"
    return None


#: A quoted name followed by a possessive, which a message built from a quoted name reads as ``'umr''s``.
_QUOTED_POSSESSIVE = re.compile(r"'([^'\s]+)''s\b")


def plain_possessive(text: str) -> str:
    """``text`` with each quoted name that takes a possessive written plain: ``body 'umr''s parts`` reads ``body
    umr's parts``."""
    return _QUOTED_POSSESSIVE.sub(r"\1's", text)


#: Where the documentation is: each MIME version's pages under its own version, ``<DOCS>/<version>/``.
DOCS = "https://microrobotica.org/mime"
#: Where a code's page is under a version's documentation: the page ``mime docs build`` writes for it.
ERROR_PAGES = "pages/errors"
#: Where a MIME defect is reported.
ISSUES = "https://github.com/Microrobotics-Simulation-Framework/MIME/issues"
_RELEASE = re.compile(r"\d+\.\d+\.\d+")


def docs_version() -> str:
    """The version whose documentation an error links to: this MIME's release (``0.3.0`` for ``0.3.0.dev0``)."""
    held = sys.modules.get("mime")
    found = _RELEASE.match(str(getattr(held, "__version__", "")))
    return found.group(0) if found else "latest"


def page_of(code: str) -> str:
    """The address of ``code``'s generated page, in this MIME version's documentation."""
    return f"{DOCS}/{docs_version()}/{ERROR_PAGES}/{code}"


class Explained(str):
    """A refusal's message with its why, and the line it was raised at, beside it, for the ``(code, message,
    action)`` tuples and problem strings that problems travel in to the build: a ``str`` wherever it goes, and the
    refusal it becomes takes its ``why`` and ``where`` from it."""

    why: str | None
    where: str | None

    def __new__(cls, message: str, why: str | None = None, *, where: str | None = None) -> Explained:
        held = super().__new__(cls, message)
        held.why = why
        held.where = where
        return held


@dataclass(frozen=True)
class Refusal:
    """One reason a call is refused, in one shape: what happened (``message``, in the experiment's own names), why
    (``why``, one plain sentence, where the message does not already say it), what to do (``action``, pasteable where
    it can be), where (``where``, the script's file and line, where MIME knows a line closer than the call), and the
    code, which names its page. A code of MIME's is checked against its refusal table when the refusal is
    constructed; a registered namespace's own is accepted as it is, and ``uncatalogued`` says so."""

    code: str
    category: RefusalClass
    message: str
    action: str
    why: str | None = None
    where: str | None = None
    #: The package an internal refusal is the defect of, where it is not MIME (an entry a lab registered).
    owner: str | None = None

    def __post_init__(self) -> None:
        problem = catalogue_problem(self.code, self.category)
        if problem is not None:
            raise UncataloguedRefusal(problem)
        if isinstance(self.message, Explained):
            if self.why is None:
                object.__setattr__(self, "why", self.message.why)
            if self.where is None:
                object.__setattr__(self, "where", self.message.where)
        for name in ("message", "action", "why"):
            held = getattr(self, name)
            if held is not None:
                object.__setattr__(self, name, plain_possessive(held))

    @property
    def uncatalogued(self) -> bool:
        """A registered namespace's own code, which MIME's table does not hold."""
        return namespace_of(self.code) != MIME_NAMESPACE

    @property
    def mimes_defect(self) -> bool:
        """An internal refusal of MIME's own: its code is MIME's and no other package is named as its owner."""
        return self.category is RefusalClass.INTERNAL and not self.uncatalogued and self.owner in (None, "mime")

    @property
    def page(self) -> str:
        """The address of the code's page: MIME's own codes have one."""
        return page_of(self.code) if not self.uncatalogued else f"the {namespace_of(self.code)} namespace's own code"

    def lines(self) -> list[str]:
        """The refusal in its shape, one line per part it has."""
        held = [f"[{self.category.value}] {self.code}: {self.message}"]
        if self.why:
            held.append(f"    why: {self.why}")
        held.append(f"    do: {self.action}")
        if self.where:
            held.append(f"    at: {self.where}")
        held.append(f"    see: {self.page}")
        return held

    @override
    def __str__(self) -> str:
        return "\n".join(self.lines())


def _counted(count: int) -> str:
    return f"{count} reason{'s' if count != 1 else ''}"


class MimeError(Exception):
    """An error MIME gives a script: one or more refusals, each with its code and class, what happened, why and what
    to do. A public call raises nothing else: an error from below is translated into one (``original`` holds it), and
    the full chain of what happened is kept on it (``chain``) when it leaves the call. ``headline`` says what was
    refused."""

    headline = "the call is refused"

    def __init__(self, refusals: Sequence[Refusal]) -> None:
        self.refusals = tuple(refusals)
        #: The error from below that this one translates, when it does.
        self.original: BaseException | None = None
        #: The full traceback, every frame and every chained error, as it was when the error left a public call.
        self.chain: str | None = None
        #: The file the full chain was written to, once it has been.
        self.traceback_file: str | None = None
        super().__init__(self.text())

    def codes(self) -> list[str]:
        return [r.code for r in self.refusals]

    @property
    def code(self) -> str:
        """The first refusal's code."""
        return self.refusals[0].code if self.refusals else "error.unrecognised"

    def text(self, headline: str | None = None) -> str:
        """The error in its shape: what was refused (``headline``, in place of the class's own: the CLI names its
        command), then each refusal."""
        count = len(self.refusals)
        held = headline or self.headline
        head = held if count == 1 else f"{held} for {_counted(count)}"
        return f"{head}:\n" + "\n".join(str(r) for r in self.refusals)

    @override
    def __str__(self) -> str:
        return self.text()


class BuildRefused(MimeError):
    """Every reason the build was refused, at once. ``report`` holds what was planned before the refusal."""

    headline = "the build is refused"

    def __init__(self, refusals: Sequence[Refusal], report: BuildReport | None = None) -> None:
        self.report = report
        super().__init__(refusals)


class ArgumentRefused(MimeError, ValueError):
    """A call's argument is a value the call does not take."""


class ArgumentTypeRefused(MimeError, TypeError):
    """A call's argument is of a type the call does not take."""


class RunFailed(MimeError, RuntimeError):
    """A run could not go on, or gave a state that is not finite numbers. ``result`` holds what it gave, where it
    gave anything."""

    headline = "the run failed"

    def __init__(self, refusals: Sequence[Refusal], result: object | None = None) -> None:
        self.result = result
        #: What a run that writes its outputs wrote before it failed (``run_to``'s ``RunWritten``), where it wrote any.
        self.written: object | None = None
        super().__init__(refusals)


class UnrecognisedError(MimeError):
    """An error from below that MIME does not recognise, as it reached a public call: ``original`` is the error. The
    script's own frames are shown with it, and it is never called MIME's defect."""

    headline = "an error MIME does not recognise"


class InternalError(MimeError):
    """An error MIME raised with no code, or a check on MIME's own construction: a MIME defect, reported with the
    file its full chain is written to."""

    headline = "MIME failed"


def entry_defect(cls: type, problem: str, *, label: str | None = None, context: str = "",
                 where: str | None = None, owner: str | None = None) -> Refusal:
    """The refusal of an entry's defect (``entry.defect``): the entry ``cls`` (shown as ``label``) did what its
    contract forbids, ``context`` saying where. It is the defect of the package that defines ``cls``, which is named
    as its owner (or ``owner``, the package whose code was found at fault), and ``where`` is the entry's own line that
    raised (or the problem's, when it is :class:`Explained` with one), where there is one."""
    name = label or cast(Any, cls).label()
    package = owner or cls.__module__.split(".", 1)[0]
    where = where or getattr(problem, "where", None)
    return Refusal("entry.defect", RefusalClass.INTERNAL, f"{name} ({cls.__module__}){context}: {problem}",
                   f"a defect in {name}, defined in {cls.__module__}: report it to that package. If that package is "
                   f"MIME: {MIME_DEFECT}", where=where, owner=package)


#: The attributes a builtin exception keeps outside its ``args`` and ``__dict__``, copied to an error that stands for
#: it.
_KEPT_ATTRIBUTES = ("name", "obj", "path", "name_from", "value", "code", "errno", "strerror", "filename", "filename2",
                    "characters_written")


@functools.cache
def _standing_for(mime: type[MimeError], below: type[BaseException]) -> type[MimeError]:
    """``mime``'s class that is also ``below``'s, so that a handler of either catches it."""
    return cast(type[MimeError], type(mime.__name__, (mime, below), {
        "__module__": mime.__module__, "__qualname__": mime.__qualname__, "__reduce__": _reduce,
        "_mime_class": mime, "_below_class": below}))


def _made(mime: type[MimeError], below: type[BaseException], args: tuple[Any, ...],
          state: Mapping[str, Any]) -> MimeError:
    cls = _standing_for(mime, below)
    try:
        held = cast(MimeError, cast(Any, below).__new__(cls, *args))
    except TypeError:  # a builtin whose own __new__ is not the one the class inherits (MemoryError's free list)
        held = BaseException.__new__(cls, *args)
    if not isinstance(below.__init__, FunctionType):  # a builtin's own: it sets what it keeps from its arguments
        below.__init__(held, *args)
    held.__dict__.update(state)
    return held


def _reduce(self: MimeError) -> tuple[Any, ...]:
    mime, below = cast(Any, type(self))._mime_class, cast(Any, type(self))._below_class
    import pickle

    try:
        pickle.dumps(below)
    except Exception:  # noqa: BLE001 - a class that cannot be pickled by name: the error crosses as MIME's alone
        return _rebuilt_plain, (mime, dict(self.__dict__))
    return _made, (mime, below, self.args, dict(self.__dict__))


def _rebuilt_plain(mime: type[MimeError], state: Mapping[str, Any]) -> MimeError:
    held = cast(MimeError, cast(Any, mime).__new__(mime))
    held.__dict__.update(state)
    Exception.__init__(held, held.text())
    return held


def standing_for(coded: MimeError, original: BaseException) -> MimeError | None:
    """``coded``, which translates ``original``, as an instance of ``original``'s class too, so that a handler written
    for the original catches it: ``coded`` itself when it already is one, else a copy of it whose class is both,
    holding the original's arguments and attributes; ``None`` when ``original``'s class cannot be made a base of one
    (its layout, or a ``__new__`` that does not take its own arguments), and the original is raised unchanged."""
    below = type(original)
    if isinstance(coded, below):
        return coded
    try:
        held = _made(type(coded), below, original.args, {})
    except Exception:  # noqa: BLE001 - a class that cannot be a base: the caller raises the original unchanged
        return None
    for name in _KEPT_ATTRIBUTES:
        if name not in original.__dict__ and hasattr(original, name):
            try:
                setattr(held, name, getattr(original, name))
            except (AttributeError, TypeError):
                pass
    held.__dict__.update({key: value for key, value in original.__dict__.items() if key != "__notes__"})
    held.__dict__.update(coded.__dict__)
    return held
