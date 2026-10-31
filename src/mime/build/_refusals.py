from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from importlib.resources import files
from types import MappingProxyType
from typing import TYPE_CHECKING, override

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


#: Where a code's page is: the documentation's address, then the page ``mime docs build`` writes for it,
#: ``pages/errors/<code>``.
ERROR_PAGES = "https://microrobotica.org/mime/pages/errors/"
#: Where a MIME defect is reported.
ISSUES = "https://github.com/Microrobotics-Simulation-Framework/MIME/issues"


def page_of(code: str) -> str:
    """The address of ``code``'s generated page."""
    return f"{ERROR_PAGES}{code}"


class Explained(str):
    """A refusal's message with its why beside it, for the ``(code, message, action)`` tuples that problems travel in
    to the build: a ``str`` wherever it goes, and the refusal it becomes takes its ``why`` from it."""

    why: str

    def __new__(cls, message: str, why: str) -> Explained:
        held = super().__new__(cls, message)
        held.why = why
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

    def __post_init__(self) -> None:
        problem = catalogue_problem(self.code, self.category)
        if problem is not None:
            raise UncataloguedRefusal(problem)
        if self.why is None and isinstance(self.message, Explained):
            object.__setattr__(self, "why", self.message.why)
        for name in ("message", "action", "why"):
            held = getattr(self, name)
            if held is not None:
                object.__setattr__(self, name, plain_possessive(held))

    @property
    def uncatalogued(self) -> bool:
        """A registered namespace's own code, which MIME's table does not hold."""
        return namespace_of(self.code) != MIME_NAMESPACE

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

    def text(self) -> str:
        count = len(self.refusals)
        head = self.headline if count == 1 else f"{self.headline} for {_counted(count)}"
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
        super().__init__(refusals)


class UnrecognisedError(MimeError):
    """An error from below that MIME does not recognise, as it reached a public call: ``original`` is the error. The
    script's own frames are shown with it, and it is never called MIME's defect."""

    headline = "an error MIME does not recognise"


class InternalError(MimeError):
    """An error MIME raised with no code, or a check on MIME's own construction: a MIME defect, reported with the
    file its full chain is written to."""

    headline = "MIME failed"
