from __future__ import annotations

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
    missing. ``INTERNAL``: a check on MIME's own construction, which no supported script should reach."""

    USER = "user"
    ENVIRONMENT = "environment"
    INTERNAL = "internal"


#: The file of MIME's refusal codes, each with the classes it is refused in, beside this module.
TABLE_FILE = "refusal_table.txt"


def read_table(text: str) -> dict[str, frozenset[RefusalClass]]:
    """The refusal table's rows, ``code  class [class]``; a line starting with ``#`` is a comment."""
    table: dict[str, frozenset[RefusalClass]] = {}
    for line in text.splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            code, *classes = line.split()
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


@dataclass(frozen=True)
class Refusal:
    """One reason a build is refused, named in the experiment's own terms, with what to do about it. A code of MIME's
    is checked against its refusal table when the refusal is constructed; a registered namespace's own is accepted
    as it is, and ``uncatalogued`` says so."""

    code: str
    category: RefusalClass
    message: str
    action: str

    def __post_init__(self) -> None:
        problem = catalogue_problem(self.code, self.category)
        if problem is not None:
            raise UncataloguedRefusal(problem)

    @property
    def uncatalogued(self) -> bool:
        """A registered namespace's own code, which MIME's table does not hold."""
        return namespace_of(self.code) != MIME_NAMESPACE

    @override
    def __str__(self) -> str:
        return f"[{self.category.value}] {self.code}: {self.message}\n    action: {self.action}"


class BuildRefused(Exception):
    """Every reason the build was refused, at once. ``report`` holds what was planned before the refusal."""

    def __init__(self, refusals: Sequence[Refusal], report: BuildReport | None = None) -> None:
        self.refusals = tuple(refusals)
        self.report = report
        count = len(self.refusals)
        super().__init__(f"the build is refused for {count} reason{'s' if count != 1 else ''}:\n"
                         + "\n".join(str(r) for r in self.refusals))

    def codes(self) -> list[str]:
        return [r.code for r in self.refusals]
