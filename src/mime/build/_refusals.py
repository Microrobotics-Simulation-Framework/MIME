from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mime.build._report import BuildReport

MIME_DEFECT = "MIME defect: report it with this build report."


class RefusalClass(Enum):
    """Who can act on a refusal. ``USER``: the script can change it. ``ENVIRONMENT``: something the build needs is
    missing. ``INTERNAL``: a check on MIME's own construction, which no supported script should reach."""

    USER = "user"
    ENVIRONMENT = "environment"
    INTERNAL = "internal"


@dataclass(frozen=True)
class Refusal:
    """One reason a build is refused, named in the experiment's own terms, with what to do about it."""

    code: str
    category: RefusalClass
    message: str
    action: str

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
