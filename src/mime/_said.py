"""A line ``push`` and ``pull`` say, with what it is.

``mime push`` and ``mime pull`` hand their lines to a callable (``say``) and their questions to another (``ask``,
``typed``), and a person at a terminal reads them better when the plan, the publish step, a permission and a site edit are
told apart. A :class:`Said` is the ordinary ``str`` that was always passed, with a ``kind`` the terminal's display reads.
Printed, compared, joined or written it is its text, byte for byte: a pipe, a file and a test see nothing of the kind, and
only :mod:`mime.cli._display`, on a terminal, looks at it."""

from __future__ import annotations

from typing import Final, Self

#: A line of the plan the push shows before it writes: ``plan-head`` (what is pushed, and where), ``plan-up`` (what goes
#: up), ``plan-reference`` (what travels as a reference), ``plan-field`` (title, licences, credit), ``plan-line`` (a row of
#: the full plan) and ``plan-section`` (a heading of the full plan).
PLAN_HEAD: Final = "plan-head"
PLAN_UP: Final = "plan-up"
PLAN_REFERENCE: Final = "plan-reference"
PLAN_FIELD: Final = "plan-field"
PLAN_LINE: Final = "plan-line"
PLAN_SECTION: Final = "plan-section"
#: A warning of the plan (a line that starts ``! ``): drawn below the plan's block, not in it.
PLAN_WARNING: Final = "plan-warning"
#: The publish step's statement: its first line (``step-head``), the sentence that says it cannot be undone
#: (``step-irreversible``; on the sandbox, which can be cleaned, ``step-trial``, which says the same of production) and
#: what is stated (``step-line``).
STEP_HEAD: Final = "step-head"
STEP_IRREVERSIBLE: Final = "step-irreversible"
STEP_TRIAL: Final = "step-trial"
STEP_LINE: Final = "step-line"
#: A declared permission, as the public statement it becomes.
PERMISSION: Final = "permission"
#: The two values of a field edited on the site: what the site holds, and what ``sharing.yaml`` writes.
EDIT_SITE: Final = "edit-site"
EDIT_OURS: Final = "edit-ours"
#: A question: a declaration (``question-permission``), a site edit (``question-edit``), the record number typed
#: (``question-number``), or any other (``question-confirm``).
QUESTION_PERMISSION: Final = "question-permission"
QUESTION_EDIT: Final = "question-edit"
QUESTION_NUMBER: Final = "question-number"
QUESTION_CONFIRM: Final = "question-confirm"
#: What was done (``done``), that a step was not taken (``stopped``), and the page to look at (``link``).
DONE: Final = "done"
STOPPED: Final = "stopped"
LINK: Final = "link"


class Said(str):
    """``text``, with its ``kind``, and whether it ``last`` ends its block (the plan's block is drawn once its last line is
    said, not when the next line of something else is)."""

    kind: str
    last: bool

    def __new__(cls, text: str, kind: str, last: bool = False) -> Self:
        held = super().__new__(cls, text)
        held.kind = kind
        held.last = last
        return held


def kind_of(line: str) -> str | None:
    """What ``line`` was said as, or ``None`` for an ordinary line."""
    return line.kind if isinstance(line, Said) else None


def ends_block(line: str) -> bool:
    """Whether ``line`` was said as the last of its block."""
    return isinstance(line, Said) and line.last


def ending(lines: list[str]) -> list[str]:
    """``lines``, the last of them marked as ending its block."""
    if lines:
        lines[-1] = Said(lines[-1], kind_of(lines[-1]) or "", last=True)
    return lines
