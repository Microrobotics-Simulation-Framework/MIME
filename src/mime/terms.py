"""Licences and terms: under what a file may be shared, and so where its bytes may live (designs #198, #199).

**A licence is an SPDX licence expression** (``CC0-1.0``, ``MIT OR Apache-2.0``), checked by the packaging standards'
own parser (``packaging.licenses``, which the build already depends on) and held to its canonical spelling, so that
one licence is written one way and no document's hash depends on how it was typed. A licence of the author's own is a
:class:`CustomLicence`: a ``LicenseRef-`` id with its title and an https link. ``NOASSERTION`` (SPDX's word for a
licence nobody stated) is accepted alone, and reads as unknown. Free text is refused.

**A third-party file is declared with its terms** (:class:`Terms`) where the file is declared:
``asset("ar4_visual.usda", terms=Terms(licence=..., credit=..., role="presentation", origin=(...)))``. A file with no
terms is the author's own, under the experiment's licence (``exp.licence("CC-BY-4.0")``). **A permission** from the
rights holder (:class:`Permission`) is declared with ``exp.permission(...)``: its form is checked and it is bound to the
files' sha256, so a changed file inherits nothing; its truth is never vetted, and it is published under the uploader's
name.

**The verdict** of a file is in OpenMaterica's words for a source's redistribution status: ``permitted``,
``not_permitted`` or ``unknown`` (its fourth, ``values_only``, describes a record's values, not a file's bytes). Only a
licence on MIME's open or conditional lists, or a declared permission that grants redistribution, makes a file
``permitted``; every other licence is ``unknown`` until a person declares otherwise, and a :class:`CustomLicence` may
say ``not_permitted`` (a person may always say no; a yes needs a permission with its evidence). **The tier**, #198's,
says where the bytes live: ``open`` and ``conditional`` travel inside a bundle (the conditions said in its NOTICE and
on its card); ``referenced`` travels as a reference to its origin, whatever its size, and is never packed.

Nothing here imports the rest of MIME: :mod:`mime.scene` declares a file's terms with these classes, and the saved
form, ``pack`` and the card read them.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

#: SPDX's word for a licence nobody stated: accepted alone as a licence, and read as unknown.
NO_ASSERTION = "NOASSERTION"
CUSTOM_PREFIX = "LicenseRef-"
#: A custom licence's id after its prefix, as SPDX's ``idstring`` allows it: letters, digits, ``.`` and ``-``.
_CUSTOM_ID = re.compile(r"LicenseRef-[A-Za-z0-9.\-]+")
_TOKEN = re.compile(r"\(|\)|[^\s()]+")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")

#: MIME's open list: licences that permit redistribution with attribution at most. A choice, not a judgement of any
#: other licence: a licence not listed here or on the conditional list is ``unknown``, and travels as a reference until
#: a person declares a permission.
OPEN: frozenset[str] = frozenset({
    "CC0-1.0", "CC-BY-1.0", "CC-BY-2.0", "CC-BY-2.5", "CC-BY-3.0", "CC-BY-4.0", "MIT", "Apache-2.0", "BSD-2-Clause",
    "BSD-3-Clause"})

NON_COMMERCIAL = "non-commercial"
SHARE_ALIKE = "share-alike"
NO_DERIVATIVES = "no-derivatives"
#: The words each Creative Commons element stands for.
_ELEMENTS = {"NC": NON_COMMERCIAL, "SA": SHARE_ALIKE, "ND": NO_DERIVATIVES}
#: MIME's conditional list: Creative Commons licences that permit redistribution on conditions that need no
#: acceptance, each with its conditions, which a NOTICE and a card state.
CONDITIONAL: Mapping[str, tuple[str, ...]] = {
    f"CC-BY-{elements}-{version}": tuple(_ELEMENTS[part] for part in elements.split("-"))
    for elements in ("NC", "SA", "ND", "NC-SA", "NC-ND") for version in ("1.0", "2.0", "2.5", "3.0", "4.0")}
#: The licences whose text must travel with every copy, with the file's credit as its copyright line.
NOTICE_TEXT_REQUIRED = frozenset({"MIT", "BSD-2-Clause", "BSD-3-Clause"})

type Status = Literal["permitted", "not_permitted", "unknown"]
type Tier = Literal["open", "conditional", "referenced"]
type Role = Literal["results", "presentation"]
type EvidenceKind = Literal["public_statement", "written_permission"]
ROLES: tuple[Role, ...] = ("results", "presentation")
EVIDENCE_KINDS: tuple[EvidenceKind, ...] = ("public_statement", "written_permission")
#: A custom licence's declared redistribution status: a person may say no, or leave it unknown; a yes is a permission.
CUSTOM_STATUSES: tuple[Status, ...] = ("not_permitted", "unknown")


def canonical(text: str) -> str | None:
    """``text`` in the SPDX expression's canonical spelling, or ``None`` where it is not an expression the parser
    reads."""
    from packaging.licenses import InvalidLicenseExpression, canonicalize_license_expression

    try:
        return str(canonicalize_license_expression(text))
    except InvalidLicenseExpression:
        return None


def licence_problem(text: object) -> str | None:
    """Why ``text`` is not a licence as MIME stores one, or ``None``: an SPDX licence expression in its canonical
    spelling, or ``NOASSERTION`` alone."""
    if not isinstance(text, str) or not text:
        return "a licence is an SPDX licence expression, such as CC-BY-4.0 or MIT OR Apache-2.0"
    if text == NO_ASSERTION:
        return None
    if text != "".join(each for each in text if each.isprintable()):
        return "the licence has a control or format character"
    held = canonical(text)
    if held is None:
        return (f"{text!r} is not an SPDX licence expression (an id of the SPDX licence list, or LicenseRef- and an id "
                f"for a licence of your own, joined by AND, OR and WITH)")
    if held != text:
        return f"{text!r} is written {held!r} in SPDX's canonical spelling"
    return None


def ids_of(expression: str) -> tuple[str, ...]:
    """The licence ids an expression names (``LicenseRef-`` ids among them), each once, in order: neither an operator,
    a parenthesis nor an exception."""
    tokens = _TOKEN.findall(expression)
    found: list[str] = []
    for index, token in enumerate(tokens):
        if token in ("AND", "OR", "WITH", "(", ")") or (index and tokens[index - 1] == "WITH"):
            continue
        found.append(token.removesuffix("+"))
    return tuple(dict.fromkeys(found))


@dataclass(frozen=True)
class CustomLicence:
    """A licence of its own: a ``LicenseRef-`` id, its title and an https link to its text. ``redistribution`` is
    what a person says it allows: ``unknown`` (the default) or ``not_permitted``; a licence that permits is declared as
    a :class:`Permission`, with its evidence."""

    id: str
    title: str
    url: str
    redistribution: Status = "unknown"

    def problem(self) -> str | None:
        if not isinstance(cast(object, self.id), str) or not _CUSTOM_ID.fullmatch(self.id):
            return f"a custom licence's id is LicenseRef- and letters, digits, '.' and '-', not {self.id!r}"
        if not isinstance(cast(object, self.title), str) or not self.title.strip():
            return f"the custom licence {self.id} has no title"
        if not isinstance(cast(object, self.url), str) or not self.url.startswith("https://") or \
                any(each.isspace() for each in self.url):
            return f"the custom licence {self.id}'s link is an https:// address, not {self.url!r}"
        if self.redistribution not in CUSTOM_STATUSES:
            return (f"the custom licence {self.id}'s redistribution is one of {', '.join(CUSTOM_STATUSES)}: a licence "
                    f"that permits redistribution is declared as a Permission, with its evidence")
        return None


type LicenceGiven = str | CustomLicence


def expression_of(licence: LicenceGiven) -> str:
    return licence.id if isinstance(licence, CustomLicence) else licence


@dataclass(frozen=True)
class Terms:
    """A third-party file's terms: its licence (an SPDX expression, or a :class:`CustomLicence`), the credit its licence
    asks for, whether results depend on it (``results``) or it only draws the scene (``presentation``), and its origin,
    the locations its exact bytes can be fetched from, primary first (``https://`` or ``doi:``, pinned: a commit, not a
    branch). A file that may not be redistributed travels as a reference to its origin, so it needs one."""

    licence: LicenceGiven
    credit: str
    role: Role
    origin: tuple[str, ...] = ()


@dataclass(frozen=True)
class Scope:
    """What a permission allows: redistributing the files, modifying them, and commercial use."""

    redistribute: bool
    modify: bool
    commercial: bool

    def words(self) -> str:
        return ", ".join(("redistribute" if self.redistribute else "no redistribution",
                          "modify" if self.modify else "no modification",
                          "commercial use" if self.commercial else "non-commercial"))


@dataclass(frozen=True)
class Evidence:
    """A permission's evidence, strongest first: a ``public_statement`` by the rights holder, its ``pointer`` the https
    link to it; or a ``written_permission`` held privately (an email, a letter), its ``pointer`` saying what it is and
    who holds it. The document itself is never published: it holds personal data. Nothing written is not accepted."""

    kind: EvidenceKind
    pointer: str


@dataclass(frozen=True)
class Permission:
    """A declaration that the rights holder (``grantor``: a person or an organisation, with their role) permitted the
    ``grantee`` (the uploader, or their institution) to do what ``scope`` says with ``files``, each named by the sha256
    of its bytes (or by the :func:`mime.scene.asset` that declares it, pinned when the experiment is saved), on
    ``date`` (``YYYY-MM-DD``), shown by ``evidence``. MIME checks its form and binds it to the bytes; it never vets
    its truth, and publishing makes it public under the uploader's name."""

    grantor: str
    grantee: str
    files: tuple[object, ...]
    scope: Scope
    date: str
    evidence: Evidence


def permission_problems(permission: object, sha256s: Sequence[str]) -> list[str]:
    """Every way ``permission`` (its files already pinned as ``sha256s``) is incomplete or malformed, each naming the
    part: the form only, never the truth."""
    if not isinstance(permission, Permission):
        return [f"{permission!r} is not a Permission"]
    problems: list[str] = []
    for part in ("grantor", "grantee"):
        value = cast(object, getattr(permission, part))
        if not isinstance(value, str) or not value.strip():
            problems.append(f"the permission names no {part}")
    if not sha256s:
        problems.append("the permission names no file")
    problems += [f"the permission names the file {each!r}, which is not the sha256 of a file's bytes"
                 for each in sha256s if not _SHA256.fullmatch(each)]
    scope = cast(object, permission.scope)
    if not isinstance(scope, Scope) or not all(isinstance(cast(object, getattr(scope, name)), bool)
                                               for name in ("redistribute", "modify", "commercial")):
        problems.append("the permission's scope is a Scope(redistribute=..., modify=..., commercial=...) of booleans")
    elif not scope.redistribute:
        problems.append("the permission's scope does not include redistribution, so it permits nothing a bundle or a "
                        "record does")
    date = cast(object, permission.date)
    if not isinstance(date, str) or not _DATE.fullmatch(date) or not _valid_date(date):
        problems.append(f"the permission's date is a date written YYYY-MM-DD, not {date!r}")
    evidence = cast(object, permission.evidence)
    if not isinstance(evidence, Evidence) or evidence.kind not in EVIDENCE_KINDS:
        problems.append(f"the permission's evidence is an Evidence of kind {' or '.join(EVIDENCE_KINDS)}: nothing "
                        f"written is not accepted")
    else:
        pointer = cast(object, evidence.pointer)
        if not isinstance(pointer, str) or not pointer.strip():
            problems.append("the permission's evidence has no pointer: the link to the statement, or what the written "
                            "permission is and who holds it")
        elif evidence.kind == "public_statement" and (not pointer.startswith("https://")
                                                      or any(each.isspace() for each in pointer)):
            problems.append(f"a public statement's pointer is its https:// link, not {pointer!r}")
    return problems


def _valid_date(text: str) -> bool:
    import datetime

    try:
        datetime.date.fromisoformat(text)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class Verdict:
    """A file's verdict: its ``status`` in OpenMaterica's words, its ``tier`` (where its bytes live), the conditions
    its redistribution carries, and what decided it (``its licence``, ``a declared permission``, ``the author's
    own``)."""

    status: Status
    tier: Tier
    conditions: tuple[str, ...] = ()
    by: str = "its licence"

    @property
    def carried(self) -> bool:
        """Whether the bytes may travel inside a bundle."""
        return self.tier != "referenced"

    def words(self) -> str:
        status = self.status.replace("_", " ")
        conditions = f" ({', '.join(self.conditions)})" if self.conditions else ""
        return f"{status}, {self.tier}{conditions}"


def _rank(verdict: Verdict) -> tuple[int, int, int]:
    """Lower is freer: the tier, then the number of conditions, then unknown before not permitted."""
    return ({"open": 0, "conditional": 1, "referenced": 2}[verdict.tier], len(verdict.conditions),
            0 if verdict.status != "not_permitted" else 1)


def _leaf(identifier: str, custom: Mapping[str, CustomLicence]) -> Verdict:
    if identifier in OPEN:
        return Verdict("permitted", "open")
    if identifier in CONDITIONAL:
        return Verdict("permitted", "conditional", CONDITIONAL[identifier])
    held = custom.get(identifier)
    if held is not None and held.redistribution == "not_permitted":
        return Verdict("not_permitted", "referenced")
    return Verdict("unknown", "referenced")


def _all(verdicts: Sequence[Verdict]) -> Verdict:
    """AND: every licence's terms hold at once, so the most restrictive tier, every condition, and not permitted if any
    one is not."""
    worst = max(verdicts, key=_rank)
    if worst.tier == "referenced":
        status: Status = "not_permitted" if any(v.status == "not_permitted" for v in verdicts) else "unknown"
        return Verdict(status, "referenced")
    conditions = tuple(dict.fromkeys(c for v in verdicts for c in v.conditions))
    return Verdict("permitted", "conditional" if conditions else "open", conditions)


def _any(verdicts: Sequence[Verdict]) -> Verdict:
    """OR: the recipient may take any one, so the freest."""
    return min(verdicts, key=_rank)


class _Parser:
    """SPDX's grammar over a canonical expression: ``WITH`` binds tighter than ``AND``, which binds tighter than
    ``OR``."""

    def __init__(self, expression: str, custom: Mapping[str, CustomLicence]) -> None:
        self.tokens = _TOKEN.findall(expression)
        self.at = 0
        self.custom = custom

    def peek(self) -> str | None:
        return self.tokens[self.at] if self.at < len(self.tokens) else None

    def take(self) -> str:
        token = self.tokens[self.at]
        self.at += 1
        return token

    def either(self) -> Verdict:
        found = [self.both()]
        while self.peek() == "OR":
            self.take()
            found.append(self.both())
        return _any(found) if len(found) > 1 else found[0]

    def both(self) -> Verdict:
        found = [self.one()]
        while self.peek() == "AND":
            self.take()
            found.append(self.one())
        return _all(found) if len(found) > 1 else found[0]

    def one(self) -> Verdict:
        token = self.take()
        if token == "(":
            held = self.either()
            self.take()
        else:
            held = _leaf(token.removesuffix("+"), self.custom)
        if self.peek() == "WITH":
            self.take()
            self.take()  # an exception grants more, never less: the licence's verdict stands
        return held


def licence_verdict(expression: str, custom: Mapping[str, CustomLicence] | None = None) -> Verdict:
    """The verdict an SPDX licence expression gives a file on its own: ``permitted`` (``open`` or ``conditional``) only
    for licences on MIME's lists; ``not_permitted`` for a custom licence a person says forbids it; ``unknown`` for
    anything else, ``NOASSERTION`` included. ``AND`` takes the most restrictive, ``OR`` the freest. Raises
    ``ValueError`` for an expression :func:`licence_problem` refuses."""
    problem = licence_problem(expression)
    if problem is not None:
        raise ValueError(problem)
    if expression == NO_ASSERTION:
        return Verdict("unknown", "referenced")
    return _Parser(expression, custom or {}).either()


def permission_verdict(scope: Scope) -> Verdict:
    """The verdict a declared permission gives the files it names: ``permitted``, ``open`` when it allows modification
    and commercial use, ``conditional`` otherwise, with the conditions it leaves."""
    if not scope.redistribute:
        return Verdict("unknown", "referenced", by="a declared permission")
    conditions = tuple(word for word, held in ((NON_COMMERCIAL, not scope.commercial),
                                               (NO_DERIVATIVES, not scope.modify)) if held)
    return Verdict("permitted", "conditional" if conditions else "open", conditions, "a declared permission")


def freest(verdicts: Sequence[Verdict]) -> Verdict:
    """The freest of a file's verdicts (its licence's and each permission's that names it): a permission only ever
    raises a file's tier."""
    return min(verdicts, key=_rank)


__all__ = ["CONDITIONAL", "CUSTOM_PREFIX", "CUSTOM_STATUSES", "EVIDENCE_KINDS", "NON_COMMERCIAL",
           "NOTICE_TEXT_REQUIRED", "NO_ASSERTION", "NO_DERIVATIVES", "OPEN", "ROLES", "SHARE_ALIKE", "CustomLicence",
           "Evidence", "EvidenceKind", "LicenceGiven", "Permission", "Role", "Scope", "Status", "Terms", "Tier",
           "Verdict", "canonical", "expression_of", "freest", "ids_of", "licence_problem", "licence_verdict",
           "permission_problems", "permission_verdict"]
