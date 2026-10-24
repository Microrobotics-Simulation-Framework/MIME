"""The number rule: a hand-written page source or a package README holds no measured, computed or fitted number. A
number such a file needs is a claim record's, named by its id (``see MIME-ANO-114``), so it has one home.

A line that carries a number token fails unless its key is listed: the file, the line's number tokens in order, and
its occurrence index ``#k``, the line's order among the lines of that file that carry the same tokens. So a wording
edit keeps a listed line, a changed, added, removed or swapped number is a new key, and a copy of a listed line is an
unlisted ``#2``. Nothing but the list is stored: the index is computed. ``docs/ratchets/number_allowlist.txt`` lists
a line with its own reason; ``docs/ratchets/readme_numbers.txt`` is the READMEs' frozen start, one shared reason,
which only shrinks.

Exempt: code fences. A number in an equation (a node's ``governing_equations``, which this rule never reads) is a
symbol bound to a parameter, or comes from a claim record; no check holds that yet, so review does. A number spelled
in words is invisible to a digit rule: the word-numeral warning shows the forms with a cue, and fails nothing."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from mime.docs._lists import RatchetList, read_list

#: What one number token is, as the rule's summary prints it.
TOKEN_GRAMMAR = (
    "a number token is a run of digits, with a sign directly before it (-, + or \u2212) where it does not follow a "
    "word, digit groups (151,887), a decimal part (0.95, .5), an exponent (1e-5) and a per cent sign (16%), and the "
    "letters or digits glued after it (4D, 5mm, 4c1). Not tokens: a digit inside a word (float32, sha256, dev1, x_0); "
    "an identifier (MIME-ANO-114); a version (0.3.0, v0.2, 0.3.0.dev1), and a release named as one, after MIME, "
    "MADDENING, OpenMaterica, Python or usd-core, or before 'line' (MIME 0.3, the 0.2 line); a phase name after "
    "'phase' or 'phases' (phase 4b, phases 4 and 7); a list item's or a heading's number; a section number "
    "(\u00a73.18); a URL; a code fence; and an idiom of docs/number_idioms.txt, removed before the line is keyed")
START_REASON = "present at D1-min"
ALLOWLIST = Path("docs/ratchets/number_allowlist.txt")
README_START = Path("docs/ratchets/readme_numbers.txt")
IDIOMS = Path("docs/number_idioms.txt")
#: The hand-written page sources under ``docs/``.
PAGE_DIRECTORIES = ("entries", "nodes", "families", "mechanisms", "assets")

_SIGN = "-+\u2212"
_TOKEN = re.compile(rf"(?<![\w.])(?:[{_SIGN}](?=\.?\d))?(?:\d+(?:,\d{{3}})*(?:\.\d+)?|\.\d+)"
                    rf"(?:[eE][{_SIGN}]?\d+)?\w*%?")
_IDENTIFIER = re.compile(r"\b[A-Z][A-Z0-9]*(?:-[A-Z][A-Z0-9]*)*-\d+(?:\.\d+)*\b")
_VERSION = re.compile(r"(?<![\w.])(?:v\d+(?:\.\d+)+|\d+\.\d+\.\d+(?:\.\d+)*)(?:[.-]?(?:dev|a|b|rc|post)\d*)*(?![\w])")
#: A release named as a release: after the name of what it is a release of, or before "line" ("the 0.2 line").
PRODUCTS = ("MIME", "MADDENING", "OpenMaterica", "Python", "usd-core")
_RELEASE = re.compile(rf"\b((?:{'|'.join(PRODUCTS)})\s+)v?\d+(?:\.\d+)*(?![\w]|\.\d)|(?<![\w.])\d+\.\d+(?=\s+line\b)")
_PHASE = r"\d+(?:[a-z]\d*){0,3}(?![\w]|\.\d)"
_PHASES = re.compile(rf"(?i:\bphases?)\s+{_PHASE}(?:(?:\s*,\s*|\s+(?:and|or|to)\s+){_PHASE})*")
_LIST_ITEM = re.compile(r"^(\s*)\d+[.)](?=\s)")
_HEADING = re.compile(r"^(#{1,6}\s+)\d+(?:\.\d+)*[.)]?(?=\s)")
_SECTION = re.compile(r"\u00a7\s*\d+(?:\.\d+)*[a-z]?")
_URL = re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s<>)\]]+")
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
_ITEM = re.compile(r"(\S+\.md) ((?:[^\s#]\S* )*[^\s#]\S*) #([1-9]\d*)")

# The word-numeral warning: a number word directly before one of four cues.
_ONES = ("one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen "
         "eighteen nineteen twenty").split()
_TENS = "thirty forty fifty sixty seventy eighty ninety".split()
_NUMBER_WORD = rf"(?:{'|'.join([*_ONES, *_TENS, 'hundred', 'thousand'])})"
_SPELLED = rf"\b{_NUMBER_WORD}(?:-{_NUMBER_WORD})*"
#: The units the warning reads, by name (any case, singular or plural) and by symbol (as written). Single-letter
#: symbols are left out, since a number word before a letter is mostly not a quantity ("one A matrix").
UNIT_NAMES = ("metre", "meter", "centimetre", "centimeter", "millimetre", "millimeter", "micrometre", "micrometer",
              "micron", "nanometre", "nanometer", "second", "millisecond", "microsecond", "nanosecond", "minute",
              "hour", "kilogram", "gram", "milligram", "newton", "pascal", "kelvin", "tesla", "millitesla", "hertz",
              "kilohertz", "radian", "degree", "ampere", "volt", "watt", "joule", "litre", "liter", "ulp")
UNIT_SYMBOLS = ("mm", "um", "\u00b5m", "\u03bcm", "nm", "ms", "us", "\u00b5s", "\u03bcs", "ns", "kg", "mg", "Pa",
                "mPa", "kPa", "mT", "Hz", "kHz", "MHz", "rad", "deg")
_WORD_CUES = (
    ("a unit", re.compile(rf"(?i:{_SPELLED})\s+(?:(?i:(?:{'|'.join(UNIT_NAMES)})(?:s|es)?)|"
                          rf"(?:{'|'.join(map(re.escape, UNIT_SYMBOLS))}))(?![\w])")),
    ("'per cent'", re.compile(rf"(?i:{_SPELLED}\s+(?:per\s+cent|percent)\b)")),
    ("'times'", re.compile(rf"(?i:{_SPELLED}\s+times\b)")),
    ("'fold'", re.compile(rf"(?i:{_SPELLED}(?:-|\s+)?fold\b)")),
)


def scoped_files(root: Path) -> list[Path]:
    """The files the rule reads under the checkout ``root``: every package README (``src/**/README*.md``) and every
    hand-written page source (``docs/<entries, nodes, families, mechanisms or assets>/**/*.md``), as paths relative to
    ``root``."""
    found = set((root / "src").rglob("README*.md"))
    for directory in PAGE_DIRECTORIES:
        found |= set((root / "docs" / directory).rglob("*.md"))
    return sorted(path.relative_to(root) for path in found if path.is_file())


def is_readme(path: Path) -> bool:
    return path.parts[:1] == ("src",) and path.name.startswith("README")


def read_idioms(path: Path) -> tuple[tuple[str, ...], list[str]]:
    """The idiom list's patterns, and what is wrong with its lines. It is a list the rule reads, not a ratchet: an
    idiom no file uses is no failure. Each line is an exact pattern holding a digit, and its reason."""
    held = read_list(path, lambda item: None if re.search(r"\d", item) else "an idiom is a digit form")
    return tuple(line.item for line in held.lines), list(held.problems)


def idiom_pattern(idioms: tuple[str, ...]) -> re.Pattern[str] | None:
    if not idioms:
        return None
    alternatives = "|".join(re.escape(idiom) for idiom in sorted(idioms, key=len, reverse=True))
    return re.compile(rf"(?<![\w.])(?:{alternatives})(?!\w)")


def number_tokens(line: str, idioms: re.Pattern[str] | None = None) -> tuple[str, ...]:
    """The number tokens of one line outside a code fence, in order, once every exempt form is removed."""
    for pattern in (_URL, _IDENTIFIER, _VERSION, _SECTION, _PHASES):
        line = pattern.sub(" ", line)
    line = _RELEASE.sub(lambda match: f"{match.group(1) or ''} ", line)
    if idioms is not None:
        line = idioms.sub(" ", line)
    line = _HEADING.sub(r"\1", _LIST_ITEM.sub(r"\1", line))
    return tuple(match.group() for match in _TOKEN.finditer(line))


def prose_lines(text: str) -> Iterator[tuple[int, str]]:
    """Each line of ``text`` outside a code fence, with its number."""
    fence: str | None = None
    for number, line in enumerate(text.splitlines(), start=1):
        opened = _FENCE.match(line)
        if fence is None and opened:
            fence = opened.group(1)
            continue
        if fence is not None:
            if opened and opened.group(1)[0] == fence[0] and len(opened.group(1)) >= len(fence):
                fence = None
            continue
        yield number, line


@dataclass(frozen=True)
class Occurrence:
    """A line carrying a number token: its file, line number, tokens and occurrence index, and its key."""

    path: Path
    line: int
    tokens: tuple[str, ...]
    index: int

    @property
    def key(self) -> str:
        return key_of(self.path, self.tokens, self.index)


def key_of(path: Path, tokens: tuple[str, ...], index: int) -> str:
    return f"{path.as_posix()} {' '.join(tokens)} #{index}"


def occurrences(root: Path, path: Path, idioms: re.Pattern[str] | None) -> list[Occurrence]:
    """Every line of ``path`` (relative to ``root``) carrying a number token, each indexed among that file's lines
    carrying the same tokens, in line order."""
    seen: dict[tuple[str, ...], int] = {}
    found: list[Occurrence] = []
    for number, line in prose_lines((root / path).read_text(encoding="utf-8")):
        tokens = number_tokens(line, idioms)
        if tokens:
            seen[tokens] = seen.get(tokens, 0) + 1
            found.append(Occurrence(path, number, tokens, seen[tokens]))
    return found


@dataclass(frozen=True)
class NumeralWarning:
    path: Path
    line: int
    phrase: str
    cue: str

    def text(self) -> str:
        return f"warning: {self.path.as_posix()}:{self.line}: {self.phrase!r} is a number spelled in words before " \
               f"{self.cue}; a number is a claim record's (see MIME-ANO-nnn), which the rule cannot check here"


def word_numerals(path: Path, text: str) -> list[NumeralWarning]:
    """Each number spelled in words directly before a cue, outside a code fence: never a failure."""
    found: list[NumeralWarning] = []
    for number, line in prose_lines(text):
        for cue, pattern in _WORD_CUES:
            found += [NumeralWarning(path, number, match.group(), cue) for match in pattern.finditer(line)]
    return sorted(found, key=lambda warning: (warning.line, warning.phrase))


def number_item_form(readme_only: bool) -> Callable[[str], str | None]:
    def problem(item: str) -> str | None:
        parsed = _ITEM.fullmatch(item)
        if parsed is None:
            return "an item is a file, its number tokens in order and an occurrence index, 'file  tokens  #k'"
        if readme_only and not is_readme(Path(parsed.group(1))):
            return "the README start list holds lines of the package READMEs (src/**/README*.md) only"
        return None
    return problem


def read_number_lists(root: Path) -> dict[str, RatchetList]:
    return {"number_allowlist": read_list(root / ALLOWLIST, number_item_form(readme_only=False)),
            "readme_numbers": read_list(root / README_START, number_item_form(readme_only=True))}


def _orphan(path: Path, line: int, item: str, held: Mapping[tuple[str, tuple[str, ...]], int]) -> str:
    parsed = _ITEM.fullmatch(item)
    assert parsed is not None
    file, tokens, index = parsed.group(1), tuple(parsed.group(2).split()), int(parsed.group(3))
    count = held.get((file, tokens), 0)
    where = f"{path}:{line}: {item!r} matches no line"
    if count == 0:
        return f"{where}: no line of {file} carries the tokens {' '.join(tokens)} now; delete this line"
    takes = "the remaining line now takes the #1 entry, whose reason was written for another line: read it" \
        if count == 1 else (f"the remaining lines now take the entries #1 to #{count}, whose reasons may have been "
                            f"written for other lines: read them")
    return (f"{where}: {count} line(s) of {file} carry the tokens {' '.join(tokens)}, keyed #1 to #{count}, not "
            f"#{index}; delete this line. Lines with the same tokens are numbered in line order, so once one is edited "
            f"or removed {takes} again (the index forces this visit; only you know which line restates which)")


@dataclass
class NumberReport:
    problems: list[str] = field(default_factory=list[str])
    warnings: list[NumeralWarning] = field(default_factory=list[NumeralWarning])
    summary: list[str] = field(default_factory=list[str])
    occurrences: list[Occurrence] = field(default_factory=list[Occurrence])


def number_report(root: Path) -> NumberReport:
    """The number rule over the checkout ``root``: every line of its READMEs and page sources carrying a number
    token, keyed and checked against both lists, and the word-numeral warnings."""
    report = NumberReport()
    idioms, report.problems = read_idioms(root / IDIOMS)
    pattern = idiom_pattern(idioms)
    files = scoped_files(root)
    for path in files:
        report.occurrences += occurrences(root, path, pattern)
        report.warnings += word_numerals(path, (root / path).read_text(encoding="utf-8"))
    lists = read_number_lists(root)
    allowlist, start = lists["number_allowlist"], lists["readme_numbers"]
    report.problems += [*allowlist.problems, *start.problems]
    for line in start.lines:
        if line.reason and line.reason != START_REASON:
            report.problems.append(f"{start.path}:{line.number}: {line.item!r} has the reason {line.reason!r}; the "
                                   f"README start list holds only lines {START_REASON}, and a line with its own reason "
                                   f"is a line of {allowlist.path}")
    for item in sorted(allowlist.items() & start.items()):
        report.problems.append(f"{item!r} is on both {allowlist.path} and {start.path}; keep one")
    keys = {occurrence.key for occurrence in report.occurrences}
    listed = allowlist.items() | start.items()
    for occurrence in report.occurrences:
        if occurrence.key in listed:
            continue
        tokens = " ".join(occurrence.tokens)
        copy = (f"; it is occurrence #{occurrence.index} of these tokens in its file, and a copy of a listed line "
                f"takes its own line") if occurrence.index > 1 else ""
        file = occurrence.path.as_posix()
        report.problems.append(f"{file}:{occurrence.line}: the number token(s) {tokens} in prose{copy}. A number is a "
                               f"claim record's (see MIME-ANO-nnn); or list the line as '{file}  {tokens}  "
                               f"#{occurrence.index}  # <reason>' in {ALLOWLIST}")
    groups: dict[tuple[str, tuple[str, ...]], int] = {}
    for occurrence in report.occurrences:
        group = (occurrence.path.as_posix(), occurrence.tokens)
        groups[group] = max(groups.get(group, 0), occurrence.index)
    for held in (allowlist, start):
        report.problems += [_orphan(held.path, line.number, line.item, groups)
                            for line in held.lines if _ITEM.fullmatch(line.item) and line.item not in keys]
    shared = sorted(size for size in groups.values() if size > 1)
    sizes = ", ".join(f"{shared.count(size)} of {size} lines" for size in sorted(set(shared)))
    readmes = sum(1 for path in files if is_readme(path))
    report.summary += [
        f"number tokens: {TOKEN_GRAMMAR}",
        f"number rule: {len(files)} files ({readmes} package READMEs, {len(files) - readmes} page sources); "
        f"{len(report.occurrences)} lines carry a number token, each keyed by file, tokens and occurrence: "
        f"{len(allowlist)} listed in {ALLOWLIST}, {len(start)} in {README_START}",
        f"same-token groups (lines of one file carrying the same tokens): {len(shared)}"
        + (f" ({sizes})" if shared else ""),
        f"idioms: {len(idioms)} ({IDIOMS}: {', '.join(idioms) or 'none'})",
        f"word-numeral warnings: {len(report.warnings)}, never a failure (a number word before a unit, 'per cent' or "
        f"'percent', 'times', or 'fold'; units by name: {', '.join(UNIT_NAMES)}; by symbol: {', '.join(UNIT_SYMBOLS)})"
        + "".join(f"\n  {warning.text()}" for warning in report.warnings),
    ]
    return report


__all__ = ["ALLOWLIST", "IDIOMS", "PAGE_DIRECTORIES", "PRODUCTS", "README_START", "START_REASON", "TOKEN_GRAMMAR",
           "UNIT_NAMES", "UNIT_SYMBOLS", "NumberReport", "NumeralWarning", "Occurrence", "idiom_pattern", "is_readme",
           "key_of", "number_item_form", "number_report", "number_tokens", "occurrences", "prose_lines", "read_idioms",
           "read_number_lists", "scoped_files", "word_numerals"]
