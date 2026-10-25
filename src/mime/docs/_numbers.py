"""The number rule: a hand-written page source or a package README holds no measured, computed or fitted number. A
number such a file needs is a claim record's, named by its id (``see MIME-ANO-114``), so it has one home.

A line that carries a number token fails unless its key is listed: the file, the line's number tokens in order, and
its occurrence index ``#k``, the line's order among the lines of that file that carry the same tokens. So a wording
edit keeps a listed line, a changed, added, removed or swapped number is a new key, and a copy of a listed line is an
unlisted ``#2``. Nothing but the list is stored: the index is computed. ``docs/ratchets/number_allowlist.txt`` lists
a line with its own reason; ``docs/ratchets/readme_numbers.txt`` is the READMEs' frozen start, one shared reason,
which only shrinks: given a base revision, a line of it that the base's start list does not hold fails.

Exempt: code fences. A number in an equation (a node's ``governing_equations``, which this rule never reads) is a
symbol bound to a parameter, or comes from a claim record; no check holds that yet, so review does. A number spelled
in words is invisible to a digit rule: the word-numeral warning shows the forms with a cue, and fails nothing."""

from __future__ import annotations

import importlib.metadata
import re
import subprocess
import tomllib
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import mime
from mime.docs._lists import RatchetList, list_from_text, read_list

START_REASON = "present at D1-min"
ALLOWLIST = Path("docs/ratchets/number_allowlist.txt")
README_START = Path("docs/ratchets/readme_numbers.txt")
IDIOMS = Path("docs/number_idioms.txt")
#: The hand-written page sources under ``docs/``.
PAGE_DIRECTORIES = ("entries", "nodes", "families", "mechanisms", "assets")
#: The products a release is named after, each with the distribution whose versions are its releases.
PRODUCTS = {"MIME": "mime", "MADDENING": "maddening", "OpenMaterica": "openmaterica", "Python": "python",
            "usd-core": "usd-core"}
#: The distributions whose installed version is a release too, beside the pins.
INSTALLED = ("maddening", "openmaterica")
#: MIME's identifier forms, by prefix, each followed by three digits: a node's algorithm id, a known anomaly, a 0.2
#: verification benchmark, and a MADDENING known anomaly.
IDENTIFIERS = ("MIME-NODE-", "MIME-ANO-", "MIME-VER-", "MADD-ANO-")

# The word-numeral warning's units, which also end a release's or a phase's exemption: by name (any case, singular or
# plural) and by symbol (as written).
UNIT_NAMES = ("metre", "meter", "centimetre", "centimeter", "millimetre", "millimeter", "micrometre", "micrometer",
              "micron", "nanometre", "nanometer", "second", "millisecond", "microsecond", "nanosecond", "minute",
              "hour", "kilogram", "gram", "milligram", "newton", "pascal", "kelvin", "tesla", "millitesla", "hertz",
              "kilohertz", "radian", "degree", "ampere", "volt", "watt", "joule", "litre", "liter", "ulp")
UNIT_SYMBOLS = ("mm", "cm", "um", "\u00b5m", "\u03bcm", "nm", "ms", "us", "\u00b5s", "\u03bcs", "ns", "kg", "mg",
                "Pa", "mPa", "kPa", "mT", "Hz", "kHz", "MHz", "rad", "deg")
#: The single-letter symbols read. The rest (A, T, K, g, V, W, J, L) are left out, since a number word before a
#: letter is mostly not a quantity ("one A matrix", "one T junction").
UNIT_LETTERS = ("m", "s", "N")

_SIGN = "-+\u2212"
_SUPERSCRIPT_DIGITS = "\u2070\u00b9\u00b2\u00b3\u2074-\u2079"
_FRACTIONS = "\u00bc-\u00be\u2150-\u215e"
_UNIT = (rf"(?:(?:{'|'.join(map(re.escape, (*UNIT_SYMBOLS, *UNIT_LETTERS)))})(?![\w])|"
         rf"(?i:(?:{'|'.join(UNIT_NAMES)})(?:s|es)?)\b)")
#: Not before a unit or a per cent sign: what ends a release's or a phase's exemption.
_UNIT_AFTER = rf"(?!\s*(?:%|(?i:per\s*cent|percent)\b|{_UNIT}))"
_TOKEN = re.compile(rf"(?<![\w.])(?:[{_SIGN}](?=\.?\d|[{_FRACTIONS}]))?"
                    rf"(?:\d+(?:,\d{{3}})*(?:[.\u00b7]\d+)?[{_FRACTIONS}]?|\.\d+|[{_FRACTIONS}])"
                    rf"(?:[eE][{_SIGN}]?\d+)?(?:[\u207a\u207b]?[{_SUPERSCRIPT_DIGITS}]+)?\w*%?")
_IDENTIFIER = re.compile(rf"(?<![\w-])(?:{'|'.join(map(re.escape, IDENTIFIERS))})\d{{3}}(?![\w]|\.\d)"
                         r"|(?<![\w&#])#\d+(?![\w%]|\.\d)")
_VERSION = re.compile(r"(?<![\w.])(?:v\d+(?:\.\d+)+|\d+\.\d+\.\d+(?:\.\d+)*)(?:[.-]?(?:dev|a|b|rc|post)\d*)*(?![\w])")
_RELEASE = re.compile(rf"\b({'|'.join(map(re.escape, PRODUCTS))})(\s+)v?(\d+(?:\.\d+)*)(?![\w%]|\.\d){_UNIT_AFTER}")
_PHASE = rf"\d(?:[a-z]\d?)?(?![\w]|\.\d){_UNIT_AFTER}"
_PHASES = re.compile(rf"(?i:\bphases?)\s+{_PHASE}(?:(?:\s*,\s*|\s*,?\s+and\s+){_PHASE})*")
_LIST_ITEM = re.compile(r"^(\s*)(\d+)[.)](?=\s)")
_SECTION = re.compile(r"\u00a7\s*\d+(?:\.\d+)*[a-z]?")
_URL = re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s<>)\]]+")
_FENCE = re.compile(r"^(\s*)(`{3,}|~{3,})(.*)$")
_ITEM = re.compile(r"(\S+\.md) ((?:[^\s#]\S* )*[^\s#]\S*) #([1-9]\d*)")
_PIN = re.compile(r"(?:===|==|~=|!=|<=|>=|<|>)\s*v?(\d+(?:\.\d+)*)")
_DISTRIBUTION = re.compile(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)")

#: What one number token is, as the rule's summary prints it.
TOKEN_GRAMMAR = (
    "a number token is a run of digits, with a sign directly before it (-, + or \u2212) where it does not follow a "
    "word, digit groups (151,887), a decimal part (0.95, .5, 0\u00b795), an exponent (1e-5, 10\u207b\u00b3), a "
    "vulgar fraction (\u00bd, 1\u00bd) and a per cent sign (16%), and the letters or digits glued after it (4D, 5mm, "
    "4c1). Not tokens: a digit inside a word (float32, sha256, dev1, x_0, m\u00b2); an identifier, one of "
    f"{', '.join(IDENTIFIERS)} and three digits (MIME-ANO-114), or a numbered reference (#146); a version (0.3.0, "
    "v0.2, 0.3.0.dev1); a release the code reads (listed below), after its product's name and before no unit or per "
    "cent sign (MIME 0.3); a phase name after 'phase' or 'phases', a digit with at most a letter and a digit, before "
    "no unit (phase 4b, phase 7a2, phases 4 and 7); an ordered list item's number where it is 1 or counts its list "
    "from 1 by one; a section number (\u00a73.18); a URL; a code fence; and an idiom of docs/number_idioms.txt, "
    "removed before the line is keyed. A heading's number is a token, and so is a list item's out of its list's count")

# The word-numeral warning: a number word directly before one of four cues.
_ONES = ("one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen "
         "eighteen nineteen twenty").split()
_TENS = "thirty forty fifty sixty seventy eighty ninety".split()
_NUMBER_WORD = rf"(?:{'|'.join([*_ONES, *_TENS, 'hundred', 'thousand'])})"
_SPELLED = rf"\b{_NUMBER_WORD}(?:-{_NUMBER_WORD})*"
_WORD_CUES = (
    ("a unit", re.compile(rf"(?i:{_SPELLED})\s+{_UNIT}")),
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
    """The idioms as one pattern, each exact and standing alone: not inside a word or a number, and not before a unit,
    a per cent sign or a decimal part, so that ``6 x 6 mm`` and ``6 x 6.5`` are keyed on their numbers."""
    if not idioms:
        return None
    alternatives = "|".join(re.escape(idiom) for idiom in sorted(idioms, key=len, reverse=True))
    return re.compile(rf"(?<![\w.])(?:{alternatives})(?![\w]|\.\d){_UNIT_AFTER}")


def _normalised(distribution: str) -> str:
    return re.sub(r"[-_.]+", "-", distribution).lower()


def _requirements(project: Mapping[str, object]) -> Iterator[str]:
    """Every requirement string of a ``pyproject.toml``: the dependencies, the extras' and the groups'."""
    table = cast(Mapping[str, object], project.get("project", {}))
    extras = cast(Mapping[str, object], table.get("optional-dependencies", {}))
    groups = cast(Mapping[str, object], project.get("dependency-groups", {}))
    for held in (table.get("dependencies", []), *extras.values(), *groups.values()):
        items = cast(list[object], held) if isinstance(held, list) else []
        yield from (item for item in items if isinstance(item, str))


def releases(root: Path) -> dict[str, frozenset[str]]:
    """The releases the code can read, by distribution: ``mime.__version__``; every version a pin of ``root``'s
    ``pyproject.toml`` names (a dependency's bounds, ``requires-python``'s); MADDENING's and OpenMaterica's versions as
    installed; and each one's release number and its major.minor (``0.3`` of ``0.3.0.dev0``). Never a major alone, so
    ``Python 3`` names no release, and never a number nothing pins, so ``the 0.2 line`` names none."""
    found: dict[str, set[str]] = {distribution: set() for distribution in PRODUCTS.values()}

    def add(distribution: str, version: str) -> None:
        release = re.match(r"\d+(?:\.\d+)*", version)
        if distribution not in found or release is None:
            return
        parts = release.group().split(".")
        found[distribution] |= {version, release.group()} | ({".".join(parts[:2])} if len(parts) > 1 else set())

    add("mime", mime.__version__)
    try:
        project = cast(Mapping[str, object], tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8")))
    except (OSError, tomllib.TOMLDecodeError):
        project = {}
    for requirement in _requirements(project):
        name = _DISTRIBUTION.match(requirement)
        for version in _PIN.findall(requirement) if name is not None else ():
            add(_normalised(name.group(1)) if name is not None else "", version)
    python = cast(Mapping[str, object], project.get("project", {})).get("requires-python")
    for version in _PIN.findall(python) if isinstance(python, str) else ():
        add("python", version)
    for distribution in INSTALLED:
        try:
            add(distribution, importlib.metadata.version(distribution))
        except importlib.metadata.PackageNotFoundError:
            pass
    return {distribution: frozenset(versions) for distribution, versions in found.items()}


@dataclass(frozen=True)
class Reading:
    """What a line's tokens are read against: the idiom list's patterns, and the releases the code reads."""

    idioms: tuple[str, ...] = ()
    releases: Mapping[str, frozenset[str]] = field(default_factory=dict[str, frozenset[str]])
    pattern: re.Pattern[str] | None = field(init=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "pattern", idiom_pattern(self.idioms))


def reading_of(root: Path) -> tuple[Reading, list[str]]:
    """The reading of the checkout ``root`` (its idiom list and its releases), and what is wrong with the idiom list."""
    idioms, problems = read_idioms(root / IDIOMS)
    return Reading(idioms, releases(root)), problems


def number_tokens(line: str, reading: Reading | None = None, *, place: int = 1) -> tuple[str, ...]:
    """The number tokens of one line outside a code fence, in order, once every exempt form is removed. ``place`` is
    the place an ordered list's item takes on this line (1 for a line read alone): a list item's number is exempt
    when it is 1 or its place."""
    held = reading or Reading()

    def release(match: re.Match[str]) -> str:
        product, space, version = match.groups()
        return f"{product}{space} " if version in held.releases.get(PRODUCTS[product], frozenset()) else match.group()

    for pattern in (_URL, _IDENTIFIER, _VERSION, _SECTION, _PHASES):
        line = pattern.sub(" ", line)
    line = _RELEASE.sub(release, line)
    if held.pattern is not None:
        line = held.pattern.sub(" ", line)
    item = _LIST_ITEM.match(line)
    if item is not None and int(item.group(2)) in (1, place):
        line = item.group(1) + line[item.end():]
    return tuple(match.group() for match in _TOKEN.finditer(line))


def _fenced(text: str) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """The lines of ``text`` outside a code fence, with their numbers, and the lines whose fences are malformed."""
    lines: list[tuple[int, str]] = []
    problems: list[tuple[int, str]] = []
    fence: str | None = None
    opened = 0
    for number, line in enumerate(text.splitlines(), start=1):
        found = _FENCE.match(line)
        if fence is not None:
            if found and found.group(2)[0] == fence[0] and len(found.group(2)) >= len(fence) \
                    and not found.group(3).strip():
                fence = None
            continue
        if found and not (found.group(2)[0] == "`" and "`" in found.group(3)):
            fence, opened = found.group(2), number
            continue
        if found:
            problems.append((number, f"it opens with {found.group(2)} and holds another backtick after it, so it is "
                                     f"inline code, not a code fence; write inline code with single backticks, or "
                                     f"open the fence on a line of its own"))
        lines.append((number, line))
    if fence is not None:
        problems.append((opened, f"the code fence opened here ({fence}) is never closed, so the rule read none of the "
                                 f"file after it; close it"))
    return lines, problems


def prose_lines(text: str) -> list[tuple[int, str]]:
    """Each line of ``text`` outside a code fence, with its number."""
    return _fenced(text)[0]


def fence_problems(path: Path, text: str) -> list[str]:
    """Each line of the file ``path`` (holding ``text``) whose code fence is malformed: one never closed, and a run
    of backticks opening a line that is inline code, not a fence."""
    return [f"{path.as_posix()}:{number}: {why}" for number, why in _fenced(text)[1]]


def list_places(lines: Iterable[tuple[int, str]]) -> Iterator[tuple[int, str, int]]:
    """Each line with the place an ordered list's item takes there. A list counts from 1 by one at its indentation; a
    line at or left of that indentation that is no item of it ends it, and so does any line left of it."""
    expected: dict[int, int] = {}
    for number, line in lines:
        if not line.strip():
            yield number, line, 1
            continue
        indent = len(line) - len(line.lstrip())
        item = _LIST_ITEM.match(line) is not None
        for depth in [depth for depth in expected if depth > indent or (depth == indent and not item)]:
            del expected[depth]
        place = expected.get(indent, 1) if item else 1
        if item:
            expected[indent] = place + 1
        yield number, line, place


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


def occurrences(root: Path, path: Path, reading: Reading | None) -> list[Occurrence]:
    """Every line of ``path`` (relative to ``root``) carrying a number token, each indexed among that file's lines
    carrying the same tokens, in line order."""
    seen: dict[tuple[str, ...], int] = {}
    found: list[Occurrence] = []
    for number, line, place in list_places(prose_lines((root / path).read_text(encoding="utf-8"))):
        tokens = number_tokens(line, reading, place=place)
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


def base_start_list(root: Path, base: str) -> tuple[RatchetList | None, str | None]:
    """The README start list as the revision ``base`` of the repository at ``root`` holds it (``None`` when that
    revision has none), or why the revision cannot be read."""
    try:
        known = subprocess.run(["git", "-C", str(root), "rev-parse", "--verify", "--quiet", f"{base}^{{commit}}"],
                               capture_output=True, text=True, check=False)
        shown = subprocess.run(["git", "-C", str(root), "show", f"{base}:./{README_START.as_posix()}"],
                               capture_output=True, text=True, check=False)
    except OSError as error:
        return None, f"the base revision {base!r} cannot be read: {error}"
    if known.returncode != 0:
        return None, (f"the base revision {base!r} cannot be read in {root} ({shown.stderr.strip()}); fetch it, or "
                      f"pass another --base")
    if shown.returncode != 0:
        return None, None
    held = list_from_text(Path(f"{base}:{README_START.as_posix()}"), shown.stdout, number_item_form(readme_only=True))
    return held, None


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


def _start_against_base(root: Path, start: RatchetList, base: str | None, report: NumberReport) -> str:
    """Each line of the start list ``start`` that the base revision's does not hold, as a failure; and what the
    comparison was, for the summary."""
    if base is None:
        return f"{README_START}: not compared with a base revision (pass --base; CI passes origin/v0.3)"
    held, problem = base_start_list(root, base)
    if problem is not None:
        report.problems.append(problem)
        return f"{README_START}: not compared with {base}, which cannot be read"
    if held is None:
        return f"{README_START}: {base} holds no start list, so it is compared with nothing: this change adds it"
    before = held.items()
    for line in start.lines:
        if line.reason == START_REASON and line.item not in before:
            report.problems.append(f"{start.path}:{line.number}: {line.item!r} is not on the start list at {base}, "
                                   f"which only shrinks; a README line's number takes its own line in {ALLOWLIST}, "
                                   f"with its own reason logged, or goes (see MIME-ANO-nnn)")
    return f"{README_START}: compared with {base}'s, {len(held)} lines there and {len(start)} here"


def number_report(root: Path, base: str | None = None) -> NumberReport:
    """The number rule over the checkout ``root``: every line of its READMEs and page sources carrying a number
    token, keyed and checked against both lists, and the word-numeral warnings. With ``base``, a revision of the
    repository at ``root``, the start list is checked to hold no line the base's does not."""
    report = NumberReport()
    reading, report.problems = reading_of(root)
    files = scoped_files(root)
    for path in files:
        text = (root / path).read_text(encoding="utf-8")
        report.problems += fence_problems(path, text)
        report.occurrences += occurrences(root, path, reading)
        report.warnings += word_numerals(path, text)
    lists = read_number_lists(root)
    allowlist, start = lists["number_allowlist"], lists["readme_numbers"]
    report.problems += [*allowlist.problems, *start.problems]
    for line in start.lines:
        if line.reason and line.reason != START_REASON:
            report.problems.append(f"{start.path}:{line.number}: {line.item!r} has the reason {line.reason!r}; the "
                                   f"README start list holds only lines {START_REASON}, and a line with its own reason "
                                   f"is a line of {allowlist.path}")
    for line in allowlist.lines:
        if line.reason == START_REASON:
            report.problems.append(f"{allowlist.path}:{line.number}: {line.item!r} gives the start list's reason, "
                                   f"{START_REASON!r}; a line of {ALLOWLIST} gives its own")
    for item in sorted(allowlist.items() & start.items()):
        report.problems.append(f"{item!r} is on both {allowlist.path} and {start.path}; keep one")
    compared = _start_against_base(root, start, base, report)
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
    named = {product: sorted(reading.releases.get(distribution, frozenset()))
             for product, distribution in PRODUCTS.items()}
    report.summary += [
        f"number tokens: {TOKEN_GRAMMAR}",
        "releases (numbers the code reads, exempt after their product's name): "
        + "; ".join(f"{product} {', '.join(versions) or 'none'}" for product, versions in named.items()),
        f"number rule: {len(files)} files ({readmes} package READMEs, {len(files) - readmes} page sources); "
        f"{len(report.occurrences)} lines carry a number token, each keyed by file, tokens and occurrence: "
        f"{len(allowlist)} listed in {ALLOWLIST}, {len(start)} in {README_START}",
        compared,
        f"same-token groups (lines of one file carrying the same tokens): {len(shared)}"
        + (f" ({sizes})" if shared else ""),
        f"idioms: {len(reading.idioms)} ({IDIOMS}: {', '.join(reading.idioms) or 'none'})",
        f"word-numeral warnings: {len(report.warnings)}, never a failure (a number word before a unit, 'per cent' or "
        f"'percent', 'times', or 'fold'; units by name: {', '.join(UNIT_NAMES)}; by symbol: {', '.join(UNIT_SYMBOLS)}; "
        f"single letters: {', '.join(UNIT_LETTERS)}, the rest left out)"
        + "".join(f"\n  {warning.text()}" for warning in report.warnings),
    ]
    return report


__all__ = ["ALLOWLIST", "IDENTIFIERS", "IDIOMS", "INSTALLED", "PAGE_DIRECTORIES", "PRODUCTS", "README_START",
           "START_REASON", "TOKEN_GRAMMAR", "UNIT_LETTERS", "UNIT_NAMES", "UNIT_SYMBOLS", "NumberReport",
           "NumeralWarning", "Occurrence", "Reading", "base_start_list", "fence_problems", "idiom_pattern",
           "is_readme", "key_of", "list_places", "number_item_form", "number_report", "number_tokens", "occurrences",
           "prose_lines", "read_idioms", "read_number_lists", "reading_of", "releases", "scoped_files",
           "word_numerals"]
