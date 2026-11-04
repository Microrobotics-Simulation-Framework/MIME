"""``mime docs build``: the pages, generated from the registry export and the hand-written sources.

A build is a function of its inputs: the same checkout and the same installed namespaces give the same text, whatever
the hash seed, and no page holds a path of the machine that built it. Pages are never committed; a build writes them
to ``docs/_build/``, or to a directory the caller names, where a site finds ``index.md`` and ``pages/``.

A hand-written source that no longer resolves (its entry, version or id is gone) fails the build and writes no page,
rather than leave a page out and the run green."""

from __future__ import annotations

import re
import shutil
import time
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from mime.docs._context import INDEX_PATH, NOTE_PATHS, Context, entry_path, newest_versions, node_path
from mime.docs._entry_page import entry_page
from mime.docs._error_pages import error_pages
from mime.docs._evidence import Collection
from mime.docs._evidence_page import Inputs, evidence_pages, lane_reports
from mime.docs._export import export, load_namespaces
from mime.docs._family_page import family_page
from mime.docs._lists import read_list
from mime.docs._markdown import Hatch, Page
from mime.docs._node_page import node_page
from mime.docs._executable import expand
from mime.docs._generated import GENERATORS
from mime.docs._numbers import GUIDE_DIRECTORY, PAGE_DIRECTORIES
from mime.docs._reference_pages import index_page, mechanism_page, vocabulary_page
from mime.docs._specification_pages import migration_page, saved_form_pages, usd_reference_page
from mime.docs._walk import PageIds, page_sources, version_problems
from mime.registry import Registered, registered
from mime.testing import registered_families
from mime.vocabulary import TargetKind

BUILD_DIRECTORY = Path("docs/_build")
PAGES = "pages"
#: Written into the pages directory by a build: a directory a build may clear is one a build made.
MARKER = ".mime-docs-build"
_DISPLAY_EQUATION = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
#: An absolute path of a machine, which no page may carry: a build on a runner would write the runner's.
_ABSOLUTE_PATH = re.compile(r"(?<![\w./-])/(?:home|Users|root|tmp|var|mnt|opt|srv|private|runner)/\S")
_TITLE = re.compile(r"\A(\s*)# ")
_FRONT_MATTER = re.compile(r"\A---[ \t]*\n.*?\n---[ \t]*(?:\n|\Z)", re.DOTALL)


def strip_front_matter(text: str) -> str:
    return _FRONT_MATTER.sub("", text, count=1)


def demote_title(text: str) -> str:
    """``text`` with a title line at its top made a third-level heading: a source opens with its title, and inlined
    under the page's own second-level heading a first-level one would leave that section empty and the page with two
    titles."""
    return _TITLE.sub(r"\1### ", text, count=1)


def repository_relative(text: str, root: Path) -> str:
    """``text`` with the checkout's absolute path made relative to it: a refusal message, an environment stamp or a
    tutorial's output that carries a runtime path reads the same on every machine."""
    for base in sorted({str(root), str(root.resolve())}, key=len, reverse=True):
        text = text.replace(base + "/", "").replace(base, ".")
    return text


@dataclass
class Built:
    """What a build made: its pages by path under the output directory, and what stopped it, if anything."""

    pages: dict[str, Page] = field(default_factory=dict[str, Page])
    problems: list[str] = field(default_factory=list[str])
    stale: list[str] = field(default_factory=list[str])
    #: Evidence the counts page could not resolve (a test not collected): ``mime docs check`` reports those itself, in
    #: the anomaly script and the specification check, and a build fails on them too.
    unresolved: list[str] = field(default_factory=list[str])
    #: Lines for the summary that no page owns: what the evidence counts page was made from.
    notes: list[str] = field(default_factory=list[str])
    #: The pages of entries and nodes outside the namespaces whose bibliography is MIME's, whose citations are counted.
    outside: set[str] = field(default_factory=set[str])
    seconds: float = 0.0
    sources: int = 0

    @property
    def failures(self) -> list[str]:
        """Everything that stops a build writing: a source that no longer resolves, and the other problems. The docs
        walk reports the first kind itself, so ``mime docs check`` takes ``problems`` alone."""
        return [*self.stale, *self.problems, *self.unresolved]

    def texts(self) -> dict[str, str]:
        return {path: page.text for path, page in sorted(self.pages.items())}

    def hatches(self) -> Counter[str]:
        return Counter(hatch.kind for page in self.pages.values() for hatch in page.hatches)

    def hatch_items(self, kind: str) -> list[Hatch]:
        return [hatch for page in self.pages.values() for hatch in page.hatches if hatch.kind == kind]

    def own_hatch_items(self, kind: str) -> set[str]:
        """The items of the hatches of ``kind`` that the pages of MIME's namespace print: the toys' and a lab's are
        outside, counted and not listed."""
        return {hatch.item for path, page in self.pages.items() if path not in self.outside
                for hatch in page.hatches if hatch.kind == kind}

    def rendered(self) -> list[tuple[str, str, str]]:
        """The free text from code that the pages of MIME's namespace print, for the number rule: the page's name as
        the rule keys it, a label and the text, in the order the pages and the text come. The pages of another
        namespace (a lab's, the toys') are counted, not judged: its numbers are its own."""
        return [(f"generated/{page.path}", label, text) for path, page in sorted(self.pages.items())
                if path not in self.outside for label, text in page.rendered]

    def rendered_outside(self) -> int:
        return sum(len(page.rendered) for path, page in self.pages.items() if path in self.outside)

    def summary(self) -> list[str]:
        kinds = Counter(_category(path) for path in self.pages)
        counts = ", ".join(f"{count} {name}" for name, count in sorted(kinds.items()))
        hatches = self.hatches()
        return [f"pages: {len(self.pages)} ({counts}) from {self.sources} hand-written sources in {self.seconds:.2f} s; "
                f"text from the code of pages outside MIME, counted, not judged by the number rule: "
                f"{self.rendered_outside()} strings",
                "hatches printed: " + (", ".join(f"{kind} {count}" for kind, count in sorted(hatches.items()))
                                       or "none"), *self.notes]


def _category(path: str) -> str:
    parts = path.split("/")
    return parts[1] if len(parts) > 2 else parts[-1].removesuffix(".md")


def out_path(page: Page) -> str:
    return page.path if page.path == INDEX_PATH else f"{PAGES}/{page.path}"


def _node_classes(live: Iterable[Registered], problems: list[str]) -> dict[str, type]:
    """The node classes the entries' ``builds`` name, by dotted path, as the classes themselves (a class a test makes
    on the fly has a dotted name no import finds)."""
    nodes: dict[str, type] = {}
    for record in live:
        for built in dict(record.cls.builds).values():
            nodes.update({f"{node.__module__}.{node.__qualname__}": node for node in built})
    names: dict[str, str] = {}
    for dotted, cls in sorted(nodes.items()):
        if cls.__name__ in names and names[cls.__name__] != dotted:
            problems.append(f"two node classes are named {cls.__name__}, {names[cls.__name__]} and {dotted}; a node "
                            f"page is keyed by the class's name, so one of them must be renamed")
        names[cls.__name__] = dotted
    return nodes


def _hand_written(docs: Path, ids: PageIds, records: Mapping[tuple[str, str], list[Any]],
                  problems: list[str]) -> tuple[dict[str, str], int]:
    """The text of each hand-written source under ``docs`` (its front matter removed), keyed by its path; every source
    that no longer resolves is a problem, naming its file."""
    held: dict[str, str] = {}
    sources = page_sources(docs, PAGE_DIRECTORIES) if docs.is_dir() else []
    for relative in sources:
        why = ids.problem(relative)
        if why is not None:
            problems.append(f"{docs / relative}: {why}; a source that names nothing is stale: delete it, or restore "
                            f"what it names")
            continue
        text = (docs / relative).read_text(encoding="utf-8")
        if relative.parts[0] == "entries":
            problems += version_problems(docs / relative, text, records[relative.parts[1], relative.stem])
        held[relative.as_posix()] = demote_title(strip_front_matter(text))
    for note in NOTE_PATHS:
        if (docs / note).is_file():
            held[note] = strip_front_matter((docs / note).read_text(encoding="utf-8"))
    return held, len(held)


def guide_pages(root: Path, problems: list[str]) -> list[Page]:
    """The pages of ``docs/guide/``: each hand-written file with its includes replaced by the files' text and its
    generated texts by what the build generates, at ``guide/<name>.md``. A directive that does not resolve is a problem,
    naming the file; the guide holds no number the number rule does not read in its source."""
    pages: list[Page] = []
    for source in sorted((root / "docs" / GUIDE_DIRECTORY).glob("*.md")):
        if source.name.startswith("_"):
            continue
        text, found = expand(source.read_text(encoding="utf-8"), root, lambda name: GENERATORS[name](root))
        problems += [f"{source.relative_to(root)}: {problem}" for problem in found]
        pages.append(Page(f"{GUIDE_DIRECTORY}/{source.name}", text))
    return pages


def equation_problems(pages: Mapping[str, Page]) -> list[str]:
    """A display equation appears once, by its LaTeX with the whitespace removed: a duplicate, on one page or two,
    fails."""
    seen: dict[str, str] = {}
    found: list[str] = []
    for path, page in sorted(pages.items()):
        for match in _DISPLAY_EQUATION.finditer(page.text):
            key = "".join(match.group(1).split())
            if key in seen:
                found.append(f"the display equation {match.group(1).strip()!r} is on {seen[key]} and on {path}; an "
                             f"equation has one home, a node's governing_equations, and the other page links to it")
            else:
                seen[key] = path
    return found


def absolute_path_problems(pages: Mapping[str, Page]) -> list[str]:
    return [f"{path}: carries an absolute path of the machine that built it ({match.group()}...); a page is made "
            f"repository-relative when it is written" for path, page in sorted(pages.items())
            if (match := _ABSOLUTE_PATH.search(page.text)) is not None]


def build_pages(root: Path, evidence: Inputs = Inputs()) -> Built:
    """Every page of the checkout at ``root``, with MIME and every installed namespace loaded; the problems that stopped
    the build are in ``Built.problems``, and no page is to be written for a build that has any. The evidence counts
    page resolves its evidence as ``evidence`` says (by default it resolves none, and every row reads "not read")."""
    started = time.perf_counter()
    built = Built()
    data = export(load_namespaces())
    records = tuple(cast(list[Mapping[str, Any]], data["entries"]))
    live = registered()
    nodes = _node_classes(live, built.problems)
    by_entry: dict[tuple[str, str], list[Any]] = {}
    for record in live:
        by_entry.setdefault((record.kind.value.replace(" ", "_"), record.name), []).append(record)
    ids = PageIds(live, nodes.values())
    docs = root / "docs"
    hand, built.sources = _hand_written(docs, ids, by_entry, built.stale)
    unchecked = read_list(docs / "ratchets" / "builds_unchecked.txt").items() \
        if (docs / "ratchets" / "builds_unchecked.txt").is_file() else set[str]()
    units_list = docs / "ratchets" / "symbol_units.txt"
    units_unchecked = read_list(units_list).items() if units_list.is_file() else set[str]()
    guide = guide_pages(root, built.problems)
    hand.update({page.path: page.text for page in guide})
    context = Context(records, {family.name: family for family in registered_families()}, hand, nodes,
                      frozenset(unchecked), units_unchecked=frozenset(units_unchecked))
    made: list[Page] = [entry_page(record, context) for record in newest_versions(records)]
    made += [node_page(cls, dotted, context) for dotted, cls in sorted(nodes.items(), key=lambda p: p[1].__name__)]
    made += [family_page(family, context) for family in registered_families()]
    made += [mechanism_page(kind, context) for kind in TargetKind]
    counted = evidence_pages(root, evidence)
    built.problems += counted.problems
    built.unresolved += counted.unresolved
    built.notes += [counted.summary] if counted.summary else []
    references = [*saved_form_pages(root), *usd_reference_page(root), *migration_page(root, live), *counted.pages]
    made += [vocabulary_page(context), index_page(context, {page.path for page in references})]
    made += [Page(note, hand[note].strip("\n") + "\n") for note in NOTE_PATHS if note in hand]
    made += guide
    if (root / "src" / "mime" / "build").is_dir():
        listed = read_list(docs / "ratchets" / "refusal_codes.txt").items() \
            if (docs / "ratchets" / "refusal_codes.txt").is_file() else set[str]()
        made += error_pages(root / "src" / "mime", listed)
    made += references
    for page in made:
        page = Page(page.path, repository_relative(page.text, root), page.rendered, page.hatches)
        built.pages[out_path(page)] = page
    built.outside = _outside(records, nodes, context)
    built.problems += equation_problems(built.pages) + absolute_path_problems(built.pages)
    built.seconds = time.perf_counter() - started
    return built


def _outside(records: Iterable[Mapping[str, Any]], nodes: Mapping[str, type], context: Context) -> set[str]:
    """The pages (by path under the output) of the entries outside the cited namespaces, and of the nodes only such
    entries build."""
    outside = {out_path(Page(entry_path(record), "")) for record in records if record["namespace"] not in context.cited}
    for dotted, cls in nodes.items():
        builders = context.builders(dotted)
        if all(record["namespace"] not in context.cited for record, _ in builders):
            outside.add(out_path(Page(node_path(cls.__name__), "")))
    return outside


def write_pages(built: Built, out: Path) -> list[Path]:
    """Write the pages of a build that has no problem under ``out`` (``index.md`` and ``pages/``), clearing only a
    pages directory an earlier build made."""
    if built.failures:
        raise ValueError("a build with problems writes no page: " + "; ".join(built.failures))
    pages = out / PAGES
    if pages.exists():
        if not (pages / MARKER).is_file():
            raise FileExistsError(f"{pages} exists and no build made it (it holds no {MARKER}); refusing to clear it")
        shutil.rmtree(pages)
    written: list[Path] = []
    for path, page in sorted(built.pages.items()):
        target = out / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(page.text, encoding="utf-8", newline="\n")
        written.append(target)
    (pages / MARKER).write_text("pages made by `mime docs build`; a build clears this directory\n", encoding="utf-8", newline="\n")
    return written


def run_build(root: Path, out: Path | None, *, junit: Path | None = None, collection: Path | None = None,
              skip_evidence: bool = False) -> int:
    """``mime docs build``: print the summary, write the pages when the build has no problem; exit 1 when it has.
    The evidence counts page resolves the evidence it cites against ``collection`` (a collection ``mime docs check``
    wrote, as JSON) or, without one, against pytest's collection of the files it names, and reads the default lane's
    JUnit reports under ``junit``; without ``junit`` every row reads "outcome not read (no lane report)", and with
    ``skip_evidence`` nothing is collected."""
    root = root.resolve()
    held = Collection.from_json(collection.read_text(encoding="utf-8")) if collection is not None else None
    built = build_pages(root, Inputs(not skip_evidence, held,
                                     lane_reports(junit) if junit is not None and not skip_evidence else None))
    for line in built.summary():
        print(line)
    if built.failures:
        for problem in built.failures:
            print(f"FAIL: {problem}")
        print("mime docs build: failed; no page written")
        return 1
    target = (out if out is not None else root / BUILD_DIRECTORY).resolve()
    written = write_pages(built, target)
    print(f"mime docs build: wrote {len(written)} pages to {_shown(target, root)}")
    return 0


def _shown(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


__all__ = ["BUILD_DIRECTORY", "MARKER", "PAGES", "Built", "absolute_path_problems", "build_pages", "demote_title",
           "equation_problems", "out_path", "repository_relative", "run_build", "strip_front_matter", "write_pages"]
