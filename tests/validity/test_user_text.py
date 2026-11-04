"""No text a user reads holds a label from the work that made it (the second walk's friction 4): the development
walk's own task names ("the walk's T1 sphere", "a run of T2's size") leaked into the stamp, the build report and
``mime pending`` through the measured words of a numerical default, and the phase names, design rows and PR numbers of
the work stood in the library's messages, its docstrings and its pages.

Two kinds of label are read. **Walk labels** (the task names of the development walks) are looked for everywhere text is
read, a private docstring and the changelog included. **Work labels** (a phase name such as ``phase 7m-b``, a design
row such as ``design #195``, a PR number, a bare ``#171``, the walks' ``D3``) are looked for in what a user reads: the
runtime strings of the library (messages, refusals, CLI output, records a command prints), the docstrings of its public
names (what ``help()`` prints), the hand-written docs pages, the package READMEs and the non-comment lines of the data
files beside them. Left on purpose: code comments and private helpers' docstrings; ``MIME-ANO-n`` (the anomaly register
ships, so it is a public record); the changelog and the register itself, which keep their history (the walk labels are
still looked for in them); and ``docs/ratchets``, the checker's own lists.

The text of an actual build (its stamp, its report, ``mime pending``) is read too.

**Controls.** Each scanner's surface is planted into, one plant at a time, in a copy of the layout built in a temporary
directory: a docs page, a library string (in the CLI), a public docstring, a package README, a data file, an example,
the changelog. Each plant fails the scan of its own file alone, so the scan reaches its surface and the regex is not
the only thing under test. A private docstring, a comment and the register are planted into too, and are not flagged:
the exemptions are held where they are written."""

import ast
import re
from collections.abc import Iterator
from pathlib import Path

import pytest
from profile_cases import Paperwork, settling_ball

from mime.registry import Entry
from mime.validity import Stamp

ROOT = Path(__file__).resolve().parents[2]
#: The task names of the development walks and the phrases that carried them.
WALK_LABELS = re.compile(
    r"walk's (?:T\d|tube|sphere|screw)|\b(?:first-hour|second|development) walk\b|\bthe walker\b|"
    r"\b(?:of|in|on) T[1-4]\b|\bT[1-4]'s\b|\bT[1-4] (?:sphere|screw|size|task)\b|\bdecision D\d+\b")
#: The labels of the work: phase names with a letter or past the build's two, design rows, PR numbers, a numbered
#: reference, the sub-phases and their pieces ("7m-b", "D3's", "P2a"), and the phrases that named a phase.
WORK_LABELS = re.compile(
    r"[Pp]hases? (?:\d[a-z]|[3-9]\b|\d\d)(?:-[a-z])?|\bdesign (?:row )?#\d+|\bPRs? #\d+|(?<![\w&/#])#\d{2,4}\b|"
    r"\b7[a-z](?:-[a-z])?\b(?!-?\d)|\bD[1-4][ab]?'s\b|\b(?:docs phase|since|before|after) D[1-4][ab]?\b|"
    r"\bdev[1-4]'s\b|\b(?:docs|errors'|specifications'|generators') phase|\b(?:first documentation|later|next) phase|"
    r"\bphase by phase\b|\bsharing piece\b|\bP[123][ab]\b|\blane [A-Z][0-9]\b")
#: Files that name a thing which only looks like a label: a GPU model.
OTHER_MEANINGS = {"src/mime/env/_gpus.py"}
#: Kept as a record of its history: the walk labels are looked for in them, the work labels are not.
RECORDS = {"CHANGELOG.md", "docs/validation/known_anomalies.yaml"}
#: The checker's own lists, which are not pages a user reads.
UNREAD = ("docs/ratchets/", "docs/unpaged.txt")
TEXT_SUFFIXES = {".md", ".txt", ".yaml", ".json"}
MAX_BYTES = 1_000_000


def walk_labels(text: str) -> list[str]:
    return WALK_LABELS.findall(text)


def work_labels(text: str) -> list[str]:
    return [match.group(0) for match in WORK_LABELS.finditer(text)]


def labels_in(text: str) -> list[str]:
    return [*walk_labels(text), *work_labels(text)]


def _exported_names(root: Path) -> set[str]:
    names: set[str] = set()
    for init in (root / "src" / "mime").rglob("__init__.py"):
        for node in ast.walk(ast.parse(init.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__"
                                                    for t in node.targets):
                names |= {e.value for e in ast.walk(node.value) if isinstance(e, ast.Constant)
                          and isinstance(e.value, str)}
    return names


def _docstring_of(node: ast.AST) -> ast.Constant | None:
    body = getattr(node, "body", None)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        return body[0].value
    return None


def _public_docstrings(tree: ast.Module, public_module: bool, exported: frozenset[str]) -> set[int]:
    """The ids of the docstring constants ``help()`` can print: the module's when no part of its path is private, and
    those of the public classes and functions (a name is public when it does not begin with an underscore and the
    module is public or the package exports it) and of the public methods of a public class."""
    found: set[int] = set()
    if public_module and (module_doc := _docstring_of(tree)) is not None:
        found.add(id(module_doc))

    def visit(parent: ast.AST, parent_public: bool, top: bool) -> None:
        for child in ast.iter_child_nodes(parent):
            if not isinstance(child, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            if top:
                public = not child.name.startswith("_") and (public_module or child.name in exported)
            else:
                dunder = child.name.startswith("__") and child.name.endswith("__")
                public = (not child.name.startswith("_") or dunder) and parent_public
            if public and (doc := _docstring_of(child)) is not None:
                found.add(id(doc))
            if isinstance(child, ast.ClassDef):
                visit(child, public, False)

    visit(tree, True, True)
    return found


def library_strings(source: str, public_module: bool = True, exported: frozenset[str] = frozenset()
                    ) -> Iterator[tuple[str, bool]]:
    """Every string of a module's source with whether a user reads it: a string that is not a docstring (a runtime
    message) is read, a docstring is read when it is a public name's. A private helper's docstring comes with False."""
    tree = ast.parse(source)
    public_docs = _public_docstrings(tree, public_module, exported)
    docstrings = {id(doc) for node in ast.walk(tree)
                  if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
                  and (doc := _docstring_of(node)) is not None}
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.value, id(node) not in docstrings or id(node) in public_docs


def labelled_strings(source: str, public_module: bool = True, exported: frozenset[str] = frozenset()) -> list[str]:
    """The strings in a module's source that hold a label: a walk label in any string, a work label in one a user
    reads."""
    return [text for text, read in library_strings(source, public_module, exported)
            if walk_labels(text) or (read and work_labels(text))]


def library_files(root: Path) -> list[Path]:
    return sorted((root / "src" / "mime").rglob("*.py"))


def page_files(root: Path) -> list[Path]:
    """The hand-written pages, the examples, and the text files beside the code."""
    candidates = [*(root / "docs").rglob("*"), *(root / "examples").rglob("*"), *(root / "src" / "mime").rglob("*"),
                  root / "README.md", root / "CHANGELOG.md"]
    return sorted(path for path in candidates if path.is_file() and path.stat().st_size <= MAX_BYTES and (
        path.suffix in TEXT_SUFFIXES or path.suffix == ".py" and path.relative_to(root).parts[0] == "examples"))


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def library_findings(root: Path) -> dict[str, list[str]]:
    exported = frozenset(_exported_names(root))
    found: dict[str, list[str]] = {}
    for path in library_files(root):
        relative = _relative(path, root)
        parts = path.relative_to(root / "src" / "mime").parts
        public_module = not any(part.startswith("_") and part != "__init__.py" for part in parts)
        if relative not in OTHER_MEANINGS and (
                held := labelled_strings(path.read_text(encoding="utf-8"), public_module, exported)):
            found[relative] = [text[:80] for text in held]
    return found


def _text_as_read(path: Path, root: Path) -> str:
    """A page's text as a user reads it: a data file beside the code (not Markdown) without its whole-line comments,
    which are code comments."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix != ".md" and _relative(path, root).startswith("src/mime/"):
        return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    return text


def page_findings(root: Path) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for path in page_files(root):
        relative = _relative(path, root)
        if relative.startswith(UNREAD):
            continue
        text = _text_as_read(path, root)
        held = walk_labels(text) + ([] if relative in RECORDS else work_labels(text))
        if held:
            found[relative] = held
    return found


def test_no_string_of_the_library_holds_a_label():
    assert library_findings(ROOT) == {}


def test_no_page_a_user_reads_holds_a_label():
    assert page_findings(ROOT) == {}


def test_the_scans_reach_the_surfaces_they_are_named_for():
    pages = {_relative(path, ROOT) for path in page_files(ROOT)}
    library = {_relative(path, ROOT) for path in library_files(ROOT)}
    assert {"README.md", "CHANGELOG.md", "docs/README.md", "docs/choosing_spacing.md", "src/mime/README.md",
            "src/mime/cli/README.md", "docs/validation/known_anomalies.yaml"} <= pages
    assert any(name.startswith("docs/entries/") for name in pages)
    assert any(name.startswith("examples/") for name in pages)
    assert any(name.startswith("src/mime/") and name.endswith(".txt") for name in pages)
    assert {"src/mime/cli/_summary.py", "src/mime/interactions/hydrodynamic.py",
            "src/mime/validity/_profiles.py"} <= library


def test_the_text_of_a_build_the_stamp_its_report_and_mime_pending_holds_no_label():
    built = settling_ball(Paperwork(density="steel")).build()
    texts = [built.report.profile.stamp().text(), built.report.text(details=True), built.report.pending(),
             *built.report.profile.stamp().measured]
    assert all(text for text in texts) and [labels_in(text) for text in texts] == [[]] * len(texts)
    assert "above Stokes's law" in built.report.pending(), "the measured words are there to be read"


# --- the controls: a plant into each scanner's surface -------------------------------------------------------------

CLEAN_FILES = {
    "README.md": "A library.\n",
    "CHANGELOG.md": "# Changelog\n- a change (MIME-ANO-114)\n",
    "docs/page.md": "A page a user reads.\n",
    "docs/entries/kind/entry.md": "An entry page.\n",
    "docs/validation/known_anomalies.yaml": "- id: MIME-ANO-1\n  text: recorded\n",
    "docs/ratchets/list.txt": "a row\n",
    "examples/example.py": "print('an example')\n",
    "src/mime/__init__.py": '__all__ = ["Exported"]\n',
    "src/mime/README.md": "A package.\n",
    "src/mime/cli/README.md": "The command.\n",
    "src/mime/cli/_main.py": 'MESSAGE = "a message"\n',
    "src/mime/things/__init__.py": '"""Things."""\n',
    "src/mime/things/bodies.py": 'def build():\n    """Builds a body."""\n',
    "src/mime/things/_hidden.py": 'def helper():\n    """A private helper."""\n',
    "src/mime/things/data.txt": "# a comment line\nrule = value\n",
}


def planted_tree(root: Path, name: str, add: str) -> None:
    """The clean layout under ``root`` with ``add`` appended to the file ``name``, a file of the layout."""
    assert name in CLEAN_FILES, "a plant goes into a file of the layout"
    for relative, text in CLEAN_FILES.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + (add if relative == name else ""), encoding="utf-8")


def all_findings(root: Path) -> dict[str, list[str]]:
    return {**library_findings(root), **page_findings(root)}


def test_the_control_the_clean_layout_is_passed(tmp_path: Path):
    planted_tree(tmp_path, "README.md", "")
    assert all_findings(tmp_path) == {}


#: (file planted into, the text added, whether the scan must flag it): each surface once, then the exemptions.
PLANTS = [
    ("docs/page.md", "Since drag 0.4.0 (phase 7m-b) the rung separates.\n", True),
    ("docs/entries/kind/entry.md", "chosen by a rule fixed before it (design #210).\n", True),
    ("docs/validation/known_anomalies.yaml", "  note: as the walk's T1 sphere showed\n", True),
    ("README.md", "More arrives in a PR #109 too.\n", True),
    ("examples/example.py", "print('as the first-hour walk did')\n", True),
    ("src/mime/README.md", "Since phase 7a2, the arm.\n", True),
    ("src/mime/cli/README.md", "Landed in dev1's phase 3.\n", True),
    ("src/mime/cli/_main.py", 'OTHER = "uploaded (#171)"\n', True),
    ("src/mime/things/bodies.py", 'def public():\n    """Chosen (design #191 item 4)."""\n', True),
    ("src/mime/things/data.txt", "rule = value  # but phase 7h\nsecond = see the review of PR #60\n", True),
    ("CHANGELOG.md", "- as the first-hour walk found\n", True),
    ("src/mime/things/_hidden.py", 'def another():\n    """A private helper (phase 7o)."""\n', False),
    ("src/mime/things/data.txt", "# a comment of phase 7o\n", False),
    ("CHANGELOG.md", "- a change (phase 7o, design #195)\n", False),
    ("docs/validation/known_anomalies.yaml", "  note: since phase 7a, dev1's\n", False),
    ("docs/ratchets/list.txt", "a row, phase 7n-b\n", False),
]


@pytest.mark.parametrize(("name", "add", "flagged"), PLANTS,
                         ids=[f"{name}-{'flagged' if flagged else 'passed'}" for name, _, flagged in PLANTS])
def test_the_control_a_plant_fails_the_scan_of_its_own_file_alone(tmp_path: Path, name: str, add: str, flagged: bool):
    planted_tree(tmp_path, name, add)
    found = all_findings(tmp_path)
    assert set(found) == ({name} if flagged else set()), found


def test_the_control_a_name_the_package_exports_is_read_from_a_private_module_and_one_it_does_not_is_not(tmp_path: Path):
    planted_tree(tmp_path, "README.md", "")
    private = tmp_path / "src" / "mime" / "things" / "_core.py"
    private.write_text('class Exported:\n    """Chosen (design #191)."""\n\n\nclass Unexported:\n'
                       '    """Chosen (design #191)."""\n', encoding="utf-8")
    assert set(library_findings(tmp_path)) == {"src/mime/things/_core.py"}
    private.write_text('class Unexported:\n    """Chosen (design #191)."""\n', encoding="utf-8")
    assert library_findings(tmp_path) == {}


def test_the_control_each_kind_of_label_is_matched_and_a_walk_label_in_a_private_docstring_still_is():
    for text in ("(phase 7m-b)", "dev1's phase 7a3", "design #220", "review of PR #60", "uploaded (#171)",
                 "since 7m-a", "D3's fix round", "push's P2a", "the specifications' phase"):
        assert work_labels(text), text
    assert labelled_strings('def _private():\n    """a run of T2\'s size in the tube"""', public_module=False)
    assert labelled_strings('MEASURED = "on the walk\'s T1 sphere (338 samples), T1\'s 1000 steps in 22 s"')
    assert labels_in("A screw turned at 5 Hz in water, as the first-hour walk's was, is in the hundreds") == \
        ["first-hour walk"]
    planted = Stamp("exploratory", ("drag:ball.spacing (a numerical default)",),
                    ("drag:ball.spacing: on the walk's T1 sphere (338 force points)",))
    assert labels_in(" ".join(planted.measured))
    assert Entry is not None


def test_the_control_text_with_no_label_and_a_gpu_named_t4_is_passed():
    assert labels_in("on a 100 um steel sphere settling in still water (338 samples), 1000 steps in 22 s") == []
    for text in ("phase 1 checks and phase 2 constructs", "the lag of 1e-4 in phase", "MIME-ANO-114 and a 7e-5 offset",
                 "float32, 0.3.0.dev1, a section #status, &#123;, #fff", "#1 in the list"):
        assert labels_in(text) == [], text
    assert labelled_strings('GPUS = {"T4": ("T4",)}') == [] and labelled_strings('X = "a T4 GPU"') == []
