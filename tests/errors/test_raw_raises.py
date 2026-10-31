"""The ratchet on MIME's raw raises: every ``raise`` of a builtin exception class (and of the few plain classes MIME
raises below its boundary) in ``src/mime`` is listed in ``raw_raises.txt`` with its count and why it carries no code.
A new one fails, until it is coded or listed with its reason; so does a listed one that is gone, so that the list
stays the source's.

**The control:** the scan of a copy of the package with one uncoded raise planted in a public call's function fails
the same check."""

from __future__ import annotations

import ast
import builtins
import re
import shutil
from collections import Counter
from pathlib import Path

import mime

PACKAGE = Path(mime.__file__).parent
LISTED = Path(__file__).parent / "raw_raises.txt"
#: The plain classes MIME raises below its boundary, beside Python's own.
PLAIN = frozenset({"UnitError", "TableRefused", "DescriptionRefused", "ClosedChainRefused", "UncataloguedRefusal"})
BUILTIN = frozenset(name for name in dir(builtins) if isinstance(getattr(builtins, name), type)
                    and issubclass(getattr(builtins, name), BaseException))
LINE = re.compile(r"(?P<key>\S+::\S+ \w+) (?P<count>\d+)  # (?P<reason>.+)")


def raw_raises(package: Path) -> Counter[str]:
    """Each raw raise in ``package``, as ``<file>::<function> <class>``, counted."""
    found: Counter[str] = Counter()
    for path in sorted(package.rglob("*.py")):
        names: list[str] = []
        relative = path.relative_to(package).as_posix()

        class Reader(ast.NodeVisitor):
            def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> None:
                names.append(node.name)
                self.generic_visit(node)
                names.pop()

            visit_AsyncFunctionDef = visit_FunctionDef
            visit_ClassDef = visit_FunctionDef

            def visit_Raise(self, node: ast.Raise) -> None:
                raised = node.exc
                if isinstance(raised, ast.Call) and isinstance(raised.func, ast.Name) \
                        and (raised.func.id in BUILTIN or raised.func.id in PLAIN):
                    found[f"{relative}::{'.'.join(names) or '<module>'} {raised.func.id}"] += 1
                self.generic_visit(node)

        Reader().visit(ast.parse(path.read_text(encoding="utf-8")))
    return found


def listed() -> dict[str, tuple[int, str]]:
    held: dict[str, tuple[int, str]] = {}
    for line in LISTED.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        found = LINE.fullmatch(line)
        assert found is not None, f"raw_raises.txt: {line!r} is not `<file>::<function> <class> <count>  # <reason>`"
        assert found["key"] not in held, f"raw_raises.txt lists {found['key']} twice"
        held[found["key"]] = (int(found["count"]), found["reason"])
    return held


def problems(found: Counter[str], held: dict[str, tuple[int, str]]) -> list[str]:
    said: list[str] = []
    for key, count in sorted(found.items()):
        if key not in held:
            said.append(f"{key}: {count} raw raise(s) with no code and no line in raw_raises.txt; code it (a Refusal "
                        f"in a MIME error) or list it with its reason")
        elif count > held[key][0]:
            said.append(f"{key}: {count} raw raises, and raw_raises.txt lists {held[key][0]}")
    said += [f"{key}: listed with {count}, and the source holds {found.get(key, 0)}; lower or remove its line"
             for key, (count, _) in sorted(held.items()) if found.get(key, 0) < count]
    return said


def test_every_raw_raise_in_mime_is_listed_with_its_reason():
    held = listed()
    assert problems(raw_raises(PACKAGE), held) == []
    assert all(len(reason) > 20 for _, reason in held.values()), "each line says why its raise carries no code"


def test_the_control_a_planted_uncoded_raise_fails_the_ratchet(tmp_path: Path):
    copy = tmp_path / "mime"
    shutil.copytree(PACKAGE, copy, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    experiment = copy / "experiment" / "_experiment.py"
    source = experiment.read_text(encoding="utf-8")
    anchor = "    def held_inputs(self) -> dict[str, dict[str, Any]]:\n"
    assert anchor in source
    experiment.write_text(source.replace(anchor, anchor + '        raise ValueError("a planted uncoded raise")\n', 1),
                          encoding="utf-8")
    found = problems(raw_raises(copy), listed())
    assert found == ["experiment/_experiment.py::BuiltExperiment.held_inputs ValueError: 1 raw raise(s) with no code "
                     "and no line in raw_raises.txt; code it (a Refusal in a MIME error) or list it with its reason"]
