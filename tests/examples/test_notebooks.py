"""The notebooks of ``docs/notebooks/`` hold their example's code verbatim, and are committed without outputs.

A code cell tagged ``example`` (the cell metadata's ``tags``) is the example's own code. Each statement in it is one of
the example's definitions (a function, a class, an assignment), the same text as the example's together with the
comment lines directly above it, or an import of a name the example imports from the same module. Every definition
of the example is in the notebook's example cells, but those the notebook's entry here leaves out, each with why. No
other cell binds a name the example defines, so a later cell cannot change what the example cells declare.

The notebooks install a released MIME from PyPI, pinned by version, while these checks hold their code to this
checkout's examples: when an example changes after a release, its notebook follows it here and the pin moves at the
next release.

Each check is a function of the notebook's and the example's text, so each control hands it the real notebook with one
planted difference, which it must report on every run."""

import ast
import copy
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = ROOT / "docs" / "notebooks"
EXAMPLE_TAG = "example"
#: Each notebook, the example whose code it holds, and the example's definitions it leaves out, each with why.
HELD: dict[str, tuple[str, dict[str, str]]] = {
    "sphere_settling_tube.ipynb": ("examples/sphere_settling_tube.py", {
        "main": "the command line's entry: the notebook's own cells run the run, print its numbers and show its view"}),
    "tutorial_a.ipynb": ("examples/script_a.py", {
        "record_arm": "the notebook builds the reproduction arm only; the guide's tutorial A builds the record arm",
        "exploring_arm": "the record arm before it is sealed, which the notebook does not build",
        "main": "the command line's entry: the notebook's own cells run the reproduction arm and read its numbers"}),
}

Statement = ast.stmt
_DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)


def code_cells(notebook: Mapping[str, Any]) -> Iterator[tuple[int, str, bool, Mapping[str, Any]]]:
    """Each code cell of ``notebook``: its number (counting every cell from 1), its source with the shell and magic
    lines (``!``, ``%``) blanked so that Python parses it, whether it is tagged ``example``, and the cell."""
    for number, cell in enumerate(notebook["cells"], 1):
        if cell["cell_type"] != "code":
            continue
        source = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
        lines = ["" if line.lstrip().startswith(("!", "%")) else line for line in source.split("\n")]
        yield number, "\n".join(lines), EXAMPLE_TAG in cell.get("metadata", {}).get("tags", []), cell


def bound_names(statement: Statement) -> list[str]:
    """The names a definition binds."""
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [statement.name]
    if isinstance(statement, ast.Assign):
        targets = statement.targets
    elif isinstance(statement, ast.AnnAssign):
        targets = [statement.target]
    else:
        return []
    return [node.id for target in targets for node in ast.walk(target)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)]


def definition_text(lines: list[str], statement: Statement) -> str:
    """The statement's text, from its first decorator, with the comment lines directly above it."""
    decorators = getattr(statement, "decorator_list", [])
    first = min([statement.lineno, *(decorator.lineno for decorator in decorators)]) - 1
    while first > 0 and lines[first - 1].lstrip().startswith("#"):
        first -= 1
    return "\n".join(lines[first:statement.end_lineno])


def definitions(source: str) -> dict[str, str]:
    """Each name the top level of ``source`` defines, with its definition's text."""
    lines = source.split("\n")
    return {name: definition_text(lines, statement) for statement in ast.parse(source).body
            if isinstance(statement, _DEFINITIONS) for name in bound_names(statement)}


def imported(statement: ast.Import | ast.ImportFrom) -> list[tuple[str, str, str]]:
    """Each name an import binds, as (module, name, the name it is bound to)."""
    module = (statement.module or "") if isinstance(statement, ast.ImportFrom) else ""
    return [(module, alias.name, alias.asname or alias.name.split(".")[0]) for alias in statement.names]


def imports(source: str) -> set[tuple[str, str, str]]:
    return {each for statement in ast.parse(source).body if isinstance(statement, (ast.Import, ast.ImportFrom))
            for each in imported(statement)}


def names_a_cell_binds(tree: ast.Module) -> set[str]:
    """Every name the cell binds in the notebook's namespace: its assignments, loop and ``with`` targets, imports,
    functions and classes at any depth outside a function's or class's body."""
    found: set[str] = set()
    pending: list[ast.AST] = list(tree.body)
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found.add(node.name)
            continue
        if isinstance(node, ast.Lambda):
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            found |= {bound for _, _, bound in imported(node)}
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            found.add(node.id)
        pending += list(ast.iter_child_nodes(node))
    return found


def notebook_problems(notebook: Mapping[str, Any], example: str, left_out: Mapping[str, str]) -> list[str]:
    """What ``notebook`` holds that is not the example's code (``example``, its text), what of the example it lacks
    beside ``left_out``, and every cell committed with outputs."""
    problems: list[str] = []
    defined, held_imports = definitions(example), imports(example)
    problems += [f"leaves out {name}, which the example does not define: remove it from HELD"
                 for name in sorted(set(left_out) - set(defined))]
    held: set[str] = set()
    for number, source, tagged, cell in code_cells(notebook):
        if cell.get("outputs") or cell.get("execution_count") is not None:
            problems.append(f"cell {number} is committed with outputs: clear them")
        tree = ast.parse(source)
        if not tagged:
            problems += [f"cell {number}, not tagged {EXAMPLE_TAG!r}, binds {name}, which the example defines"
                         for name in sorted(names_a_cell_binds(tree) & set(defined))]
            continue
        lines = source.split("\n")
        for statement in tree.body:
            if isinstance(statement, (ast.Import, ast.ImportFrom)):
                problems += [f"cell {number} imports {name} from {module or 'the top level'}, which the example does "
                             f"not" for module, name, bound in imported(statement)
                             if (module, name, bound) not in held_imports]
            elif isinstance(statement, _DEFINITIONS):
                text = definition_text(lines, statement)
                for name in bound_names(statement):
                    if name not in defined:
                        problems.append(f"cell {number} defines {name}, which the example does not")
                    elif text != defined[name]:
                        problems.append(f"cell {number}: {name} is not the example's text; copy it from the example")
                    held.add(name)
            else:
                problems.append(f"cell {number}, tagged {EXAMPLE_TAG!r}, holds line {statement.lineno}, which is "
                                f"neither a definition nor an import of the example's")
    problems += [f"the example's {name} is in no cell tagged {EXAMPLE_TAG!r}"
                 for name in sorted(set(defined) - set(left_out) - held)]
    return problems


def read_notebook(name: str) -> dict[str, Any]:
    return json.loads((NOTEBOOKS / name).read_text(encoding="utf-8"))


def example_text(name: str) -> str:
    return (ROOT / HELD[name][0]).read_text(encoding="utf-8")


@pytest.mark.parametrize("name", sorted(HELD))
def test_the_notebook_holds_its_examples_code_verbatim_and_no_outputs(name):
    notebook = read_notebook(name)
    tagged = [source for _, source, is_tagged, _ in code_cells(notebook) if is_tagged]
    assert tagged, f"{name} has no cell tagged {EXAMPLE_TAG!r}, so it could not differ from its example"
    assert notebook_problems(notebook, example_text(name), HELD[name][1]) == []


def test_every_notebook_is_held_to_an_example():
    assert sorted(path.name for path in NOTEBOOKS.glob("*.ipynb")) == sorted(HELD)


def planted(name: str, cell_holding: str, old: str, new: str) -> dict[str, Any]:
    """The real notebook ``name`` with ``old`` replaced by ``new`` in the one code cell whose source holds
    ``cell_holding``."""
    notebook = copy.deepcopy(read_notebook(name))
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"
             and cell_holding in "".join(cell["source"])]
    assert len(cells) == 1, f"{len(cells)} code cells of {name} hold {cell_holding!r}"
    source = "".join(cells[0]["source"])
    assert old in source, f"the cell holding {cell_holding!r} does not hold {old!r}"
    cells[0]["source"] = source.replace(old, new, 1)
    return notebook


def problems_of(name: str, notebook: Mapping[str, Any]) -> list[str]:
    return notebook_problems(notebook, example_text(name), HELD[name][1])


def test_a_changed_definition_is_reported_on_every_run():
    name = "sphere_settling_tube.ipynb"
    changed = planted(name, "def sphere_settling_tube", "spacing=um(SPACING_UM)", "spacing=um(2 * SPACING_UM)")
    assert [problem for problem in problems_of(name, changed) if "sphere_settling_tube is not" in problem]
    commented = planted(name, "SPACING_UM, MARGIN_UM =", "#: The drag's samples", "#: The drag's points")
    assert [problem for problem in problems_of(name, commented) if "SPACING_UM is not" in problem], \
        "a changed comment above a definition passes"
    retuned = planted("tutorial_a.ipynb", "def script_a", "amplitude\": mT(1.2)", "amplitude\": mT(1.3)")
    assert [problem for problem in problems_of("tutorial_a.ipynb", retuned) if "script_a is not" in problem]


def test_a_missing_definition_and_a_stale_exception_are_reported_on_every_run():
    name = "tutorial_a.ipynb"
    notebook = copy.deepcopy(read_notebook(name))
    notebook["cells"] = [cell for cell in notebook["cells"] if "FL9 = Screw(" not in "".join(cell["source"])]
    assert "the example's FL9 is in no cell tagged 'example'" in problems_of(name, notebook)
    stale = notebook_problems(read_notebook(name), example_text(name), {**HELD[name][1], "gone": "a removed arm"})
    assert stale == ["leaves out gone, which the example does not define: remove it from HELD"]


def test_a_rebinding_an_import_a_statement_and_an_output_are_reported_on_every_run():
    name = "sphere_settling_tube.ipynb"
    rebound = planted(name, "written = run()", "written = run()", "SPACING_UM = 5.0\nwritten = run()")
    assert [problem for problem in problems_of(name, rebound) if "binds SPACING_UM" in problem]
    other = planted(name, "from mime.interactions import", "from mime.interactions import", "from mime.things import")
    assert [problem for problem in problems_of(name, other) if "imports Contact from mime.things" in problem]
    statement = planted(name, "def stokes_speed", "def stokes_speed", "print(STEEL)\n\n\ndef stokes_speed")
    assert [problem for problem in problems_of(name, statement) if "neither a definition nor an import" in problem]
    ran = copy.deepcopy(read_notebook(name))
    next(cell for cell in ran["cells"] if cell["cell_type"] == "code")["execution_count"] = 1
    assert [problem for problem in problems_of(name, ran) if "committed with outputs" in problem]
