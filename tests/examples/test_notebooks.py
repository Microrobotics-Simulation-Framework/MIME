"""The notebooks of ``docs/notebooks/`` still work against this checkout's MIME, are saved without outputs, and every link
that opens one in Colab names the one branch ``scripts/colab_links.py`` sets.

Each notebook's code cells before the one tagged ``run`` declare its experiment; the test runs them (the install line
left out) and builds the experiment. Tutorial A's run cell is cheap, so it is run too, for 10 ms of simulated time in
place of the notebook's 5 s: that exercises what the notebooks read after a run (the observation's series, the
history, the recording the view draws). The settling ball's run is the example's, which
``tests/examples/test_sphere_settling_tube.py`` runs.

**What this bounds.** That a newcomer's notebook builds on this checkout's MIME, and A's runs; not that the
notebook runs on the release its install line pins, which is checked by running it against that release before it is
merged. The control: the notebook with its drag's weights left out is refused at the build.

**A cell run twice** (``mime.experiment._redeclared``), in a notebook as IPython marks one: every cell after the one
that makes the experiment, run again as it is, declares the same experiment, each declaration kept once with a note;
A's cells with a model swapped as their comments say replace the drag and the wall; A's field cell run again with its
frequency changed is refused at the build, saying a cell has probably run twice, since the torque names the field. The
notebooks still pin the released MIME, which refuses a cell run twice, so their own words about it change with the
release that ships this."""

import builtins
import importlib.util
import json
import re
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import tube_asset

from mime._notebook import RAN_TWICE, RUN_AGAIN
from mime.build import BuildRefused

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = ROOT / "docs" / "notebooks"
RUN_TAG = "run"
NOTEBOOK_NAMES = ("sphere_settling_tube.ipynb", "tutorial_a.ipynb")
#: A's run time in the test, in its settings' unit (seconds), in place of the notebook's.
SHORT_RUN_S = 0.01


def colab_links():
    spec = importlib.util.spec_from_file_location("colab_links", ROOT / "scripts" / "colab_links.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_notebook(name: str) -> dict[str, Any]:
    return json.loads((NOTEBOOKS / name).read_text(encoding="utf-8"))


def source_of(cell: Mapping[str, Any]) -> str:
    """The cell's Python, its shell and magic lines (``!``, ``%``) left out."""
    source = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
    return "\n".join(line for line in source.split("\n") if not line.lstrip().startswith(("!", "%")))


def code_cells(notebook: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


def output_problems(notebook: Mapping[str, Any]) -> list[str]:
    """Each code cell saved with outputs or an execution count."""
    return [f"code cell {number} is saved with outputs: clear them" for number, cell in
            enumerate(code_cells(notebook), 1) if cell.get("outputs") or cell.get("execution_count") is not None]


def split_at_run(notebook: Mapping[str, Any]) -> tuple[list[str], str]:
    """The code cells before the one tagged ``run``, and that cell."""
    cells = code_cells(notebook)
    tagged = [index for index, cell in enumerate(cells) if RUN_TAG in cell.get("metadata", {}).get("tags", [])]
    assert len(tagged) == 1, f"{len(tagged)} code cells are tagged {RUN_TAG!r}; one is the run"
    return [source_of(cell) for cell in cells[:tagged[0]]], source_of(cells[tagged[0]])


def declared(setup: list[str]) -> dict[str, Any]:
    """The namespace the setup cells leave: the experiment and the settings."""
    namespace: dict[str, Any] = {}
    for number, source in enumerate(setup, 1):
        exec(compile(source, f"<code cell {number}>", "exec"), namespace)
    return namespace


@pytest.mark.parametrize("name", NOTEBOOK_NAMES)
def test_the_notebook_is_saved_without_outputs(name):
    assert output_problems(read_notebook(name)) == []


def test_every_notebook_is_tested():
    assert sorted(path.name for path in NOTEBOOKS.glob("*.ipynb")) == sorted(NOTEBOOK_NAMES)


def test_every_colab_link_names_the_one_branch_and_a_notebook_there():
    links = colab_links()
    assert links.problems() == []
    held = [line for path in links.files() for line in path.read_text(encoding="utf-8").splitlines()
            if links.LINK.search(line)]
    assert len(held) >= 2 * len(NOTEBOOK_NAMES), "too few Colab links found: the scan could not fail"


def test_a_planted_output_and_a_link_to_another_branch_or_notebook_are_reported(tmp_path):
    notebook = read_notebook("tutorial_a.ipynb")
    code_cells(notebook)[0]["execution_count"] = 1
    assert output_problems(notebook) == ["code cell 1 is saved with outputs: clear them"]
    links = colab_links()
    (tmp_path / "docs" / "notebooks").mkdir(parents=True)
    (tmp_path / "docs" / "notebooks" / "tutorial_a.ipynb").write_text("{}", encoding="utf-8")
    (tmp_path / "README.md").write_text(f"[a]({links.link('tutorial_a.ipynb')}) [b]({links.link('gone.ipynb')})\n"
                                        f"[c]({links.link('tutorial_a.ipynb', 'release/v9')})\n", encoding="utf-8")
    assert len(links.problems(tmp_path)) == 2
    assert links.rewrite(tmp_path) == [tmp_path / "README.md"]
    assert [problem for problem in links.problems(tmp_path) if "branch" in problem] == []


def test_the_settling_balls_experiment_builds():
    tube_asset.path()  # stored by CI; a missing table fails here by name, with the command that stores it
    setup, _ = split_at_run(read_notebook("sphere_settling_tube.ipynb"))
    built = declared(setup)["experiment"].build()
    assert built.report.text().startswith("build report: experiment")


def test_tutorial_as_experiment_builds_and_its_run_cell_runs(tmp_path, monkeypatch):
    setup, run = split_at_run(read_notebook("tutorial_a.ipynb"))
    namespace = declared(setup)
    assert "run_time" in namespace, "the settings cell sets no run_time, which the test shortens"
    namespace["run_time"] = SHORT_RUN_S
    monkeypatch.chdir(tmp_path)
    with pytest.warns(UserWarning, match="Reynolds number"):
        exec(compile(run, "<the run cell>", "exec"), namespace)
    written = namespace["run"]
    along = np.asarray(written.result.history["robot"]["pose"])[:, 0]
    assert len(along) == round(SHORT_RUN_S / 0.5e-3) and np.isfinite(along).all()
    assert written.recording is not None and written.recording.is_file()


def test_control_a_notebook_whose_drag_lacks_its_weights_is_refused():
    setup, _ = split_at_run(read_notebook("tutorial_a.ipynb"))
    drag = 'Drag.Surrogate(weights=asset("dejongh_mlp_v2"))'
    assert sum(drag in source for source in setup) == 1, "the control's cell is gone: plant it elsewhere"
    namespace = declared([source.replace(drag, "Drag.Surrogate()") for source in setup])
    with pytest.raises(BuildRefused, match="weights"):
        namespace["experiment"].build()


@pytest.fixture
def notebook(monkeypatch):
    """IPython's mark, as IPython leaves it among the builtins while it runs a cell."""
    monkeypatch.setattr(builtins, "get_ipython", lambda: None, raising=False)


def after_the_start(setup: list[str]) -> list[str]:
    """The setup cells after the one that makes the experiment: those a reader may run again on their own."""
    starts = [index for index, source in enumerate(setup) if "= Experiment(" in source]
    assert len(starts) == 1, "no one cell makes the experiment"
    return setup[starts[0] + 1:]


def run_again(sources: list[str], namespace: dict[str, Any]) -> list[str]:
    """Each cell of ``sources`` run in ``namespace``, as a reader runs it again; the notes they gave."""
    with pytest.warns(UserWarning) as noted:
        for number, source in enumerate(sources, 1):
            exec(compile(source, f"<code cell {number}, run again>", "exec"), namespace)
    return [str(note.message) for note in noted]


def cell_holding(setup: list[str], text: str) -> str:
    held = [source for source in setup if text in source]
    assert len(held) == 1, f"{len(held)} cells hold {text!r}"
    return held[0]


def swapped(source: str, first: str) -> str:
    """The cell with the model swapped as its comment says: a # before the line that begins ``first``, and the
    commented-out line of the other model uncommented."""
    lines = source.split("\n")
    chosen = [index for index, line in enumerate(lines) if line.startswith(first)]
    other = [index for index, line in enumerate(lines) if line.startswith("# experiment.attach(")]
    assert len(chosen) == 1 and len(other) == 1, "the cell's swap is no longer one line for one line"
    lines[chosen[0]] = f"# {lines[chosen[0]]}"
    lines[other[0]] = lines[other[0]].removeprefix("# ")
    return "\n".join(lines)


@pytest.mark.parametrize("name", NOTEBOOK_NAMES)
def test_every_cell_after_the_start_run_again_in_a_notebook_declares_the_same_experiment(name, notebook):
    setup, _ = split_at_run(read_notebook(name))
    once = declared(setup)["experiment"].document()
    namespace = declared(setup)
    notes = run_again(after_the_start(setup), namespace)
    held = vars(namespace["experiment"])["_declarations"]
    assert len(notes) == len(held) and all(note.endswith("again, the same as before: it is kept once")
                                           for note in notes), notes
    assert namespace["experiment"].document() == once


def test_tutorial_as_experiment_with_every_cell_run_again_builds_in_a_notebook_and_in_a_script(monkeypatch):
    setup, run = split_at_run(read_notebook("tutorial_a.ipynb"))
    observe = run.split("\n")[0]
    assert observe.startswith("swimming = experiment.observe("), "the run cell no longer begins with its observation"
    for place in ("script", "notebook"):
        if place == "notebook":
            monkeypatch.setattr(builtins, "get_ipython", lambda: None, raising=False)
        namespace = declared([*setup, observe])
        run_again([*after_the_start(setup), observe], namespace)
        namespace["experiment"].plan()


@pytest.mark.parametrize(("first", "law", "replaced"), [
    ("experiment.attach(Drag.Surrogate(", "drag:robot", ("drag", "mrs")),
    ("experiment.attach(SimpleWall.Projection(", "contact:robot", ("contact", "0.4.0")),
])
def test_tutorial_as_swap_lines_run_in_a_notebook_replace_the_earlier_model(first, law, replaced, notebook):
    setup, _ = split_at_run(read_notebook("tutorial_a.ipynb"))
    namespace = declared(setup)
    notes = run_again([swapped(cell_holding(setup, first), first)], namespace)
    assert f"exp.attach(...) declares {law} again, not the same as before: in a notebook the last one run wins, so " \
           f"it replaces the earlier one" in notes
    held = [entry for entry in namespace["experiment"].document()["interactions"] if entry["id"] == law]
    family, which = replaced
    assert len(held) == 1 and held[0]["name"] == family
    assert which in (held[0].get("settings", {}).get("rung"), held[0]["version"])


@pytest.mark.parametrize("place", ["script", "notebook"])
def test_tutorial_as_field_cell_run_again_changed_is_refused_in_a_notebook_saying_a_cell_ran_twice(place, request):
    """The torque names the field, so a field changed in a notebook is not put in place of the earlier one: both are
    kept and refused, in a notebook's words there and in a script's words in a script."""
    if place == "notebook":
        request.getfixturevalue("notebook")
    setup, _ = split_at_run(read_notebook("tutorial_a.ipynb"))
    namespace = declared(setup)
    namespace["field_turns_per_second"] = 20.0
    with warnings.catch_warnings(record=True) as noted:
        warnings.simplefilter("always")
        exec(compile(cell_holding(setup, "experiment.add_apparatus("), "<the field cell, changed>", "exec"), namespace)
    notes = [str(note.message) for note in noted]
    kept = [note for note in notes if re.search(r"declares the apparatus 'field' again, not the same as before, and "
                                                r"magnetic\.torque:robot names the earlier one: .* so both are kept and "
                                                r"the build will refuse them", note)]
    assert len(kept) == (1 if place == "notebook" else 0), notes
    with pytest.raises(BuildRefused) as refused:
        namespace["experiment"].plan()
    held = [refusal for refusal in refused.value.refusals if refusal.code == "thing.duplicate"]
    assert held
    if place == "notebook":
        assert all(refusal.message.endswith(RAN_TWICE) and refusal.action == RUN_AGAIN for refusal in held)
    else:
        assert all(RAN_TWICE not in refusal.message and refusal.action != RUN_AGAIN for refusal in held)
