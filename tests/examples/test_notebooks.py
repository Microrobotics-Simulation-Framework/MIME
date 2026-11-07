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
A's cells with a model swapped as their comments say replace the drag and the wall; A's field cell run again with a new
strength replaces the apparatus, the torque law that names it follows it, and the run records the new strength's torque
(the same as a notebook that declared it from the top, and not the old); the tube's cell and the robot's likewise. In a
script the same field cell is still refused at the build. The notebooks still pin the released MIME, which refuses a
cell run twice, so their own words about it change with the release that ships this."""

import builtins
import importlib.util
import json
import os
import re
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import tube_asset

import mime._lineage as lineage
import mime.experiment._redeclared as redeclared
from mime._notebook import RAN_TWICE, RUN_AGAIN
from mime.build import BuildRefused
from mime.errors import MimeError
from mime.units import ms

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
    ("experiment.attach(SimpleWall.Projection(", "contact:robot", ("contact", "0.5.0")),
])
def test_tutorial_as_swap_lines_run_in_a_notebook_replace_the_earlier_model(first, law, replaced, notebook):
    setup, _ = split_at_run(read_notebook("tutorial_a.ipynb"))
    namespace = declared(setup)
    notes = run_again([swapped(cell_holding(setup, first), first)], namespace)
    assert f"exp.attach(...) declares {law} again, not the same as before: in a notebook or IPython the last one run wins, so " \
           f"it replaces the earlier one" in notes
    held = [entry for entry in namespace["experiment"].document()["interactions"] if entry["id"] == law]
    family, which = replaced
    assert len(held) == 1 and held[0]["name"] == family
    assert which in (held[0].get("settings", {}).get("rung"), held[0]["version"])


def recorded_torque(namespace: dict[str, Any], run: str, folder: Path) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """The run cell's short run (the notebook's, with ``run_time`` shortened) in ``folder``: the torque on the robot and
    the drive's amplitude it recorded, the two that depend on the field's strength."""
    namespace["run_time"] = SHORT_RUN_S
    folder.mkdir()
    previous = Path.cwd()
    os.chdir(folder)
    try:
        with pytest.warns(UserWarning, match="Reynolds number"):
            exec(compile(run, "<the run cell>", "exec"), namespace)
        history = namespace["run"].result.history
        return (np.asarray(history["magnetic.torque:robot"]["wrench"]),
                np.asarray(history["drive:field.amplitude"]["value"]))
    finally:
        os.chdir(previous)


def natural_edit_problems(folder: Path) -> list[str]:
    """The natural edit: change ``field_strength``, run the field cell again, then the run cell. The apparatus is
    replaced; the note names the law that follows it; the run records the new strength's torque, the same as a notebook
    that declared 2.4 from the top, and not the old (the field's amplitude is 2.4 mT in the run's record)."""
    setup, run = split_at_run(read_notebook("tutorial_a.ipynb"))
    old = recorded_torque(declared(setup), run, folder / "old")
    namespace = declared(setup)
    namespace["field_strength"] = 2.4
    notes = run_again([cell_holding(setup, "experiment.add_apparatus(")], namespace)
    changed = [note for note in notes if "declares the apparatus 'field' again, not the same as before" in note]
    problems: list[str] = []
    if not (len(changed) == 1 and changed[0].endswith("what holds its name now finds the new one: the law "
                                                       "magnetic.torque:robot")):
        problems.append(f"the note is {notes}")
    namespace["experiment"].plan()
    new = recorded_torque(namespace, run, folder / "new")
    from_the_top = declared([source.replace("field_strength = 1.2 ", "field_strength = 2.4 ") for source in setup])
    assert from_the_top["field_strength"] == 2.4
    fresh = recorded_torque(from_the_top, run, folder / "fresh")
    if not (np.allclose(new[1], 2.4e-3) and np.allclose(old[1], 1.2e-3)):
        problems.append("the run's recorded amplitude is not 2.4 mT after the edit and 1.2 mT before")
    if not all(np.array_equal(a, b) for a, b in zip(new, fresh, strict=True)):
        problems.append("the run does not record what a notebook that declared 2.4 from the top records")
    if np.array_equal(new[0], old[0]):
        problems.append("the run records the old strength's torque")
    return problems


def test_tutorial_as_field_cell_run_again_with_a_new_strength_in_a_notebook_runs_with_the_new_one(notebook, tmp_path):
    assert natural_edit_problems(tmp_path) == []


def test_tutorial_as_field_cell_run_again_under_a_filter_that_ignores_warnings_still_shows_in_the_plan_and_report(notebook):
    """A notebook whose first cell silences warnings: the replacement and the law that follows it are on the plan and in
    the build report, and the saved form is the one an experiment declared with the new strength holds."""
    setup, _ = split_at_run(read_notebook("tutorial_a.ipynb"))
    namespace = declared(setup)
    namespace["field_strength"] = 2.4
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(compile(cell_holding(setup, "experiment.add_apparatus("), "<the field cell, run again>", "exec"), namespace)
    experiment = namespace["experiment"]
    (note,) = experiment.plan().replaced
    assert "declares the apparatus 'field' again" in note and note.endswith("the law magnetic.torque:robot")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = experiment.build().report
    assert note in report.text()
    from_the_top = declared([source.replace("field_strength = 1.2 ", "field_strength = 2.4 ") for source in setup])
    assert experiment.document() == from_the_top["experiment"].document()


def test_control_a_re_bind_that_keeps_the_old_apparatus_fails_the_natural_edit_every_run(notebook, tmp_path, monkeypatch):
    """The note says it replaced; the earlier apparatus stays, so the run keeps its parameters under the new name."""
    def keeps_the_old_one(held, earlier, declaration, known, name, replaced, why):
        warnings.warn(f"{known.call} declares {known.noun} again, not the same as before: in a notebook or IPython the "
                      f"last one run wins, so it replaces the earlier one", UserWarning)

    monkeypatch.setattr(redeclared, "_replace_thing", keeps_the_old_one)
    assert natural_edit_problems(tmp_path)


def test_tutorial_as_field_cell_run_again_changed_is_refused_in_a_script_at_the_build():
    """The script path is unchanged: both apparatus are kept, with no note, and the build refuses the two in a script's
    words."""
    setup, _ = split_at_run(read_notebook("tutorial_a.ipynb"))
    namespace = declared(setup)
    namespace["field_strength"] = 2.4
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        exec(compile(cell_holding(setup, "experiment.add_apparatus("), "<the field cell, changed>", "exec"), namespace)
    assert not [note for note in caught if "not the same as before" in str(note.message)]
    with pytest.raises(BuildRefused) as refused:
        namespace["experiment"].plan()
    held = [refusal for refusal in refused.value.refusals if refusal.code == "thing.duplicate"]
    assert held and all(RAN_TWICE not in refusal.message and refusal.action != RUN_AGAIN for refusal in held)


@pytest.mark.parametrize(("cell", "text", "changed", "thing", "holders"), [
    ("experiment.add_structure(", "mm(3.175)", "mm(3.2)", "the structure 'tube'",
     ("the medium 'water'", "the apparatus 'field'", "the law contact:robot", "the observation")),
    ("experiment.add_body(", "mm((0.0, 0.0, -1.0))", "mm((0.0, 0.0, -0.9))", "the body 'robot'",
     ("the value robot.mass", "the law magnetic.torque:robot", "the law drag:robot", "the law gravity:robot")),
], ids=["the tube", "the robot"])
def test_tutorial_as_a_thing_cell_run_again_changed_in_a_notebook_replaces_the_thing_and_still_plans(
        cell, text, changed, thing, holders, notebook):
    """The tube's cell (its radius) and the robot's (where it starts) run again with a number changed: each is replaced,
    the one note names what follows it, and the experiment still plans. The observation is not made yet in the tube's
    case: the run cell makes it, so it is not listed."""
    setup, _ = split_at_run(read_notebook("tutorial_a.ipynb"))
    source = cell_holding(setup, cell)
    assert text in source
    namespace = declared(setup)
    notes = run_again([source.replace(text, changed)], namespace)
    (replaced,) = [note for note in notes if f"declares {thing} again, not the same as before: in a notebook or IPython "
                                             f"the last one run wins, so it replaces the earlier one" in note]
    assert all(holder in replaced for holder in holders if holder != "the observation"), replaced
    namespace["experiment"].plan()


# -- a stale build, result and view: the cell run again with the one that builds not run again ----------------------------

def ran_and_edited(folder: Path, edit: str) -> dict[str, Any]:
    """Tutorial A's setup and run cell (10 ms of simulated time), then ``edit``: ``"field"`` runs the field cell again with
    a new strength (a replacement), ``"gravity"`` removes the gravity law."""
    setup, run = split_at_run(read_notebook("tutorial_a.ipynb"))
    namespace = declared(setup)
    namespace["run_time"] = SHORT_RUN_S
    os.chdir(folder)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(compile(run, "<the run cell>", "exec"), namespace)
        namespace["page_before"] = namespace["viz"].display("robot_run")
        if edit == "field":
            namespace["field_strength"] = 2.4
            exec(compile(cell_holding(setup, "experiment.add_apparatus("), "<the field cell, run again>", "exec"),
                 namespace)
        else:
            namespace["experiment"].remove("gravity:robot")
    return namespace


SLOW_RUN = "run = built.run_to('robot_run_slow', s(0.005), record=[swimming], history=True, stage_every=ms(1.0))"


def stale_problems(folder: Path, place: str) -> list[str]:
    """After the edit, the old ``built`` is refused in a notebook and warned about in a script (the tutorial's own
    commented ``slow_run`` line); the result and the view of the first run say they are stale; a fresh build clears all
    of them."""
    previous = Path.cwd()
    try:
        namespace = ran_and_edited(folder, "field" if place == "notebook" else "gravity")
        problems: list[str] = []
        first = namespace["run"]
        before = namespace["page_before"]  # made when the build was current: only its banner can say it is stale now
        if "STALE" in before.html or before.stale is None or "STALE" not in before._repr_html_() \
                or "STALE" not in repr(before):
            problems.append("a view made before the change does not say it is stale once the change is made")
        what = "the replacement of the apparatus 'field'" if place == "notebook" else \
            "the removal of the law gravity:robot"
        if what not in (namespace["built"].stale or ""):
            problems.append(f"the old build does not say {what}: {namespace['built'].stale}")
        if "STALE" not in repr(first.result) or what not in repr(first.result):
            problems.append("the first run's result, displayed, does not say it is stale")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            page = namespace["viz"].display("robot_run")
        shown = [str(note.message) for note in caught if str(note.message).startswith("this view predates")]
        if not (len(shown) == 1 and what in shown[0] and page.stale is not None and "STALE" in page._repr_html_()):
            problems.append(f"the view does not say it is stale: {shown}")
        if "STALE" not in first.view().html or "STALE" not in repr(first.view()):
            problems.append("the view made from the run does not say it is stale")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                exec(compile(SLOW_RUN, "<the slow run>", "exec"), namespace)
                ran = True
            except MimeError as refused:
                ran = False
                if refused.refusals[0].code != "run.stale" or what not in refused.refusals[0].message:
                    problems.append(f"the refusal is {refused.refusals[0]}")
        if place == "notebook" and ran:
            problems.append("the stale build ran in a notebook")
        if place == "script" and not (ran and [n for n in caught if str(n.message).startswith("this build predates")]):
            problems.append("the stale build did not run with its warning in a script")
        namespace["built"] = namespace["experiment"].build()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exec(compile(SLOW_RUN.replace("robot_run_slow", "robot_run_fresh"), "<the fresh run>", "exec"), namespace)
        fresh = namespace["run"]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            page = namespace["viz"].display("robot_run_fresh")
        if fresh.result.stale is not None or "STALE" in repr(fresh.result) or page.stale is not None \
                or [n for n in caught if "predates" in str(n.message)]:
            problems.append("a fresh build, its result and its view are stale")
        return problems
    finally:
        os.chdir(previous)


def test_tutorial_as_old_built_after_the_field_cell_run_again_is_refused_and_a_fresh_build_clears_it(notebook, tmp_path):
    assert stale_problems(tmp_path, "notebook") == []


def test_tutorial_as_old_built_after_a_removal_runs_with_a_warning_in_a_script(tmp_path):
    assert stale_problems(tmp_path, "script") == []


def test_control_a_marker_that_never_raises_fails_the_tutorials_stale_check_every_run(notebook, tmp_path, monkeypatch):
    monkeypatch.setattr(lineage.Origin, "since", lambda self: ())
    assert stale_problems(tmp_path, "notebook")


def test_control_a_marker_that_raises_on_an_unchanged_notebook_fails_the_tutorials_stale_check_every_run(
        notebook, tmp_path, monkeypatch):
    monkeypatch.setattr(lineage.Origin, "since",
                        lambda self: (lineage.Change("replacement", "the apparatus 'field'", "a note"),))
    previous = Path.cwd()
    try:
        setup, run = split_at_run(read_notebook("tutorial_a.ipynb"))
        namespace = declared(setup)
        namespace["run_time"] = SHORT_RUN_S
        os.chdir(tmp_path)
        with pytest.raises(MimeError, match="predates"):  # the plant refuses the unchanged tutorial's own run
            exec(compile(run, "<the run cell>", "exec"), namespace)
            namespace["built"].run(ms(1))
    finally:
        os.chdir(previous)
