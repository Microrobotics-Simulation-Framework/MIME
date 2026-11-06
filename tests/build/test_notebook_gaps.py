"""The notebook's three gaps after re-declaration and re-binding (``mime.experiment._removed``, ``mime._lineage``).

**``exp.remove``** takes a declaration out, in a script and in a notebook alike: a thing, a law, an observation, a declared
value, a source, the controller. It is refused while another declaration still names it, the refusal listing each with the
cell or line that made it, and it is recorded like a replacement (``plan().removed``, the build report) whatever the
warnings filter. Removing and declaring again under the name is how a thing changes kind.

**Laws and observations a source or the controller names** are replaced in a notebook as things are (``test_rebind.py``):
the source or the controller follows by the id, the one note lists it, and the graph built is the graph a notebook that
declared the change from the top builds. A script still refuses the two.

**A stale result says so.** A build, the result of its run and the view of that run remember the declarations they came
from. After a replacement or a removal a build is refused in a notebook (the trap is a cell run again with the one that
builds not run again) and warned about in a script (a script that changes its experiment after a build did it on
purpose); its report, a result shown and a view shown say what changed since. A fresh build clears it; a change put back
is not stale; a declaration only added is not a change to what was built.

**The controls.** Each check is held against a plant, and the plant must fail it every run (the experiments are fixed):
a marker that never raises, one that raises on an unchanged experiment, a re-bind that keeps the old observation, a removal
that ignores what names the thing, one that is silent."""

from __future__ import annotations

import builtins
import inspect
import re
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from control_cases import Watcher, spun_magnet
from test_rebind import apparatus, declarations, experiment, noted

import mime._lineage as lineage
import mime.experiment._redeclared as redeclared
import mime.experiment._removed as removed
from mime import Experiment
from mime._notebook import in_notebook, where_called
from mime.build import BuildRefused, Refusal
from mime.errors import MimeError
from mime.observations import SwimSpeed
from mime.scene import Y
from mime.units import mm, ms


@pytest.fixture
def notebook(monkeypatch):
    """IPython's mark, as IPython leaves it among the builtins while it runs a cell."""
    monkeypatch.setattr(builtins, "get_ipython", lambda: None, raising=False)
    assert in_notebook()


@pytest.fixture(params=["script", "notebook"])
def place(request):
    if request.param == "notebook":
        request.getfixturevalue("notebook")
    else:
        assert not in_notebook(), "this process is IPython: the script's path cannot be told from the notebook's"
    return request.param


def _calls(refusal: Refusal) -> list[str]:
    """The ``exp.remove(...)`` calls a ``remove.named`` refusal's action gives, in its order."""
    return re.findall(r"exp\.remove\((?!\.\.\.)[^)]*\)", refusal.action)


def refusal_of(call: Callable[[], object]) -> Refusal:
    try:
        call()
    except MimeError as refused:
        (refusal,) = getattr(refused, "refusals")
        return refusal
    raise AssertionError("the call was not refused")


def watched(along: Any = None) -> Experiment:
    """A motor alone, a body, an observation of its swim speed, and a controller that reads that speed."""
    kept = {} if along is None else {"along": along}
    return spun_magnet(controller=Watcher(), observe=True, reads={"pose": "umr.pose", "speed": "swim_speed:umr.speed"},
                       **kept)


def spare(made: Any) -> None:
    """An apparatus nothing names."""
    made.exp.add_apparatus(apparatus(made.tube, name="spare"))


# -- remove: each kind of declaration -----------------------------------------------------------------------------------

def a_spare_thing(made: Any) -> tuple[Experiment, Callable[[Experiment], object]]:
    spare(made)
    return made.exp, lambda exp: exp.remove("spare")


def a_handle(made: Any) -> tuple[Experiment, Callable[[Experiment], object]]:
    handle = made.exp.add_apparatus(apparatus(made.tube, name="spare"))
    return made.exp, lambda exp: exp.remove(handle)


def a_law(made: Any) -> tuple[Experiment, Callable[[Experiment], object]]:
    return made.exp, lambda exp: exp.remove("gravity:robot")


def an_observation(made: Any) -> tuple[Experiment, Callable[[Experiment], object]]:
    return made.exp, lambda exp: exp.remove("swim_speed:robot")


def a_value(made: Any) -> tuple[Experiment, Callable[[Experiment], object]]:
    return made.exp, lambda exp: exp.remove("robot.inertia")


def a_source(made: Any) -> tuple[Experiment, Callable[[Experiment], object]]:
    made.exp.source("tube.inner_radius", reason="a datasheet")
    return made.exp, lambda exp: exp.remove("tube.inner_radius")


#: What each case removes, as the note names it.
CASES = {
    "a thing": (a_spare_thing, "the apparatus 'spare'"),
    "a thing, by its handle": (a_handle, "the apparatus 'spare'"),
    "a law": (a_law, "the law gravity:robot"),
    "an observation": (an_observation, "the observation swim_speed:robot"),
    "a declared value": (a_value, "the value robot.inertia"),
    "a source": (a_source, "the source of tube.inner_radius"),
}


def removal_problems(case: str, *, filter: str = "always") -> list[str]:
    """The removal of ``case``: one declaration fewer, the one note (``exp.remove(...) removes ...; nothing else names
    it``), and the plan and the report hold it whatever the warnings filter."""
    make, noun = CASES[case]
    made = experiment()
    exp, remove = make(made)
    count = len(declarations(exp))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter(filter)
        remove(exp)
    problems: list[str] = []
    if len(declarations(exp)) != count - 1:
        problems.append(f"{len(declarations(exp))} declarations held, not {count - 1}")
    expected = f"exp.remove(...) removes {noun}; nothing else names it"
    if filter == "always" and [str(note.message) for note in caught] != [expected]:
        problems.append(f"the warning is {[str(note.message) for note in caught]}, not {[expected]}")
    try:
        recorded = exp.plan().removed
    except BuildRefused as refused:  # the value removed is one the build needs: the refused report carries the note
        report = refused.report
        recorded = report.removed if report is not None else ()
    if recorded != (expected,):
        problems.append(f"the plan records {recorded}, not {(expected,)}")
    return problems


@pytest.mark.parametrize("case", CASES)
def test_each_kind_is_removed_with_one_note_and_the_plan_records_it(case, place):
    assert removal_problems(case) == []


@pytest.mark.parametrize("case", CASES)
def test_a_removal_under_a_filter_that_ignores_warnings_is_still_recorded(case, place):
    assert removal_problems(case, filter="ignore") == []


def test_the_build_report_says_what_was_removed_in_this_session(place):
    made = experiment()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        made.exp.remove("gravity:robot")
        built = made.exp.build()
    assert built.report.removed == ("exp.remove(...) removes the law gravity:robot; nothing else names it",)
    assert "removed in this session (1; the saved form holds what was left):" in built.report.text(details=True)
    assert "removed in this session" in "\n".join(built.report.summary())


def test_the_controller_is_removed_by_kind_and_a_second_removal_is_refused(place):
    exp = watched()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exp.remove(kind="controller")
    with pytest.raises(BuildRefused) as refused:  # nothing drives the motor's angle now: the refused report keeps the note
        exp.plan()
    assert refused.value.report.removed == ("exp.remove(...) removes the controller; nothing else names it",)
    assert refusal_of(lambda: exp.remove(kind="controller")).code == "remove.unknown"


def test_a_removal_whose_warning_is_raised_as_an_error_leaves_the_experiment_as_it_was(place):
    made = experiment()
    spare(made)
    count = len(declarations(made.exp))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(UserWarning, match="removes the apparatus 'spare'"):
            made.exp.remove("spare")
    assert len(declarations(made.exp)) == count and made.exp.plan().removed == ()


def test_the_saved_form_holds_what_was_left_and_equals_an_experiment_declared_without_it(place):
    made, fresh = experiment(), experiment()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spare(made)
        made.exp.remove("spare")
    assert made.exp.document() == fresh.exp.document()


# -- remove: refused while something names it --------------------------------------------------------------------------

def named_refusal_problems() -> list[str]:
    """``tube`` is named by four declarations: removing it is refused, listing each, and nothing is removed."""
    made = experiment()
    count = len(declarations(made.exp))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            made.exp.remove("tube")
    except MimeError as refused:
        (refusal,) = getattr(refused, "refusals")
    else:
        return ["the structure was removed while four declarations name it"]
    problems = [] if refusal.code == "remove.named" else [f"the code is {refusal.code}"]
    for holder in ("the medium 'water'", "the apparatus 'field'", "the law contact:robot",
                   "the observation swim_speed:robot"):
        if holder not in refusal.message:
            problems.append(f"{holder} is not listed")
    if "4 declarations still name it" not in refusal.message or refusal.message.count("test_rebind.py, line") != 4:
        problems.append("the refusal does not list four declarations, each with its line")
    if len(declarations(made.exp)) != count or made.exp.plan().removed:
        problems.append("the refused removal changed the experiment")
    return problems


def test_a_thing_something_names_is_refused_listing_each_declaration_that_names_it_and_nothing_is_removed(place):
    assert named_refusal_problems() == []


def test_control_a_removal_that_ignores_what_names_the_thing_fails_the_refusal_check_every_run(monkeypatch):
    monkeypatch.setattr(removed, "referrers", lambda held, found: [])
    assert named_refusal_problems()


def test_the_refusal_names_the_line_that_made_each_declaration_in_a_script():
    made = experiment()
    line = inspect.currentframe().f_lineno + 1  # the next line
    made.exp.source("tube.inner_radius", reason="a datasheet")
    refusal = refusal_of(lambda: made.exp.remove("tube"))
    assert f"the source of tube.inner_radius ({__file__}, line {line})" in refusal.message
    assert "exp.remove('tube.inner_radius', kind='source')" in refusal.action


def test_the_refusal_names_the_cell_and_its_line_in_a_notebook(monkeypatch):
    shell = type("Shell", (), {"execution_count": 5})()
    monkeypatch.setattr(builtins, "get_ipython", lambda: shell, raising=False)
    made = experiment()
    cell = compile("\nexp.source('tube.inner_radius', reason='a datasheet')\n", "<ipython-input-5-0abc>", "exec")
    exec(cell, {"exp": made.exp})
    refusal = refusal_of(lambda: made.exp.remove("tube"))
    assert "the source of tube.inner_radius (cell [5], line 2)" in refusal.message
    assert where_called().startswith(__file__)  # called from a file, not a cell: a file and its line


def test_a_declaration_loaded_from_a_document_is_named_as_loaded(tmp_path):
    made = experiment()
    made.exp.save(tmp_path / "saved", steps=2)
    loaded = Experiment.load(tmp_path / "saved")
    assert "declared in the loaded document" in refusal_of(lambda: loaded.remove("tube")).message


def test_a_law_a_source_names_and_an_observation_the_controller_reads_are_refused():
    made = experiment()
    made.exp.source("contact:robot.margin", reason="a datasheet")
    law = refusal_of(lambda: made.exp.remove("contact:robot"))
    assert law.code == "remove.named" and "the source of contact:robot.margin" in law.message
    exp = watched()
    observation = refusal_of(lambda: exp.remove("swim_speed:umr"))
    assert observation.code == "remove.named" and "the controller" in observation.message
    assert exp.plan().removed == ()


def test_removing_what_names_it_first_lets_it_go_in_the_order_the_refusal_gives(place):
    """The refusal's own list is the way out: remove each declaration it names, then the thing."""
    made = experiment()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(40):
            refusal = None
            try:
                made.exp.remove("robot")
                break
            except MimeError as refused:
                (refusal,) = getattr(refused, "refusals")
            assert refusal.code == "remove.named"
            eval(_calls(refusal)[0], {"exp": made.exp})
    assert "robot" not in [getattr(d, "thing", None) and d.thing.name for d in declarations(made.exp)]


def test_the_wrong_name_the_wrong_kind_and_the_wrong_type_are_refused_with_the_candidates(place):
    made = experiment()
    unknown = refusal_of(lambda: made.exp.remove("tubee"))
    assert unknown.code == "remove.unknown" and "Did you mean 'tube'" in unknown.action
    assert refusal_of(lambda: made.exp.remove("tube", kind="law")).code == "remove.unknown"
    assert refusal_of(lambda: made.exp.remove("tube", kind="stuff")).code == "remove.unknown"
    assert refusal_of(lambda: made.exp.remove(kind="value")).code == "remove.unknown"
    assert refusal_of(lambda: made.exp.remove(3)).code == "call.wrong-type"  # type: ignore[arg-type]
    assert refusal_of(lambda: made.exp.remove(kind="controller")).code == "remove.unknown"


def test_a_path_that_is_a_value_and_a_source_is_removed_by_the_kind_that_says_which(place):
    made = experiment()
    made.exp.source("robot.inertia", reason="a datasheet")
    ambiguous = refusal_of(lambda: made.exp.remove("robot.inertia"))
    assert ambiguous.code == "remove.ambiguous" and "kind='value'" in ambiguous.action \
        and "kind='source'" in ambiguous.action
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        made.exp.remove("robot.inertia", kind="source")
    assert made.exp.plan().removed == ("exp.remove(...) removes the source of robot.inertia; nothing else names it",)
    assert [d for d in declarations(made.exp) if "inertia" in str(getattr(d, "path", ""))]  # the value is still held


# -- remove, then declare again: how a thing changes kind ----------------------------------------------------------------

def kind_change_problems() -> list[str]:
    """A body named ``robot`` becomes an apparatus named ``robot``: removed, then declared again."""
    made = experiment()
    exp = made.exp
    problems: list[str] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(40):
            try:
                exp.remove("robot")
                break
            except MimeError as refused:
                (refusal,) = getattr(refused, "refusals")
                eval(_calls(refusal)[0], {"exp": exp})
        exp.add_apparatus(apparatus(made.tube, name="robot"))
    kinds = [d.thing.kind.value for d in declarations(exp) if hasattr(d, "thing") and d.thing.name == "robot"]
    if kinds != ["apparatus"]:
        problems.append(f"'robot' is held as {kinds}, not as one apparatus")
    try:
        exp.plan()
    except BuildRefused as refused:
        codes = {r.code for r in refused.refusals}
        if "thing.duplicate" in codes:
            problems.append("the plan refuses two things under one name")
    return problems


def test_a_thing_removed_and_declared_again_under_the_name_changes_its_kind(place):
    assert kind_change_problems() == []


def test_a_body_declared_again_as_an_apparatus_without_removal_is_not_a_change_of_kind(notebook):
    """The control: with no ``remove``, the notebook keeps both and the build refuses the name held twice."""
    made = experiment()
    noted(lambda: made.exp.add_apparatus(apparatus(made.tube, name="robot")))
    with pytest.raises(BuildRefused) as refused:
        made.exp.plan()
    assert "thing.duplicate" in {r.code for r in refused.value.refusals}


# -- laws and observations a source or the controller names: replaced, and what names them follows -----------------------

def rebind_problems(plant: bool = False) -> list[str]:
    """A notebook declares an observation the controller reads, then declares it again along another direction: the
    graph built, its saved-form hashes and a short run are those of a notebook that declared the change from the top, and
    not those of the first."""
    exp = watched()
    before = exp.build()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        exp.observe(SwimSpeed("umr", along=Y))
    problems: list[str] = []
    notes = [str(note.message) for note in caught if "not the same as before" in str(note.message)]
    if not (len(notes) == 1 and notes[0].endswith("it replaces the earlier one; what holds its id now finds the new "
                                                  "one: the controller")):
        problems.append(f"the note is {notes}")
    fresh = watched(Y)
    built, wanted = exp.build(), fresh.build()
    if built.record.hashes() != wanted.record.hashes():
        problems.append("the saved-form hashes are not those of a notebook that declared the change from the top")
    if built.record.hashes() == before.record.hashes():
        problems.append("the saved-form hashes are those of the first declaration")
    if built.topology() != wanted.topology():
        problems.append("the built graph is not the fresh notebook's")
    ran, expected = built.run(ms(6), history=True), wanted.run(ms(6), history=True)
    for node in expected.history:
        for name in expected.history[node]:
            if not np.array_equal(np.asarray(ran.history[node][name]), np.asarray(expected.history[node][name])):
                problems.append(f"the run differs from the fresh notebook's at {node}.{name}")
    return problems


def test_an_observation_the_controller_reads_declared_again_builds_what_a_fresh_notebook_builds(notebook):
    assert rebind_problems() == []


def test_control_a_re_bind_that_keeps_the_old_observation_fails_the_check_every_run(notebook, monkeypatch):
    real = redeclared._replace

    def keeps_the_old_one(held, earlier, declaration, note, replaced):
        real(held, earlier, held[earlier[0]], note, replaced)  # says it replaced; puts the earlier one back

    monkeypatch.setattr(redeclared, "_replace", keeps_the_old_one)
    assert rebind_problems()


def law_rebind_problems(plant: bool = False) -> list[str]:
    """A source names the contact law's margin; the law is declared again with another margin in a notebook: the note
    lists the source, and the graph built, its hashes and a short run are those of a notebook that declared the change
    from the top."""
    from test_rebind import CentreContact

    made = experiment(sourced=True)
    before = made.exp.build()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        made.exp.attach(CentreContact.Projection(margin=mm(0.3)), body=made.robot, surface=made.tube.inner_surface)
    problems: list[str] = []
    notes = [str(note.message) for note in caught if "not the same as before" in str(note.message)]
    if not (len(notes) == 1 and notes[0].endswith("what holds its id now finds the new one: the source of "
                                                  "contact:robot.margin")):
        problems.append(f"the note is {notes}")
    built, wanted = made.exp.build(), experiment(sourced=True, margin=0.3).exp.build()
    if built.record.hashes() != wanted.record.hashes():
        problems.append("the saved-form hashes are not those of a notebook that declared 0.3 mm from the top")
    if built.record.hashes() == before.record.hashes():
        problems.append("the saved-form hashes are those of the first declaration")
    if built.topology() != wanted.topology():
        problems.append("the built graph is not the fresh notebook's")
    ran, expected = built.run(ms(5), history=True), wanted.run(ms(5), history=True)
    for node in expected.history:
        for name in expected.history[node]:
            if not np.array_equal(np.asarray(ran.history[node][name]), np.asarray(expected.history[node][name])):
                problems.append(f"the run differs from the fresh notebook's at {node}.{name}")
    return problems


def test_a_law_a_source_names_is_replaced_and_builds_what_a_fresh_notebook_builds(notebook):
    assert law_rebind_problems() == []


def test_control_a_re_bind_that_keeps_the_old_law_fails_the_law_check_every_run(notebook, monkeypatch):
    real = redeclared._replace
    monkeypatch.setattr(redeclared, "_replace", lambda held, earlier, declaration, note, replaced:
                        real(held, earlier, held[earlier[0]], note, replaced))
    assert law_rebind_problems()


def test_a_script_still_refuses_a_changed_law_or_observation_a_source_names():
    made = experiment()
    made.exp.source("swim_speed:robot.along", reason="a datasheet")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        made.exp.observe(SwimSpeed(made.robot, along=Y))
    with pytest.raises(BuildRefused) as refused:
        made.exp.plan()
    assert "id.duplicate" in {r.code for r in refused.value.refusals}


# -- a stale build, result and view -------------------------------------------------------------------------------------

def edit_the_field(made: Any, strength: float = 2.4) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        made.exp.add_apparatus(apparatus(made.tube, strength))


def remove_gravity(made: Any) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        made.exp.remove("gravity:robot")


#: The change each place makes: a script cannot replace (it would hold both and be refused), so it removes.
EDITS = {"script": (remove_gravity, "the removal of the law gravity:robot"),
         "notebook": (edit_the_field, "the replacement of the apparatus 'field'")}


def staleness_problems(place: str) -> list[str]:
    """After a change, the old build is refused in a notebook and warned about in a script; its report, its result
    and the view of its run say what changed; a fresh build clears all three."""
    edit, said_so = EDITS[place]
    made = experiment()
    built = made.exp.build()
    if built.stale is not None:
        return ["a build is stale before any change"]
    result = built.run(ms(2))
    problems: list[str] = []
    if result.stale is not None:
        problems.append("a result is stale before any change")
    edit(made)
    if built.stale is None or said_so not in built.stale:
        problems.append(f"the build does not say what changed: {built.stale}")
    if not built.report.text().splitlines()[1].startswith("STALE: this build predates"):
        problems.append("the report does not say so on its first lines")
    if "STALE" in str(built.report_summary()):
        problems.append("the report's summary for a client, written into run folders, carries the mark")
    if result.stale is None or "STALE" not in repr(result) or "STALE" not in result.text():
        problems.append("the result does not say it is stale")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            built.run(ms(2))
            ran = True
        except MimeError as refused:
            ran = False
            (refusal,) = getattr(refused, "refusals")
            if refusal.code != "run.stale" or "build the experiment again" not in refusal.action:
                problems.append(f"the refusal is {refusal.code}: {refusal.action}")
    said = [str(note.message) for note in caught if "predates" in str(note.message)]
    if place == "notebook" and ran:
        problems.append("a stale build ran in a notebook")
    if place == "script" and not (ran and len(said) == 1):
        problems.append(f"in a script the build should run with one warning: ran {ran}, said {said}")
    fresh = made.exp.build()
    again = fresh.run(ms(2))
    if fresh.stale is not None or again.stale is not None or "STALE" in repr(again) or "STALE" in fresh.report.text():
        problems.append("a fresh build is stale")
    return problems


def test_a_stale_build_is_refused_in_a_notebook_and_warned_about_in_a_script(place):
    assert staleness_problems(place) == []


@pytest.mark.parametrize("edit", [
    pytest.param(lambda made: made.exp.remove("gravity:robot"), id="a removed law"),
    pytest.param(lambda made: made.exp.remove("swim_speed:robot"), id="a removed observation"),
])
def test_a_removal_makes_a_build_stale_as_a_replacement_does(place, edit):
    made = experiment()
    built = made.exp.build()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        edit(made)
    assert built.stale is not None and "the removal of" in built.stale


def test_a_replacement_put_back_is_not_stale(notebook):
    made = experiment()
    built = made.exp.build()
    edit_the_field(made, 2.4)
    assert built.stale is not None
    edit_the_field(made, 1.2)  # the field as it was when built
    assert built.stale is None
    built.run(ms(2))  # runs: nothing it was built from differs


def test_a_cell_run_again_and_again_is_named_once(notebook):
    made = experiment()
    built = made.exp.build()
    for strength in (2.4, 3.6, 4.8):
        edit_the_field(made, strength)
    assert built.stale is not None and built.stale.count("the replacement of the apparatus 'field'") == 1


def test_a_declaration_only_added_is_not_a_change_to_what_was_built(place):
    made = experiment()
    built = made.exp.build()
    made.exp.add_apparatus(apparatus(made.tube, name="spare"))
    assert built.stale is None
    built.run(ms(2))


def test_control_a_marker_that_never_raises_fails_the_staleness_check_every_run(place, monkeypatch):
    monkeypatch.setattr(lineage.Origin, "since", lambda self: ())
    assert staleness_problems(place)


def test_control_a_marker_that_raises_on_an_unchanged_experiment_fails_the_check_every_run(place, monkeypatch):
    change = lineage.Change("replacement", "the apparatus 'field'", "a note")
    monkeypatch.setattr(lineage.Origin, "since", lambda self: (change,))
    assert staleness_problems(place) == ["a build is stale before any change"]


def test_a_loaded_experiments_build_is_never_stale_and_runs_in_a_notebook(notebook, tmp_path):
    made = experiment()
    made.exp.save(tmp_path / "saved", steps=2)
    built = Experiment.load(tmp_path / "saved").build()
    assert built.stale is None
    built.run(ms(2))


def test_a_run_folder_written_by_a_stale_build_is_never_given_its_mark(tmp_path, notebook):
    """The marks are on the objects in the session. A run folder holds the build's record and the report's summary, as
    for an experiment never changed."""
    made = experiment()
    built = made.exp.build()
    clean = built.run_to(tmp_path / "clean", ms(2))
    edit_the_field(made)
    assert "STALE" not in (tmp_path / "clean" / "report.json").read_text(encoding="utf-8")
    assert clean.result.stale is not None


# -- added after the first table of mutants: what it missed --------------------------------------------------------------

def test_a_thing_held_as_an_object_by_the_controller_is_refused(place):
    from test_rebind import held_object

    made = experiment()
    held_object(made)
    refusal = refusal_of(lambda: made.exp.remove("robot"))
    assert refusal.code == "remove.named" and "the controller" in refusal.message


def test_every_declaration_under_a_name_goes_with_it_in_a_script():
    """A script holds two things under one name until the build refuses them: one removal takes both, and says so."""
    made = experiment()
    made.exp.add_apparatus(apparatus(made.tube, 2.4, name="spare"))
    made.exp.add_apparatus(apparatus(made.tube, 3.6, name="spare"))
    count = len(declarations(made.exp))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        made.exp.remove("spare")
    assert len(declarations(made.exp)) == count - 2
    assert [str(note.message) for note in caught] == ["exp.remove(...) removes the apparatus 'spare' (2 declarations); "
                                                      "nothing else names it"]


def test_a_stale_build_is_refused_for_a_live_session_and_a_save_in_a_notebook_and_warned_about_in_a_script(place, tmp_path):
    edit, _ = EDITS[place]
    made = experiment()
    built = made.exp.build()
    edit(made)
    for call in (lambda: built.live(), lambda: built.save(tmp_path / "saved", steps=2)):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if place == "notebook":
                assert refusal_of(call).code == "run.stale"
            else:
                call()
                assert [note for note in caught if "this build predates" in str(note.message)]


def test_the_summary_for_a_client_is_not_marked_even_when_first_read_after_the_change(place):
    edit, _ = EDITS[place]
    made = experiment()
    built = made.exp.build()
    edit(made)
    assert built.stale is not None and "STALE" not in str(built.report_summary())


def test_a_build_names_only_the_changes_made_after_it(notebook):
    made = experiment()
    edit_the_field(made, 2.4)  # before the build: this build is of the new field
    built = made.exp.build()
    assert built.stale is None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        made.exp.remove("gravity:robot")
    assert built.stale is not None and "the removal of the law gravity:robot" in built.stale \
        and "replacement" not in built.stale
