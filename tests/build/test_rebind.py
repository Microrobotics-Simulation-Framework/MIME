"""A changed structure, medium, body or apparatus, declared again under its name and kind in a notebook, replaces the
earlier one, and what names it follows it to the new one, with one note that says which (``mime.experiment._redeclared``).

**What follows it.** A declaration holds a thing by its name (a law's binding, a handle, a typed path, an observation's
binding, the controller's reads and writes, a value's or a source's path, a medium's ``fills``, a setting that names a
direction), and the build resolves a name against the things the experiment holds. So the replacement is the whole
re-binding: there is no declaration to rewrite. What would not follow is a declaration that holds the earlier *object*
(its settings under the new name, the silent re-bind): it is kept beside the new one and refused at the build, as is a
name of another kind. A law the new thing cannot serve (a field removed from an apparatus) is refused by that law's own
plan at the build, and the note has named it.

**What each test shows** (the notebook is IPython's mark among the builtins; the script is this process as it is):
- the natural edit, in tutorial A's order (declare, attach, observe; the field cell run again with a new strength): the
  run records the new strength's torque, the same as an experiment declared with it from the start, and not the old;
- the same edit for each kind of thing, shown on what the plan builds from it;
- the note names every law, observation, value, source, controller and other thing that holds the name;
- the refusals: a name of another kind, a held object, a law the new thing cannot serve;
- a script still refuses; an identical declaration is still kept once; a saved experiment holds only the new one.

**The controls.** Each check is held against a re-bind planted wrong, and the plant must fail it on every run (the
experiments are fixed, so it does): one that keeps the old declaration, one that adds beside it, one that puts the new
one at the end, one that is silent, one whose note skips the observations, one whose note skips a medium's ``fills``,
one that replaces across kinds, one that does not look for a held object."""

from __future__ import annotations

import builtins
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from control_cases import Ticker
from script_a import CONDITIONS, REASON, WATER_RECORD, fl9

import mime.experiment._redeclared as redeclared
from mime import Apparatus, Body, Experiment, Medium, State, Structure
from mime._notebook import RAN_TWICE, RUN_AGAIN, in_notebook
from mime.build import BuildRefused, Refusal, ThingDeclaration
from mime.dynamics import RigidInertial
from mime.fields import UniformRotating
from mime.interactions import KEPT, Gravity, Magnetic
from mime.observations import SwimSpeed
from mime.scene import Scene, Tube, X, Z
from mime.things import Part
from mime.units import A_m2, Hz, kg_m2, kg_per_m3, m_per_s, m_per_s2, mm, ms, mT, rad_per_s

CentreContact, _ = KEPT["contact", "0.1.0"]


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


# -- the experiment, declared as tutorial A declares one: the handles each add returns are what the laws bind -----------

@dataclass
class Made:
    exp: Experiment
    tube: Any
    water: Any
    robot: Any
    field: Any


def structure(radius: float = 3.175) -> Structure:
    return Structure("tube", geometry=Tube(axis=+X, inner_radius=mm(radius), wall=mm(1.6), length=mm(100.0)))


def medium(tube: Any, record: Any = None) -> Medium:
    return Medium("water", fills=tube.lumen, material=record)


def body(length: float = 7.47, name: str = "robot") -> Body:
    return Body(name, dynamics=RigidInertial(), initial=State(
        position=mm((10.0, 0.0, -1.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0))), parts={"shell": Part(fl9(length=mm(length)))})


def apparatus(tube: Any, strength: float = 1.2, with_field: bool = True, name: str = "field") -> Apparatus:
    fields = {"magnetic": UniformRotating(rotation_axis=tube.axis, initial_direction=(0.0, 1.0, 0.0))}
    return Apparatus(name, fields=fields if with_field else {},
                     commands={"frequency": Hz(10.0), "amplitude": mT(strength)})


def experiment(*, radius: float = 3.175, record: Any = None, density: bool | None = None, length: float = 7.47,
               strength: float = 1.2, margin: float = 0.1, sourced: bool = False) -> Made:
    """Tutorial A's order. The water's density is declared unless ``density`` is False (or a ``record`` gives it); the
    contact law's margin is a source's to give where ``sourced``."""
    exp = Experiment("rebind", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(9.81), dt=ms(0.5),
                     conditions=CONDITIONS)
    tube = exp.add_structure(structure(radius))
    water = exp.add_medium(medium(tube, record))
    if density if density is not None else record is None:
        exp.declare("water.density", kg_per_m3(1000.0), reason=REASON)
    robot = exp.add_body(body(length))
    exp.declare("robot.parts.shell.density", kg_per_m3(1410.0), reason=REASON)
    exp.declare("robot.inertia", kg_m2((1.03e-10,) * 3), reason=REASON)
    exp.declare("robot.magnetic_moment", A_m2((0.0, 1.68e-3, 0.0)), reason=REASON)
    field = exp.add_apparatus(apparatus(tube, strength))
    exp.attach(Magnetic.Torque(), body=robot, field=field.magnetic)
    exp.attach(Gravity(), body=robot, medium=water)
    if sourced:
        exp.source("contact:robot.margin", reason="a datasheet")
    exp.attach(CentreContact.Projection(margin=mm(margin)), body=robot, surface=tube.inner_surface)
    exp.observe(SwimSpeed(robot, along=tube.axis))
    return Made(exp, tube, water, robot, field)


def declarations(exp: Experiment) -> list[Any]:
    return vars(exp)["_declarations"]


def thing_names(exp: Experiment) -> list[str]:
    return [declaration.thing.name for declaration in declarations(exp) if isinstance(declaration, ThingDeclaration)]


def noted(call: Callable[[], object]) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        call()
    return [str(note.message) for note in caught]


def refusals_of(exp: Experiment) -> list[Refusal]:
    try:
        exp.plan()
    except BuildRefused as refused:
        return list(refused.refusals)
    return []


def value_line(exp: Experiment, path: str) -> tuple[Any, str]:
    (line,) = [line for line in exp.plan().values if line.path == path]
    return line.value, line.source


def inner_radius(exp: Experiment) -> float:
    return exp.plan().geometries["tube"].geometry.inner_radius.value


def amplitude(exp: Experiment) -> float:
    (entry,) = [entry for entry in exp.plan().entries if entry.owner == "drive:field.amplitude"]
    return entry.entry.value.value


# -- the four kinds of thing: the edit a cell run again makes, and what the plan builds from it ------------------------

@dataclass(frozen=True)
class Kind:
    """``edit(made)`` declares the thing again, changed, as its cell run again does; ``same(made)`` declares it again
    as before; ``seen(exp)`` is what the plan builds from it that the change moves; ``fresh()`` is the experiment
    declared with the change from the start; ``holders`` is every declaration of ``experiment()`` that holds its name."""

    name: str
    call: str
    edit: Callable[[Made], object]
    same: Callable[[Made], object]
    seen: Callable[[Experiment], Any]
    fresh: Callable[[], Made]
    holders: frozenset[str]
    thing: str


KINDS = (
    Kind("structure", "exp.add_structure(...)", lambda made: made.exp.add_structure(structure(3.2)),
         lambda made: made.exp.add_structure(structure()), inner_radius, lambda: experiment(radius=3.2),
         frozenset({"the medium 'water'", "the apparatus 'field'", "the law contact:robot",
                    "the observation swim_speed:robot"}), "tube"),
    Kind("medium", "exp.add_medium(...)", lambda made: made.exp.add_medium(medium(made.tube, WATER_RECORD)),
         lambda made: made.exp.add_medium(medium(made.tube)), lambda exp: value_line(exp, "water.density"),
         lambda: experiment(record=WATER_RECORD, density=True),
         frozenset({"the value water.density", "the law gravity:robot"}), "water"),
    Kind("body", "exp.add_body(...)", lambda made: made.exp.add_body(body(9.0)),
         lambda made: made.exp.add_body(body()), lambda exp: value_line(exp, "robot.mass")[0],
         lambda: experiment(length=9.0),
         frozenset({"the value robot.parts.shell.density", "the value robot.inertia", "the value robot.magnetic_moment",
                    "the law magnetic.torque:robot", "the law gravity:robot", "the law contact:robot",
                    "the observation swim_speed:robot"}), "robot"),
    Kind("apparatus", "exp.add_apparatus(...)", lambda made: made.exp.add_apparatus(apparatus(made.tube, 2.4)),
         lambda made: made.exp.add_apparatus(apparatus(made.tube)), amplitude, lambda: experiment(strength=2.4),
         frozenset({"the law magnetic.torque:robot"}), "field"),
)
STRUCTURE, MEDIUM, BODY, APPARATUS = KINDS


def named(kind: Kind) -> str:
    return kind.name


# -- the checks: each returns what is wrong, so that a plant can be held against it ---------------------------------------

def edit_problems(kind: Kind) -> list[str]:
    """The edit of ``kind`` in a notebook: one note; as many declarations as before; the plan builds what an experiment
    declared with the change builds, and not what it built before."""
    made = experiment()
    before = kind.seen(made.exp)
    count = len(declarations(made.exp))
    notes = noted(lambda: kind.edit(made))
    problems: list[str] = []
    if len(notes) != 1:
        problems.append(f"{len(notes)} notes, not one: {notes}")
    if len(declarations(made.exp)) != count:
        problems.append("the experiment holds another number of declarations")
    refused = refusals_of(made.exp)
    if refused:
        return [*problems, f"the plan refuses: {[refusal.code for refusal in refused]}"]
    after = kind.seen(made.exp)
    if after == before:
        problems.append("the plan builds what it built before the edit")
    if after != kind.seen(kind.fresh().exp):
        problems.append("the plan does not build what an experiment declared with the change builds")
    return problems


def note_problems(kind: Kind) -> list[str]:
    """The note of the edit: one, naming the thing replaced and every declaration that holds its name, each as what it
    is (a law, an observation, a value, another thing)."""
    made = experiment()
    notes = noted(lambda: kind.edit(made))
    if len(notes) != 1:
        return [f"{len(notes)} notes, not one: {notes}"]
    text, _, _ = notes[0].partition("; of these, ")  # the clause on the declared values is checked on its own
    head, _, rest = text.partition("; ")
    expected = (f"{kind.call} declares the {kind.name} {kind.thing!r} again, not the same as before: in a notebook or "
                f"IPython the last one run wins, so it replaces the earlier one")
    problems = [] if head == expected else [f"the note begins {head!r}, not {expected!r}"]
    clause = "what holds its name now finds the new one: "
    if not rest.startswith(clause):
        return [*problems, f"the note lists no declaration that holds the name: {rest!r}"]
    listed = rest.removeprefix(clause).split(", ")
    if len(listed) != len(set(listed)) or set(listed) != kind.holders:
        problems.append(f"the note names {sorted(listed)}, not {sorted(kind.holders)}")
    return problems


def order_problems(kind: Kind) -> list[str]:
    """The replacement stands where the earlier declaration stood."""
    made = experiment()
    kinds = [type(declaration) for declaration in declarations(made.exp)]
    names = thing_names(made.exp)
    noted(lambda: kind.edit(made))
    held = [type(declaration) for declaration in declarations(made.exp)]
    return [] if held == kinds and thing_names(made.exp) == names else ["the replacement is not in its place"]


def torque(made: Made) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """What a short run records that depends on the field: the torque on the robot and the drive's amplitude."""
    history = made.exp.build().run(ms(5), history=True).history
    return (np.asarray(history["magnetic.torque:robot"]["wrench"]),
            np.asarray(history["drive:field.amplitude"]["value"]))


def natural_edit_problems() -> list[str]:
    """Tutorial A's field cell run again with a new strength, then ``plan()`` and a short run: the run records the new
    strength's torque (the same as an experiment declared with it from the start; at the first step, before the robot has
    turned, twice the old one's)."""
    old = torque(experiment())
    fresh = torque(experiment(strength=2.4))
    made = experiment()
    noted(lambda: made.exp.add_apparatus(apparatus(made.tube, 2.4)))
    made.exp.plan()
    new = torque(made)
    problems: list[str] = []
    if not all(np.array_equal(a, b) for a, b in zip(new, fresh, strict=True)):
        problems.append("the run does not record what an experiment declared with 2.4 mT records")
    if any(np.array_equal(a, b) for a, b in zip(new, old, strict=True)):
        problems.append("the run records what the old strength recorded")
    if not np.isclose(new[0][0, 3] / old[0][0, 3], 2.0, rtol=1e-4):  # the first step, before the robot has turned
        problems.append("the first step's torque is not twice the old one's")
    return problems


def duplicates(exp: Experiment) -> list[Refusal]:
    return [refusal for refusal in refusals_of(exp) if refusal.code == "thing.duplicate"]


def kind_change_problems() -> list[str]:
    """A name declared again as another kind is not replaced: both are kept, the note says so, and the build refuses
    the two in the notebook's words."""
    problems: list[str] = []
    for add, expected in ((lambda made: made.exp.add_body(body(name="field")), "apparatus"),
                          (lambda made: made.exp.add_apparatus(apparatus(made.tube, name="robot")), "body")):
        made = experiment()
        count, first = len(declarations(made.exp)), list(declarations(made.exp))
        notes = noted(lambda: add(made))
        if len(notes) != 1 or f"is of another kind ({expected})" not in notes[0] or "nothing is replaced" not in notes[0]:
            problems.append(f"the note is {notes}")
        if len(declarations(made.exp)) != count + 1 or declarations(made.exp)[:count] != first:
            problems.append("the earlier declarations are not all kept, with the new one after them")
        refused = duplicates(made.exp)
        if not refused or not all(refusal.message.endswith(RAN_TWICE) and refusal.action == RUN_AGAIN
                                  for refusal in refused):
            problems.append("the build does not refuse the two in the notebook's words")
    return problems


def object_problems() -> list[str]:
    """A declaration that holds the earlier object itself, not its name (the controller reads the body object), would
    keep its old settings under the new name: the new one is kept beside it, and the build refuses the two."""
    made = experiment()
    made.exp.set_controller(Ticker(), reads={"pose": body()}, writes={"angle": "field.frequency"}, mode="callback",
                            rate=ms(3))
    count = len(declarations(made.exp))
    notes = noted(lambda: made.exp.add_body(body(9.0)))
    problems: list[str] = []
    if len(notes) != 1 or "the controller hold" not in notes[0] or "not its name" not in notes[0]:
        problems.append(f"the note is {notes}")
    if len(declarations(made.exp)) != count + 1:
        problems.append("the new body is not kept beside the old")
    refused = duplicates(made.exp)
    if not refused or not all(refusal.message.endswith(RAN_TWICE) for refusal in refused):
        problems.append("the build does not refuse the two in the notebook's words")
    return problems


def fails(check: Callable[[], list[str]]) -> bool:
    try:
        return bool(check())
    except BuildRefused:
        return True


# -- the replacement ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", KINDS, ids=named)
def test_a_changed_thing_declared_again_in_a_notebook_is_what_the_plan_builds(kind, notebook):
    assert edit_problems(kind) == []


@pytest.mark.parametrize("kind", KINDS, ids=named)
def test_the_note_names_the_thing_and_each_law_observation_and_value_that_holds_its_name(kind, notebook):
    assert note_problems(kind) == []


@pytest.mark.parametrize("kind", KINDS, ids=named)
def test_the_replacement_takes_the_place_of_the_earlier_declaration(kind, notebook):
    assert order_problems(kind) == []


def test_the_field_cell_run_again_with_a_new_strength_runs_with_the_new_one(notebook):
    assert natural_edit_problems() == []


def test_a_medium_declared_again_with_its_record_gives_the_water_its_density_from_it(notebook):
    """The first medium has no material and no declared density, so the plan refuses; its cell run again with the
    record builds, with the density the record gives (998.2 kg/m^3), as an experiment declared with it does."""
    made = experiment(density=False)
    assert [refusal.code for refusal in refusals_of(made.exp)]
    (note,) = noted(lambda: made.exp.add_medium(medium(made.tube, WATER_RECORD)))
    assert "the law gravity:robot" in note and "the value water.density" not in note
    value, source = value_line(made.exp, "water.density")
    assert (value, source) == value_line(experiment(record=WATER_RECORD, density=False).exp, "water.density")
    assert value == pytest.approx(998.207) and "record of material 'water'" in source


def test_the_note_names_a_source_and_the_controller_that_hold_the_name(notebook):
    made = experiment()
    made.exp.source("robot.parts.shell.density", reason="a datasheet")
    made.exp.set_controller(Ticker(), reads={"pose": "robot.pose"}, writes={"angle": "field.frequency"},
                            mode="callback", rate=ms(3))
    (note,) = noted(lambda: made.exp.add_body(body(9.0)))
    assert "the source of robot.parts.shell.density" in note and "the controller" in note, note
    (note,) = noted(lambda: made.exp.add_apparatus(apparatus(made.tube, 2.4)))
    assert "the controller" in note and "the law magnetic.torque:robot" in note, note


def test_a_thing_nothing_names_is_replaced_and_the_note_says_nothing_holds_its_name(notebook):
    exp = Experiment("alone", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(9.81), dt=ms(0.5))
    exp.add_structure(structure())
    (note,) = noted(lambda: exp.add_structure(structure(3.2)))
    assert note.endswith("so it replaces the earlier one; nothing else holds its name"), note


def test_a_second_replacement_is_the_same_as_the_first_and_is_kept_once(notebook):
    made = experiment()
    noted(lambda: APPARATUS.edit(made))
    count = len(declarations(made.exp))
    (note,) = noted(lambda: APPARATUS.edit(made))
    assert note.endswith("again, the same as before: it is kept once") and len(declarations(made.exp)) == count


def test_several_earlier_declarations_of_the_name_are_all_replaced_by_the_one(monkeypatch):
    """An experiment can hold two of a name (declared in a script, or beside a kind that was refused): a notebook's
    declaration of that name and kind takes the first's place and takes the rest away."""
    made = experiment()
    with monkeypatch.context() as script:
        script.delattr(builtins, "get_ipython", raising=False)
        noted(lambda: BODY.edit(made))
        noted(lambda: BODY.edit(made))  # the same as the second: kept once
    assert thing_names(made.exp).count("robot") == 2
    monkeypatch.setattr(builtins, "get_ipython", lambda: None, raising=False)
    (note,) = noted(lambda: made.exp.add_body(body(8.0)))
    assert "replaces the earlier one" in note
    assert thing_names(made.exp).count("robot") == 1 and refusals_of(made.exp) == []


def test_a_handle_from_the_first_run_binds_a_law_to_the_new_thing_by_its_name(notebook):
    """The law holds ``field.magnetic`` from the first run's handle; the apparatus is declared again and the law's cell
    is not run again: the plan reads the new strength."""
    made = experiment()
    held = [declaration for declaration in declarations(made.exp) if type(declaration).__name__ == "Attachment"]
    assert any(type(binding).__name__ == "FieldPath" for law in held for binding in law.bindings.values())
    noted(lambda: APPARATUS.edit(made))
    assert amplitude(made.exp) == amplitude(experiment(strength=2.4).exp) != amplitude(experiment().exp)
    assert refusals_of(made.exp) == []


# -- the refusals ------------------------------------------------------------------------------------------------------

def test_a_name_declared_again_as_another_kind_is_kept_beside_and_refused_in_the_notebooks_words(notebook):
    assert kind_change_problems() == []


def test_a_declaration_that_holds_the_earlier_object_and_not_its_name_is_kept_beside_and_refused(notebook):
    assert object_problems() == []


def test_the_words_of_the_note_and_the_refusal_say_a_notebook_or_ipython(notebook):
    assert "a notebook or IPython" in RAN_TWICE and RAN_TWICE.startswith("; in a notebook or IPython,")
    made = experiment()
    (note,) = noted(lambda: made.exp.add_body(body(name="field")))
    assert "notebook" not in note.replace("notebook or IPython", "")


@pytest.mark.parametrize("what", ["a field removed from an apparatus", "a structure with no geometry"])
def test_a_law_the_new_thing_cannot_serve_is_named_by_the_note_and_refused_by_its_own_plan(what, notebook):
    made = experiment()
    if what.startswith("a field"):
        edit, law = (lambda: made.exp.add_apparatus(apparatus(made.tube, with_field=False))), "magnetic.torque:robot"
    else:
        edit, law = (lambda: made.exp.add_structure(Structure("tube"))), "contact:robot"
    (note,) = noted(edit)
    assert f"the law {law}" in note
    refused = refusals_of(made.exp)
    assert "thing.duplicate" not in {refusal.code for refusal in refused}
    assert [refusal.code for refusal in refused if f"interaction {law}" in refusal.message] == ["binding.unknown"]


# -- a script, an identical declaration, and the saved form -----------------------------------------------------------------

@pytest.mark.parametrize("kind", KINDS, ids=named)
def test_a_changed_declaration_in_a_script_is_added_with_no_note_and_refused_in_the_scripts_words(kind):
    assert not in_notebook()
    made = experiment()
    count = len(declarations(made.exp))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        kind.edit(made)
    assert len(declarations(made.exp)) == count + 1
    refused = duplicates(made.exp)
    assert refused and all(RAN_TWICE not in refusal.message and refusal.action != RUN_AGAIN for refusal in refused)


@pytest.mark.parametrize("kind", KINDS, ids=named)
def test_the_same_declaration_again_is_kept_once_in_a_script_and_in_a_notebook(kind, place):
    made = experiment()
    before = [id(declaration) for declaration in declarations(made.exp)]
    (note,) = noted(lambda: kind.same(made))
    assert note.endswith("again, the same as before: it is kept once")
    assert [id(declaration) for declaration in declarations(made.exp)] == before
    assert refusals_of(made.exp) == []


def test_a_saved_experiment_holds_only_the_new_declaration_and_loads_and_rebuilds_against_its_lock(notebook, tmp_path):
    made = experiment()
    noted(lambda: APPARATUS.edit(made))
    document = made.exp.document()
    assert sorted(document["things"]) == ["field", "robot", "tube", "water"]
    assert document["things"]["field"]["commands"]["amplitude"]["settings"]["value"]["value"] == pytest.approx(2.4e-3)
    made.exp.save(tmp_path / "saved", steps=2)
    loaded = Experiment.load(tmp_path / "saved")
    assert loaded.load_report is not None and loaded.load_report.derived_differences == ()
    assert loaded.document() == document
    loaded.build()


# -- the trace: a warnings filter removes the warning, and not the record of the replacement ----------------------------

def replaced_of(exp: Experiment) -> list[str]:
    return vars(exp)["_replaced"]


def trace_problems(kind: Kind) -> list[str]:
    """The edit of ``kind`` under ``simplefilter("ignore")`` (a notebook's usual first cell): ``plan()`` and the build
    report hold the note, with every declaration that follows the thing; the saved form is the fresh one's."""
    made = experiment()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kind.edit(made)
    problems: list[str] = []
    plans = made.exp.plan()
    if len(plans.replaced) != 1:
        return [f"plan().replaced holds {len(plans.replaced)} notes, not one"]
    (note,) = plans.replaced
    problems += [f"plan().replaced does not name {holder}" for holder in sorted(kind.holders) if holder not in note]
    if f"declares the {kind.name} {kind.thing!r} again" not in note:
        problems.append("plan().replaced does not name the thing replaced")
    report = made.exp.build().report
    if report.replaced != plans.replaced:
        problems.append("the build report holds another record than the plan")
    if not any(note in line for line in report.text().splitlines()) or \
            "replaced in this session" not in report.text(details=True):
        problems.append("the build report's text does not show the replacement")
    if made.exp.document() != kind.fresh().exp.document():
        problems.append("the saved form is not what an experiment declared with the change holds")
    return problems


def error_filter_problems(kind: Kind) -> list[str]:
    """The edit of ``kind`` under ``simplefilter("error")``: the note raises, and the experiment is as it was."""
    made = experiment()
    before = list(declarations(made.exp))
    problems: list[str] = []
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            kind.edit(made)
            problems.append("the edit did not raise under the error filter")
        except UserWarning:
            pass
    if len(declarations(made.exp)) != len(before) or any(a is not b for a, b in zip(declarations(made.exp), before)):
        problems.append("the call raised and has also changed the declarations")
    if replaced_of(made.exp):
        problems.append("the call raised and has also recorded a replacement")
    if made.exp.plan().replaced:
        problems.append("the plan shows a replacement that did not happen")
    return problems


@pytest.mark.parametrize("kind", KINDS, ids=named)
def test_a_replacement_is_recorded_for_plan_and_report_when_a_filter_removes_the_warning(kind, notebook):
    assert trace_problems(kind) == []


@pytest.mark.parametrize("kind", KINDS, ids=named)
def test_a_warning_raised_as_an_error_leaves_the_experiment_unchanged(kind, notebook):
    assert error_filter_problems(kind) == []


def test_a_leaf_replaced_under_the_error_filter_leaves_the_experiment_unchanged_and_under_ignore_is_recorded(notebook):
    made = experiment()
    before = list(declarations(made.exp))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(UserWarning):
            made.exp.declare("water.density", kg_per_m3(998.0), reason=REASON)
    assert all(a is b for a, b in zip(declarations(made.exp), before, strict=True)) and replaced_of(made.exp) == []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        made.exp.declare("water.density", kg_per_m3(998.0), reason=REASON)
    (note,) = made.exp.plan().replaced
    assert "the value water.density" in note and "replaces the earlier one" in note


def test_a_kept_beside_declaration_is_not_added_when_its_warning_raises(notebook):
    made = experiment()
    count = len(declarations(made.exp))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(UserWarning):
            made.exp.add_body(body(name="field"))
    assert len(declarations(made.exp)) == count


def held_object(made: Made) -> None:
    made.exp.set_controller(Ticker(), reads={"pose": body()}, writes={"angle": "field.frequency"}, mode="callback",
                            rate=ms(3))


def test_a_declaration_kept_beside_because_it_holds_the_object_is_not_added_when_its_warning_raises(notebook):
    made = experiment()
    held_object(made)
    count = len(declarations(made.exp))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(UserWarning, match="not its name"):
            made.exp.add_body(body(9.0))
    assert len(declarations(made.exp)) == count and replaced_of(made.exp) == []


def test_a_law_a_source_names_is_replaced_and_its_warning_raised_as_an_error_leaves_the_experiment_as_it_was(notebook):
    made = experiment()
    made.exp.source("robot.parts.shell.density", reason="a datasheet")
    made.exp.source("contact:robot.margin", reason="a datasheet")
    count = len(declarations(made.exp))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(UserWarning):
            made.exp.attach(CentreContact.Projection(margin=mm(0.2)), body=made.robot, surface=made.tube.inner_surface)
    assert len(declarations(made.exp)) == count and replaced_of(made.exp) == []
    (note,) = noted(lambda: made.exp.attach(CentreContact.Projection(margin=mm(0.2)), body=made.robot,
                                            surface=made.tube.inner_surface))
    assert note.endswith("it replaces the earlier one; what holds its id now finds the new one: the source of "
                         "contact:robot.margin")
    assert len(declarations(made.exp)) == count and replaced_of(made.exp) == [note]


def test_the_note_of_a_held_object_says_lines_or_cells(notebook):
    made = experiment()
    held_object(made)
    (note,) = noted(lambda: made.exp.add_body(body(9.0)))
    assert "run the lines or cells again from the one that makes the experiment" in note and "a cell " not in note


def test_replacements_are_listed_in_the_order_they_were_made_and_a_kept_once_one_is_not_a_replacement(notebook):
    made = experiment()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        APPARATUS.edit(made)
        APPARATUS.edit(made)  # the same as the last: kept once
        STRUCTURE.edit(made)
    first, second = made.exp.plan().replaced
    assert "exp.add_apparatus(...)" in first and "exp.add_structure(...)" in second
    assert "replaced in this session, the last one run winning (2;" in made.exp.build().report.text(details=True)


def test_nothing_is_recorded_for_a_script_or_for_declarations_that_replace_nothing():
    made = experiment()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        APPARATUS.edit(made)  # a script: added beside, refused at the build
        APPARATUS.same(made)
    assert replaced_of(made.exp) == []
    assert "replaced in this session" not in experiment().exp.build().report.text(details=True)


def test_a_refused_plan_carries_the_record_of_what_was_replaced(notebook):
    made = experiment()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        made.exp.add_apparatus(apparatus(made.tube, with_field=False))
    with pytest.raises(BuildRefused) as refused:
        made.exp.plan()
    assert refused.value.report is not None
    (note,) = refused.value.report.replaced
    assert "the law magnetic.torque:robot" in note


# -- the declared values that still apply ------------------------------------------------------------------------------

def test_a_declared_value_of_the_thing_still_applies_and_the_note_says_it_overrides_the_new_record(notebook):
    """Tutorial A declares ``water.density``; the medium declared again with a material record that gives another: the
    declared value wins, as a value declared in the experiment always does, and the note says so."""
    made = experiment()
    (note,) = noted(lambda: MEDIUM.edit(made))
    assert ("; of these, the value water.density (1000.0 kg/m^3) still applies and overrides what the new medium "
            "brings") in note, note
    value, source = value_line(made.exp, "water.density")
    assert value == pytest.approx(1000.0) and "record of material" not in source


def test_several_declared_values_still_apply_and_override(notebook):
    made = experiment()
    (note,) = noted(lambda: BODY.edit(made))
    assert "still apply and override what the new body brings" in note
    assert "the value robot.inertia (" in note.partition("; of these, ")[2]


def test_a_thing_with_no_declared_value_has_no_such_clause(notebook):
    made = experiment()
    (note,) = noted(lambda: APPARATUS.edit(made))
    assert "still appl" not in note


# -- the words in IPython's terminal, where there are lines and no cells --------------------------------------------------

def test_the_refusals_words_fit_a_terminal_as_well_as_a_notebook():
    assert RAN_TWICE == "; in a notebook or IPython, a line or cell has probably run twice, changed the second time"
    assert RUN_AGAIN.startswith("run the lines or cells again from the one that makes the experiment, exp = Experiment")
    assert RUN_AGAIN.endswith("or run them all (Run all, in a notebook)")


def test_the_notes_that_keep_both_say_lines_or_cells(notebook):
    made = experiment()
    (note,) = noted(lambda: made.exp.add_body(body(name="field")))
    assert "run the lines or cells again from the one that makes the experiment" in note and "a cell " not in note


# -- what the first run of the checks missed (the reviewer's four surviving mutants) --------------------------------------

def test_a_thing_whose_name_only_contains_the_replaced_ones_is_not_listed_as_naming_it(notebook):
    """``robot2`` holds ``robot`` as a substring and does not name it: not the body, nor the value declared of it."""
    made = experiment()
    made.exp.add_body(body(name="robot2"))
    made.exp.declare("robot2.inertia", kg_m2((1.03e-10,) * 3), reason=REASON)
    (note,) = noted(lambda: BODY.edit(made))
    assert "robot2" not in note, note
    assert "the value robot.inertia" in note


def test_a_law_named_by_a_near_name_is_replaced_and_not_kept_beside(notebook):
    """A law is replaced where nothing names its id: a value declared at ``contact:robot2.margin`` does not name
    ``contact:robot``, so the law changed and declared again replaces the earlier one."""
    made = experiment()
    made.exp.declare("contact:robot2.margin", mm(0.1), reason=REASON)
    count = len(declarations(made.exp))
    (note,) = noted(lambda: made.exp.attach(CentreContact.Projection(margin=mm(0.2)), body=made.robot,
                                            surface=made.tube.inner_surface))
    assert "replaces the earlier one" in note and "both are kept" not in note and len(declarations(made.exp)) == count


def test_a_name_held_twice_with_one_of_another_kind_is_refused_and_nothing_is_removed(monkeypatch):
    """Two earlier declarations of ``x`` (a body and an apparatus, declared in a script): a body declared again in a
    notebook is of the kind of one and not of the other, so nothing is replaced."""
    exp = Experiment("twice", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(9.81), dt=ms(0.5))
    tube = exp.add_structure(structure())
    exp.add_body(body(name="x"))
    exp.add_apparatus(apparatus(tube, name="x"))
    assert thing_names(exp).count("x") == 2
    monkeypatch.setattr(builtins, "get_ipython", lambda: None, raising=False)
    count = len(declarations(exp))
    (note,) = noted(lambda: exp.add_body(body(9.0, name="x")))
    assert "is of another kind (apparatus)" in note and "nothing is replaced" in note
    assert len(declarations(exp)) == count + 1 and thing_names(exp).count("x") == 3


def test_a_body_inside_a_list_a_declaration_holds_is_found_as_a_held_object(notebook):
    made = experiment()
    made.exp.set_controller(Ticker(), reads={"pose": [body()]}, writes={"angle": "field.frequency"}, mode="callback",
                            rate=ms(3))
    count = len(declarations(made.exp))
    (note,) = noted(lambda: made.exp.add_body(body(9.0)))
    assert "the controller holds the earlier one itself" in note and "not its name" in note
    assert len(declarations(made.exp)) == count + 1


def test_the_action_the_refusal_gives_in_a_notebook_is_these_words(notebook):
    made = experiment()
    noted(lambda: made.exp.add_body(body(name="field")))
    (refusal, *_) = duplicates(made.exp)
    assert refusal.action.startswith("run the lines or cells again from the one that makes the experiment, ")
    assert refusal.action.endswith("or run them all (Run all, in a notebook)")


# -- the controls: a re-bind planted wrong fails the check it belongs to, every run ----------------------------------------

def keeps_the_old_declaration(monkeypatch):
    """Says it replaced; the earlier declaration stays (the old settings under the new name)."""
    def planted(held, earlier, declaration, known, name, replaced):
        warnings.warn(f"{known.call} declares {known.noun} again, not the same as before: in a notebook or IPython the "
                      f"last one run wins, so it replaces the earlier one; nothing else holds its name", UserWarning)
    monkeypatch.setattr(redeclared, "_replace_thing", planted)


def adds_beside_the_earlier_one(monkeypatch):
    monkeypatch.setattr(redeclared, "_replace_thing", lambda held, earlier, declaration, known, name, replaced:
                        held.append(declaration))


def puts_the_new_one_at_the_end(monkeypatch):
    def planted(held, earlier, declaration, known, name, replaced):
        del held[earlier[0]]
        held.append(declaration)
    monkeypatch.setattr(redeclared, "_replace_thing", planted)


def says_nothing(monkeypatch):
    def planted(held, earlier, declaration, known, name, replaced):
        held[earlier[0]] = declaration
    monkeypatch.setattr(redeclared, "_replace_thing", planted)


def a_note_that_skips_the_observations(monkeypatch):
    real = redeclared._naming
    monkeypatch.setattr(redeclared, "_naming", lambda *args: [text for text in real(*args)
                                                              if not text.startswith("the observation")])


def a_note_that_skips_a_mediums_fills(monkeypatch):
    real = redeclared.names_in

    def planted(value, _open=frozenset()):
        yield from real(dataclasses_replace_fills(value), _open)
    monkeypatch.setattr(redeclared, "names_in", planted)


def dataclasses_replace_fills(value):
    import dataclasses
    return dataclasses.replace(value, fills=None) if isinstance(value, Medium) else value


def records_no_replacement(monkeypatch):
    """Replaces and warns, and leaves no trace on the experiment."""
    def planted(held, earlier, declaration, note, replaced):
        warnings.warn(note, UserWarning)
        held[earlier[0]] = declaration
        for index in reversed(earlier[1:]):
            del held[index]
    monkeypatch.setattr(redeclared, "_replace", planted)


def changes_the_experiment_before_the_note(monkeypatch):
    """The note comes after the change: under the error filter the call raises and has also changed the experiment."""
    def planted(held, earlier, declaration, note, replaced):
        held[earlier[0]] = declaration
        for index in reversed(earlier[1:]):
            del held[index]
        replaced.append(note)
        warnings.warn(note, UserWarning)
    monkeypatch.setattr(redeclared, "_replace", planted)


def a_record_without_the_followers(monkeypatch):
    """The trace names the thing and nothing that follows it."""
    real = redeclared._replace
    monkeypatch.setattr(redeclared, "_replace", lambda held, earlier, declaration, note, replaced:
                        real(held, earlier, declaration, note, replaced) or
                        replaced.__setitem__(-1, note.partition("; what holds")[0]))


def a_note_that_hides_the_value_that_overrides(monkeypatch):
    monkeypatch.setattr(redeclared, "_still_applies", lambda values, kind: "")


def replaces_across_kinds(monkeypatch):
    monkeypatch.setattr(redeclared, "_other_kinds", lambda held, earlier, declaration: [])


def does_not_look_for_a_held_object(monkeypatch):
    monkeypatch.setattr(redeclared, "_holding_the_object", lambda held, earlier, name: [])


CONTROLS = (
    (keeps_the_old_declaration, natural_edit_problems),
    (keeps_the_old_declaration, lambda: edit_problems(APPARATUS)),
    (keeps_the_old_declaration, lambda: edit_problems(STRUCTURE)),
    (adds_beside_the_earlier_one, natural_edit_problems),
    (adds_beside_the_earlier_one, lambda: edit_problems(BODY)),
    (puts_the_new_one_at_the_end, lambda: order_problems(MEDIUM)),
    (puts_the_new_one_at_the_end, lambda: order_problems(STRUCTURE)),
    (says_nothing, lambda: note_problems(APPARATUS)),
    (says_nothing, lambda: note_problems(BODY)),
    (a_note_that_skips_the_observations, lambda: note_problems(STRUCTURE)),
    (a_note_that_skips_the_observations, lambda: note_problems(BODY)),
    (a_note_that_skips_a_mediums_fills, lambda: note_problems(STRUCTURE)),
    (records_no_replacement, lambda: trace_problems(APPARATUS)),
    (records_no_replacement, lambda: trace_problems(MEDIUM)),
    (a_record_without_the_followers, lambda: trace_problems(STRUCTURE)),
    (a_record_without_the_followers, lambda: trace_problems(BODY)),
    (changes_the_experiment_before_the_note, lambda: error_filter_problems(APPARATUS)),
    (changes_the_experiment_before_the_note, lambda: error_filter_problems(STRUCTURE)),
    (replaces_across_kinds, kind_change_problems),
    (does_not_look_for_a_held_object, object_problems),
)


@pytest.mark.parametrize(("plant", "check"), CONTROLS, ids=lambda item: getattr(item, "__name__", "check"))
def test_control_a_re_bind_planted_wrong_fails_its_check_every_run(plant, check, notebook, monkeypatch):
    assert not fails(check), "the check fails without the plant"
    plant(monkeypatch)
    assert fails(check), f"{plant.__name__} passed a check that cannot then tell it from the real re-bind"
