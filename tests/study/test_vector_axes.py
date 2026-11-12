"""A study over a kind's setting that is not one number (an ellipsoid's semi-axes, a tube tree's radii, a plane's
normal): the setting whole, each point a vector, or one component by its index (``egg.parts.shell.semi_axes[2]``,
``vessel.radii[3]``, ``floor.normal[1]``), each value a build of its own, its group split by that value and the split
reason naming the axis.

**Gate 1** (``predictions_vector_axes.json``, and its fix round's ``predictions_vector_axes_fix.json``): the study
equals the same points written by hand as experiments of their own, bit for bit per point. Each point's program (its
structure key without the axis's value), leaves and initial state are the hand-written experiment's, and each member's
final state and series are the hand-written experiment's member, run the same way: a study of the hand-written
experiment over a scalar axis at the same values, the same batched scan at the same batch size (a member against
``run()`` is not bit for bit: a batched member is another compiled program, ``tests/study/predictions.json``). The
cases: the MRS rung on an ellipsoid (its semi-axes whole; its third semi-axis by the water's viscosity, two members a
group), on a ball above a plane whose normal is the axis (whole; one component), and the FVM in a tube tree (one
radius, at its points). The plane's normal is a leaf of the drag node, so every normal builds one structure: its two
groups share one structure key and are split by the axis's value alone, the case the value is in a point's key for.

**The controls**, planted by ``MIME_VECTOR_PLANT`` (each must fail gate 1 on every run, measured with
``.github/probe/detection_rate.py``): ``keep-base``, a point that rebuilds the kind and silently keeps the experiment's own
value; ``next-index``, a component written one place along; ``struct-cache``, a group's leaves taken from the first
point of its structure key, as a cache of builds keyed by the structure key alone would (the plane's cases: elsewhere
each value has a structure key of its own, and the plant changes nothing). The cases move each value far enough to
change the build (the test holds the hand-written builds of two values to differ, and on the plane their runs), so a
plant cannot pass by changing nothing."""

from __future__ import annotations

import dataclasses
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import jax
import numpy as np
import pytest
import yaml
from mrs_cases import WATER, free_space, mrs_plane, uncoupled

import mime.study._axes as axes_module
from mime import Study
from mime.build import BuildRefused
from mime.build._refusals import ArgumentRefused
from mime.dynamics import FVM, RigidInertial, Scheme
from mime.experiment import Experiment
from mime.interactions import BodyForce
from mime.results import StudyResult
from mime.scene import Ellipsoid, PeriodicBox, Scene, Sphere, TubeTree, Z
from mime.study import along, grid
from mime.saved._pins import keeping_pins
from mime.results._study_folder import decode_value, encode_value
from mime.study._axes import STUDY_REASON, Axis, AxisKind, differ, resolve
from mime.study._point import Point, build_point
from mime.study._ranges import assembled_program, programs_differ
from mime.study._spec import value_of, value_text
from mime.things import Body, Medium, Part, State, Structure
from mime.units import DIMENSIONLESS, Pa_s, Quantity, kg, kg_m2, kg_per_m3, m_per_s, m_per_s2, mPa_s, mm, ms, rad_per_s

EGG = (0.5, 0.7, 1.1)
SEMI_AXES = "egg.parts.shell.semi_axes"
#: The periodic tree of the FVM's tree tests: a vessel narrowing to 0.8 of its radius over half its 1 mm period.
POINTS = (0.0, 0.0, -0.5, 0.0, 0.0, 0.25, 0.0, 0.0, 0.5, 0.0, 0.0, 0.75, 0.0, 0.0, 1.5)
SEGMENTS = ((0, 1), (1, 2), (2, 3), (3, 4))
RADII = (0.5, 0.5, 0.5, 0.4, 0.4, 0.5, 0.5, 0.5)
#: The plane's normal in ``mrs_plane``, and a normal at the world's up.
NORMAL, UP = (0.0, 0.6, 0.8), (0.0, 0.0, 1.0)
PLANT = os.environ.get("MIME_VECTOR_PLANT", "")


def egg(semi_axes: Sequence[float] = EGG, viscosity: Any = mPa_s(1.0)) -> Experiment:
    """The MRS rung (drag 0.4.0, a node a body) on an ellipsoid in free space, moving and turning from the start."""
    experiment = free_space("egg", {"egg": Ellipsoid(semi_axes=mm(tuple(semi_axes)))}, uncoupled(mm(0.35)))
    if viscosity != mPa_s(1.0):
        experiment.declare("water.viscosity", viscosity, reason="water")
    return experiment


def tree(radii: Sequence[float] = RADII) -> Experiment:
    """The FVM's steady Stokes flow in the periodic tree's lumen, under a body force on the water."""
    experiment = Experiment("tree", scene=Scene.new(up=Z, units="m"), dt=ms(5.0))
    experiment.add_structure(Structure("vessel", geometry=TubeTree(points=mm(POINTS), segments=SEGMENTS,
                                                                   radii=mm(tuple(radii)))))
    flow = FVM(domain=PeriodicBox(length=mm(1.0), margin=4), cells_across=8, stokes=True, scheme=Scheme.Steady())
    experiment.add_medium(Medium("water", fills="vessel.lumen", fields={"flow": flow}))
    experiment.declare("water.viscosity", Pa_s(1e-3), reason="water near 20 C")
    experiment.declare("water.density", kg_per_m3(1000.0), reason="water near 20 C")
    experiment.attach(BodyForce(along=(0.0, 0.0, 1.0), steady=m_per_s2(0.01)), medium="water")
    return experiment


def _with(values: Sequence[float], index: int, value: float) -> tuple[float, ...]:
    held = list(values)
    held[index] = value
    return tuple(held)


def _cases() -> dict[str, Any]:
    """Gate 1's cases: the base, the study's points, the duration, and for each group the hand-written experiment and
    the water's viscosity at each of its members (the hand-written study's points, in the order the group holds
    them). The tree's case is held at its points (below): a tree's build and compile is the costliest here."""
    return {
        "egg-whole": (egg, along({SEMI_AXES: [mm(EGG), mm((0.55, 0.65, 1.2))]}), ms(2.0),
                      [(lambda: egg(EGG), [mPa_s(1.0)]), (lambda: egg((0.55, 0.65, 1.2)), [mPa_s(1.0)])]),
        "egg-component": (egg, grid({f"{SEMI_AXES}[2]": [mm(1.1), mm(1.25)], "water.viscosity": [mPa_s(1.0),
                                                                                                    mPa_s(1.2)]}),
                          ms(2.0), [(lambda: egg(EGG), [mPa_s(1.0), mPa_s(1.2)]),
                                    (lambda: egg(_with(EGG, 2, 1.25)), [mPa_s(1.0), mPa_s(1.2)])]),
        "plane-whole": (mrs_plane, along({"floor.normal": [NORMAL, UP]}), ms(2.0),
                        [(lambda: mrs_plane(NORMAL), [WATER]), (lambda: mrs_plane(UP), [WATER])]),
        "plane-component": (mrs_plane, along({"floor.normal[1]": [0.6, 0.5]}), ms(2.0),
                            [(lambda: mrs_plane(NORMAL), [WATER]), (lambda: mrs_plane((0.0, 0.5, 0.8)), [WATER])]),
    }


#: Gate 1's cases whose two groups share one structure key, split by the axis's value alone (the plane's normal is a
#: leaf of the drag node): where a cache of builds keyed by the structure key would run one value's leaves for the other.
SHARED = ("plane-whole", "plane-component")


def _planted(axis: Axis, current: object, value: object, written: Callable[..., object]) -> object:
    """The controls' plants on the point's write: the experiment's own value kept, or a component written one place
    along."""
    if PLANT == "keep-base":
        return current
    if PLANT == "next-index" and axis.component is not None:
        count = axes_module.length(current)
        assert count is not None
        return written(Axis(axis.path, axis.kind, axis.owner, axis.name, (axis.component + 1) % count, True), current,
                       value)
    return written(axis, current, value)


def _first_of_its_structure(held: Callable[..., Any]) -> Callable[..., Any]:
    """The ``struct-cache`` plant on a batch's leaves: its first member's read from the first point of the study that
    has its structure key."""
    def stacked(points: Sequence[Point], members: Sequence[int], varying: Any) -> Any:
        first = points[members[0]]
        cached = next(point for point in points if point.structure == first.structure and point.tree)
        moved = list(points)
        moved[members[0]] = dataclasses.replace(first, tree=cached.tree)
        return held(moved, members, varying)

    return stacked


@pytest.fixture
def plant(monkeypatch):
    if PLANT == "struct-cache":
        import mime.study._study as study_module

        monkeypatch.setattr(study_module, "stacked_tree", _first_of_its_structure(study_module.stacked_tree))
    elif PLANT:
        held = axes_module.written_value
        monkeypatch.setattr(axes_module, "written_value",
                            lambda axis, current, value: _planted(axis, current, value, held))
    yield PLANT


def _arrays_differ(a: Mapping[str, Mapping[str, Any]], b: Mapping[str, Mapping[str, Any]]) -> list[str]:
    found = [f"node {node}" for node in sorted(set(a) ^ set(b))]
    for node in sorted(set(a) & set(b)):
        for name in sorted(set(a[node]) | set(b[node])):
            x, y = a[node].get(name), b[node].get(name)
            if x is None or y is None or np.asarray(x).dtype != np.asarray(y).dtype or \
                    not np.array_equal(np.asarray(x), np.asarray(y), equal_nan=True):
                found.append(f"{node}.{name}")
    return found


def _member_differs(a: Any, b: Any) -> list[str]:
    found = _arrays_differ(a.final, b.final)
    return [f"final {each}" for each in found]


def gate_one(name: str) -> dict[str, Any]:
    make, points, duration, hand = _cases()[name]
    study = Study(make()).sweep(points, duration=duration)
    plan = study.plan()
    result = study.run()
    assert len(plan.groups) == len(hand), plan.text(memory=False)
    found: list[str] = []
    members = 0
    hand_points: list[Point] = []
    hand_finals: list[Mapping[str, Mapping[str, Any]]] = []
    for group, (written, values) in zip(plan.groups, hand, strict=True):
        scalar = "water.viscosity"
        reference = Study(written()).sweep(along({scalar: values}), duration=duration)
        reference_plan = reference.plan()
        reference_result: StudyResult = reference.run()
        assert len(reference_plan.groups) == 1, reference_plan.text(memory=False)
        theirs = [reference_plan.points[i] for i in reference_plan.groups[0].members]
        hand_points.append(theirs[0])
        hand_finals.append(reference_result.member(0).final)
        for at, (index, other) in enumerate(zip(group.members, theirs, strict=True)):
            mine = plan.points[index]
            if mine.structure != other.key:
                found.append(f"point {index}: the structure key")
            found += [f"point {index}: leaf {each}" for each in _arrays_differ(mine.tree, other.tree)]
            found += [f"point {index}: initial {each}" for each in _arrays_differ(mine.state, other.state)]
            member = result.member(list(result.points).index(index))
            found += [f"point {index}: {each}" for each in _member_differs(member, reference_result.member(at))]
            members += 1
    return {"plan": plan, "found": found, "members": members, "hand": hand_points, "finals": hand_finals}


GATE_CASES = ["egg-whole", "egg-component", *SHARED]


def judged(name: str) -> None:
    """Gate 1 on the case ``name``, with its premises: the hand-written builds of its two values are not one program
    with the same leaves, and on a shared case one structure key holds two groups, split by the axis's value alone,
    whose runs differ (so a group run on the other value's leaves is seen)."""
    held = gate_one(name)
    plan = held["plan"]
    assert held["members"] == len(plan.points) == sum(len(g.members) for g in plan.groups)
    first, second = held["hand"]
    assert first.key != second.key or _arrays_differ(first.tree, second.tree), "the two values build the same"
    if name in SHARED:
        structures = {plan.points[group.members[0]].structure for group in plan.groups}
        assert len(structures) == 1 and len(plan.groups) == 2, plan.text(memory=False)
        assert _arrays_differ(*held["finals"]), "the two values run the same"
    assert held["found"] == [], held["found"]


@pytest.mark.parametrize("name", GATE_CASES)
def test_a_study_over_a_vector_setting_equals_the_points_written_by_hand(name, plant, float64_gate):
    judged(name)


@pytest.fixture
def float64_gate():
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", held)


def test_the_trees_points_over_one_radius_are_the_points_written_by_hand(plant, float64_gate):
    """Gate 1 on the tube tree, at its points: each point of a study over ``vessel.radii[3]`` (built as the study builds
    it, ``mime.study._point.build_point``) has the hand-written tree's structure key, leaves, initial state and traced
    step, its text and its constants: the same program on the same inputs. Held here and not through a run: the study
    and its hand-written studies took 5.6 min on a runner (tracing and lowering 74 %). It does not hold the study's run
    path (grouping, the leaves and state each group is fed, chunking, results): the egg's and the plane's cases above
    do, the plane's with two groups of one structure key as the tree's two radii are."""
    base = tree()
    settings, declarations, saved = base.declared_as()
    axis = resolve("vessel.radii[3]", declarations)
    found: list[str] = []
    hand: list[Point] = []
    for value, radii in ((0.4, RADII), (0.42, _with(RADII, 3, 0.42))):
        with keeping_pins(saved):
            mine = build_point(0, settings, declarations, {axis: mm(value)}, STUDY_REASON)
        written = tree(radii)
        held_settings, held_declarations, held_saved = written.declared_as()
        with keeping_pins(held_saved):
            theirs = build_point(0, held_settings, held_declarations, {}, STUDY_REASON)
        assert not mine.refused and not theirs.refused, (mine.refusals, theirs.refusals)
        hand.append(theirs)
        if mine.structure != theirs.key:
            found.append(f"radius {value}: the structure key")
        found += [f"radius {value}: leaf {each}" for each in _arrays_differ(mine.tree, theirs.tree)]
        found += [f"radius {value}: initial {each}" for each in _arrays_differ(mine.state, theirs.state)]
        with keeping_pins(saved):
            program = assembled_program(mine.declared.settings, mine.declared.declarations)
        with keeping_pins(held_saved):
            if programs_differ(program, assembled_program(held_settings, held_declarations)):
                found.append(f"radius {value}: the traced step")
    assert hand[0].key != hand[1].key or _arrays_differ(hand[0].tree, hand[1].tree), "the two values build the same"
    assert found == [], found


@pytest.mark.parametrize("name", ["egg-component", "plane-whole"])
def test_a_study_in_float32_equals_the_points_written_by_hand(name, plant):
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        judged(name)
    finally:
        jax.config.update("jax_enable_x64", held)


def test_each_value_is_a_group_of_its_own_split_by_the_axis_and_the_plan_states_the_cut(float64_gate):
    path = f"{SEMI_AXES}[2]"
    plan = Study(egg()).sweep(along({path: [mm(1.1), mm(1.25), mm(1.1)]}), duration=ms(2.0)).plan()
    assert [group.members for group in plan.groups] == [[0, 2], [1]], plan.text(memory=False)
    assert plan.groups[1].reasons[0] == f"its value of {path}, a kind's setting the build holds as built (one build a value)"
    text = plan.text(memory=False)
    assert f"built per value, each value a group of its own: {path}; no derivative is taken along it" in text
    assert plan.points[0].key == plan.points[2].key != plan.points[1].key
    assert plan.points[0].structure is not None and plan.points[0].key.startswith(plan.points[0].structure + ":")


def codes(refused: ArgumentRefused | BuildRefused) -> list[str]:
    return [each.code for each in refused.refusals]


@pytest.mark.parametrize(("path", "code"), [
    (f"{SEMI_AXES}[3]", "study.component-index"),
    ("egg.mass[0]", "study.component-path"),
    ("egg.initial.position[1]", "study.component-path"),
    ("vessel.segments", "study.held-geometry"),
    ("vessel.ends", "study.held-geometry"),
    ("vessel.fillet[0]", "study.component-path"),
    ("vessel.segments[0]", "study.held-geometry"),
    ("vessel.radii[3][0]", "study.component-path"),
    (f"{SEMI_AXES}[2][1]", "study.component-path"),
    ("vessel.radii[1][0]", "study.component-path"),
    ("vessel.radii[03]", "study.component-spelling"),
    ("vessel.radii[00]", "study.component-spelling"),
    ("vessel.radii[-1]", "study.component-spelling"),
    ("vessel.radii[ 3]", "study.component-spelling"),
    ("vessel.radii[]", "study.component-spelling"),
    ("vessel.radii[03][0]", "study.component-spelling"),
], ids=["past-the-end", "a-value", "an-initial-value", "a-structure", "names", "a-bare-numbers-component",
        "a-structures-component", "a-components-component", "a-semi-axis-component", "a-segments-start",
        "a-leading-zero", "zero-twice", "a-sign", "a-space", "no-index", "a-leading-zero-then-an-index"])
def test_a_path_that_names_no_vector_component_is_refused_with_its_code(path, code):
    declarations = [*egg().declared_as()[1], *(d for d in tree().declared_as()[1])]
    with pytest.raises(ArgumentRefused) as refused:
        resolve(path, declarations)
    assert codes(refused.value) == [code] and path.partition("[")[0] in refused.value.refusals[0].message


@pytest.mark.parametrize("index", range(len(RADII)))
def test_a_component_path_writes_the_number_it_names_and_saves_in_one_spelling(index):
    """Each index of the tree's radii writes that number alone, and the axis's path is the one spelling: a nested or a
    zero-led index, which named another number or a second axis on one number, is refused above."""
    settings, declarations, _ = tree().declared_as()
    axis = resolve(f"vessel.radii[{index}]", declarations)
    assert (axis.path, axis.component) == (f"vessel.radii[{index}]", index)
    written = axes_module.declared(settings, declarations, {axis: mm(0.42)}, "a test")
    held = axes_module.base_value(resolve("vessel.radii", declarations), written.settings, written.declarations)
    assert held == mm(_with(RADII, index, 0.42))


@pytest.mark.parametrize("component", range(3))
def test_a_directions_component_writes_that_component_alone(component):
    """A component of a plane's normal (bare numbers, no unit) writes that number into the normal as the experiment
    declares it, the others kept: the review's ``dir-next`` (written one place along) fails here on every run."""
    settings, declarations, _ = mrs_plane().declared_as()
    axis = resolve(f"floor.normal[{component}]", declarations)
    assert axis.built and axis.component == component
    written = axes_module.declared(settings, declarations, {axis: 0.25}, "a test")
    held = axes_module.base_value(resolve("floor.normal", declarations), written.settings, written.declarations)
    assert held == _with(NORMAL, component, 0.25)
    assert axes_module.base_value(axis, written.settings, written.declarations) == 0.25


def test_a_vector_setting_resolves_whole_and_by_its_components():
    declarations = tree().declared_as()[1]
    whole, third = resolve("vessel.radii", declarations), resolve("vessel.radii[3]", declarations)
    assert (whole.built, whole.component, whole.name) == (True, None, "radii")
    assert (third.built, third.component, third.name, third.path) == (True, 3, "radii", "vessel.radii[3]")
    fillet = resolve("vessel.fillet", declarations)
    assert fillet.built and fillet.component is None
    assert "vessel.radii" in axes_module.candidates(declarations)
    written = axes_module.declared(tree().declared_as()[0], declarations, {third: mm(0.42)}, "a test")
    assert axes_module.base_value(whole, written.settings, written.declarations) == mm(_with(RADII, 3, 0.42))
    assert axes_module.base_value(third, written.settings, written.declarations) == mm(0.42)
    with pytest.raises(ArgumentRefused) as refused:
        axes_module.declared(tree().declared_as()[0], declarations, {whole: mm(RADII[:6])}, "a test")
    assert codes(refused.value) == ["study.vector-value"]


@pytest.mark.parametrize(("path", "value"), [
    (SEMI_AXES, mm((0.5, 0.7))),
    (SEMI_AXES, Pa_s((0.5, 0.7, 1.1))),
    (f"{SEMI_AXES}[0]", Pa_s(0.5)),
    (f"{SEMI_AXES}[0]", mm((0.5, 0.7, 1.1))),
], ids=["short", "units", "component-units", "component-vector"])
def test_a_point_of_another_form_is_refused_with_its_code(path, value):
    with pytest.raises(ArgumentRefused) as refused:
        Study(egg()).sweep(along({path: [value]}), duration=ms(2.0)).plan()
    assert codes(refused.value) == ["study.vector-value"] and path in refused.value.refusals[0].message


@pytest.mark.parametrize("axes", [{SEMI_AXES: [mm(EGG)], f"{SEMI_AXES}[0]": [mm(0.5)]},
                                  {"floor.normal": [NORMAL], "floor.normal[2]": [0.8]}], ids=["semi-axes", "a-normal"])
def test_a_setting_named_whole_and_by_a_component_is_refused(axes):
    experiment = egg() if SEMI_AXES in axes else mrs_plane()
    with pytest.raises(ArgumentRefused) as refused:
        Study(experiment).sweep(grid(axes), duration=ms(2.0)).plan()
    assert codes(refused.value) == ["study.axis-twice"]


def test_a_part_of_a_body_of_two_is_written_alone():
    """A body of two parts of one kind: a component path of one part's semi-axes writes that part's, and the other
    part keeps its own (a write to another part of the same kind would pass every gate case above, whose bodies have
    one part)."""
    experiment = free_space("pair", {}, uncoupled(mm(1.0)))
    experiment.add_body(Body("pair", dynamics=RigidInertial(), parts={
        "front": Part(Ellipsoid(semi_axes=mm(EGG)), at=mm((0.0, 0.0, 1.0))),
        "back": Part(Ellipsoid(semi_axes=mm(EGG)), at=mm((0.0, 0.0, -1.0)))},
        initial=State(position=mm((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0))))
    settings, declarations, _ = experiment.declared_as()
    axis = resolve("pair.parts.back.semi_axes[1]", declarations)
    written = axes_module.declared(settings, declarations, {axis: mm(0.75)}, "a test")
    held = {part: axes_module.base_value(resolve(f"pair.parts.{part}.semi_axes", declarations), written.settings,
                                         written.declarations) for part in ("front", "back")}
    assert held == {"front": mm(EGG), "back": mm(_with(EGG, 1, 0.75))}


def test_a_count_is_written_from_any_whole_number():
    """A count takes a NumPy whole number as it takes a Python one (the review's minor 4), and refuses a fraction."""
    axis = Axis("screw.parts.shell.starts", AxisKind.GEOMETRY, "screw", "parts.shell.starts", built=True)
    assert axes_module.written_value(axis, 2, np.int64(3)) == 3
    assert type(axes_module.written_value(axis, 2, np.int64(3))) is int
    with pytest.raises(ArgumentRefused) as refused:
        axes_module.written_value(axis, 2, 2.5)
    assert codes(refused.value) == ["study.vector-value"]


def test_a_derivative_a_fit_an_override_and_a_range_along_a_built_axis_are_refused(float64_gate):
    path = f"{SEMI_AXES}[2]"
    experiment = egg()
    with pytest.raises(BuildRefused) as fitted:
        Study(experiment).fit(lambda final: final["egg"]["pose"][:3], start={path: mm(1.1)}, noise=1e-6,
                              duration=ms(2.0))
    assert codes(fitted.value) == ["gradient.built-value"] and "a declared cut" in fitted.value.refusals[0].message
    study = Study(experiment).sweep(along({path: [mm(1.1), mm(1.25)]}), duration=ms(2.0))
    with pytest.raises(BuildRefused) as derived:
        study.gradient(lambda final: final["egg"]["pose"][0], wrt=[path])
    assert codes(derived.value) == ["gradient.built-value"]
    built = experiment.build()
    with pytest.raises(BuildRefused) as overridden:
        built.run(ms(2.0), override={path: mm(1.25)})
    assert codes(overridden.value) == ["override.structural"] and path in overridden.value.refusals[0].message
    ranged = egg()
    ranged.declare_range(path, mm(1.0), mm(1.2), reason="a range of one semi-axis")
    with pytest.raises(BuildRefused) as refused:
        ranged.build()
    assert "range.units" in codes(refused.value)


def test_a_built_axis_nothing_reads_is_refused_as_inert(float64_gate):
    """The tree as a structure no medium fills, beside a body that moves: its radii reach no node, so two values give
    one structure, the same leaves and state, and one traced step."""
    with pytest.raises(BuildRefused) as refused:
        Study(vessel_alone()).sweep(along({"vessel.radii[3]": [mm(0.4), mm(0.42)]}), duration=ms(10.0)).plan()
    assert codes(refused.value) == ["study.axis-inert"]
    assert "the same traced step" in refused.value.refusals[0].message


def vessel_alone() -> Experiment:
    """The tree as a structure no medium fills, beside a ball that moves on its own: the radii reach no node."""
    experiment = Experiment("vessel_alone", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(5.0))
    experiment.add_structure(Structure("vessel", geometry=TubeTree(points=mm(POINTS), segments=SEGMENTS,
                                                                   radii=mm(RADII))))
    experiment.add_body(Body("ball", dynamics=RigidInertial(), parts={"shell": Part(Sphere(radius=mm(0.1)))},
                             initial=State(position=mm((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0),
                                           velocity=m_per_s((1e-3, 0.0, 0.0)),
                                           angular_velocity=rad_per_s((0.0, 0.0, 1.0)))))
    experiment.declare("ball.mass", kg(1e-9), reason="a 0.1 mm ball's order")
    experiment.declare("ball.inertia", kg_m2((1e-18, 1e-18, 1e-18)), reason="a 0.1 mm ball's order")
    return experiment


def test_a_built_axis_whose_traced_step_alone_differs_is_not_inert(monkeypatch, float64_gate):
    """The other side of the inert check: where a value a node holds as built shows in the traced step alone (planted:
    each point's step traced with its radii written into it, as a node holding them as constants would be), the axis
    is not refused, since each value is built; the same study without the plant is (above)."""
    import mime.study._ranges as ranges

    held = ranges.assembled_program

    def baked(settings: Any, declarations: Any) -> Any:
        text, constants = held(settings, declarations)
        radii = axes_module.base_value(resolve("vessel.radii", declarations), settings, declarations)
        return f"{text} {radii}", constants

    monkeypatch.setattr(ranges, "assembled_program", baked)
    plan = Study(vessel_alone()).sweep(along({"vessel.radii[3]": [mm(0.4), mm(0.42)]}), duration=ms(10.0)).plan()
    assert [group.members for group in plan.groups] == [[0], [1]] and not plan.refused


HALF_TURN = math.sqrt(0.5)
#: Every form of value an axis takes, as a study holds it: one number with its unit, a vector of them, a dimensionless
#: one, a bare number, a count, a name, a switch, and tuples of bare numbers (a direction of floats, one of whole numbers,
#: a quaternion).
VALUE_FORMS = {"a-quantity": mm(1.1), "a-vector-quantity": mm(EGG), "a-dimensionless-quantity": Quantity(0.5, DIMENSIONLESS),
               "a-dimensionless-vector": Quantity((0.0, 0.6, 0.8), DIMENSIONLESS), "a-number": 0.25, "a-count": 3,
               "a-name": "float32", "a-switch": True, "a-direction": NORMAL, "a-direction-of-whole-numbers": (0, 0, 1),
               "a-quaternion": (HALF_TURN, HALF_TURN, 0.0, 0.0)}


def same_form(a: object, b: object) -> bool:
    """Whether two values are one value in one form: a quantity by its dimension and SI numbers, a tuple number by
    number with each number's type, anything else equal and of one type."""
    if isinstance(a, Quantity) or isinstance(b, Quantity):
        return isinstance(a, Quantity) and isinstance(b, Quantity) and not differ(a, b)
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(type(x) is type(y) and x == y for x, y in zip(a, b, strict=True))
    return type(a) is type(b) and a == b


@pytest.mark.parametrize("value", list(VALUE_FORMS.values()), ids=list(VALUE_FORMS))
def test_each_value_form_reads_back_from_the_folder_and_the_spec_as_it_was_written(value):
    """study.json's form (through JSON) and the spec file's (through YAML) read back as the value the study was made
    with: a tuple of bare numbers comes back the tuple, not a dimensionless quantity, so the folder's coordinates are the
    study's and the spec makes the same study."""
    record = json.loads(json.dumps(encode_value(value, "an.axis")))
    assert same_form(decode_value(record), value), (record, decode_value(record))
    text = yaml.safe_load(yaml.safe_dump({"values": [value_text(value, "an.axis")]}))["values"][0]
    assert same_form(value_of(text, "an.axis"), value), (text, value_of(text, "an.axis"))


#: Gate 5's studies, one a form of axis: two components of a length vector taken together; the whole vector; a
#: direction whole and by a component (bare numbers, built per value); a body's initial orientation (bare numbers, a
#: leaf of the state, not built).
ROUND_TRIPS = {
    "components": (lambda: free_space("egg", {"egg": Ellipsoid(semi_axes=mm(EGG))}, uncoupled(mm(1.0))),
                   {f"{SEMI_AXES}[2]": [mm(1.1), mm(1.25)], f"{SEMI_AXES}[0]": [mm(0.5), mm(0.55)]}),
    "whole": (lambda: free_space("egg", {"egg": Ellipsoid(semi_axes=mm(EGG))}, uncoupled(mm(1.0))),
              {SEMI_AXES: [mm(EGG), mm((0.55, 0.65, 1.2))]}),
    "direction": (mrs_plane, {"floor.normal": [NORMAL, UP]}),
    "direction-component": (mrs_plane, {"floor.normal[1]": [0.6, 0.5]}),
    "orientation": (lambda: free_space("egg", {"egg": Ellipsoid(semi_axes=mm(EGG))}, uncoupled(mm(1.0))),
                    {"egg.initial.orientation": [(1.0, 0.0, 0.0, 0.0), (HALF_TURN, HALF_TURN, 0.0, 0.0)]}),
}


@pytest.mark.parametrize("case", list(ROUND_TRIPS))
def test_a_study_over_built_axes_saves_and_reads_back(case, tmp_path, monkeypatch, float64_gate):
    """The folder read back equals the result in memory, the spec file names the axes' paths with values that read
    back as the study's, a study made from it is the same study (its id and its groups' keys), and a second run into the
    folder runs no member again."""
    from folder_rigs import same_study

    from mime.results._study_folder import STUDY_FILE

    make, axes = ROUND_TRIPS[case]
    study = Study(make()).sweep(along(axes), duration=ms(2.0))
    memory = study.run()
    folder = tmp_path / "folder"
    returned = study.run_to(folder)
    read_back = StudyResult.read(folder)
    assert same_study(memory, returned) == [] and same_study(memory, read_back) == []
    for path in axes:
        mine, theirs = memory.coordinates[path], read_back.coordinates[path]
        assert same_form(mine, theirs) if isinstance(mine, Quantity) else \
            all(same_form(a, b) for a, b in zip(mine, theirs, strict=True)), (path, mine, theirs)
    document = yaml.safe_load((folder / "study.mime.yaml").read_text())
    assert list(document["axes"]) == list(axes)
    for path, values in axes.items():
        read = [value_of(text, path) for text in document["axes"][path]["values"]]
        assert all(same_form(a, b) for a, b in zip(read, values, strict=True)), (path, document["axes"][path])
    again = Study.from_spec(folder / "study.mime.yaml")
    assert [g.key for g in again.plan().groups] == [g.key for g in study.plan().groups]
    import mime.study._folder_run as folder_run

    asked: list[int | None] = []
    real = folder_run.run_group

    def watching(plan: Any, number: Any, record: Any, every: Any, members: Any, *, todo: Any = None,
                 after: Any = None) -> Any:
        asked.append(len(todo) if todo is not None else None)
        return real(plan, number, record, every, members, todo=todo, after=after)

    monkeypatch.setattr(folder_run, "run_group", watching)
    held = StudyResult.read(folder).record
    rerun = Study.from_spec(folder / "study.mime.yaml").run_to(folder)
    assert StudyResult.read(folder).record["id"] == held["id"] and same_study(memory, rerun) == []
    assert all(n == 0 for n in asked), f"a resumed run ran members again: {asked}"
    assert (folder / STUDY_FILE).is_file()
