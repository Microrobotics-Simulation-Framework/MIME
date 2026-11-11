"""Graded sampling on the MRS rung (drag 0.5.0's ``grading`` and ``contact``, design #296), against exact answers and
its controls.

- **The sphere's wall contact, held facing the plane** (``contact=(0, 0, -1)``, graded for half a force spacing),
  against Brenner's exact normal resistance at half and one force spacing: at blob 0.1 the graded normal entry lies
  within 7 % at half a spacing and nearer the exact answer than the uniform sampling's at both gaps (the gates' line,
  predicted before it ran; the uniform one is +14.0 % and +3.6 %, W1's map).
- **The count-matched uniform arm** (design #329, the standard control for a sampling claim): a uniform sampling of the
  graded count lies farther from Brenner's answer at half a spacing than the graded contact, so it helps beyond its
  count.
- **The wrong-way control** (coarser at the contact, the uniform count) at blob 0.025, where the gates measured it worse
  than uniform at both gaps every time it ran (nothing is drawn): it fails the line the graded sampling passes.
- **The flag when the body's nearest points leave the patch** (``off_patch``), the patch judged at each point of the
  separated quadrature and every point within ``OFF_PATCH_BAND`` force spacings of the least gap read: the graded
  sphere held at the closest gap raises none at its held orientation and turned 30 degrees, and raises it turned 46,
  48, 60, 90 and 180 degrees, where its nearest point lies outside the patch (its edge at 45.6 degrees), and turned 40
  degrees, where only the band reaches past the edge (the band's pin: the flag judged at the nearest point alone stays
  silent there); a box's face turned 89 to 90 degrees towards the wall, a tie or nearly one, raises it, and the box
  held or turned up to 80 degrees raises none. The flag's own band, one force spacing: raised at 0.7 spacings, none
  at 1.2 or two. The uniform sampling's node holds no such field.
- **The wall's measure in a graded run** (decision #329, superseding #308; ``GRADED_WALL_MEASURE``): the graded
  sphere's integrated normal entry at half a spacing, blob 0.1, pinned to the value measured for the measure the switch
  selects (+3.333 % for the force spacing, as shipped; +5.243 % for the cell's own size), so that the measure cannot
  change without a red.
- **The contact's frame**: the direction written is the body's, turned into the kind's frame by the part's rotation's
  transpose, so that on a turned part the patch lies on the side that faces the wall in the body's frame.
- **The plan's refusals**, each by its code, and the settings' own.
- **A graded experiment saves and loads** with its settings, and builds the same sampling again.

Float64 (``tests/oracles/conftest.py``)."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from graded_cases import RADIUS, SPACING, VISCOSITY, above_plane, cylinder, sphere, wrong_way
from mrs_cases import BALL, free_space, mrs
from plane_cases import drag_node, stepped
from plane_oracles import brenner

from mime.build import BuildRefused
from mime.build._maddening import parameter_leaves
from mime.core import quaternion
from mime.dynamics import RigidInertial
from mime.experiment import Experiment
import mime.nodes.hydro.mrs._node as mrs_node
from mime.interactions import Contact, Drag, ForceSampling
from mime.nodes.hydro.mrs._node import GRADED_WALL_MEASURE
from mime.registry import Hull
from mime.scene import KEPT, Box, Capsule, Cylinder, Scene, Sphere, Tube, Z
from mime.scene._grading import graded_patch
from mime.scene._sampling import quadrature_arrays, quadrature_counts
from mime.things import Body, Medium, Part, State, Structure
from mime.units import kg, kg_m2, m, m_per_s, m_per_s2, mm, mPa_s, ms, rad_per_s

GAPS = (0.5, 1.0)


def normal_error(built, gap: float, turn: float = 0.0) -> tuple[float, dict]:
    """The normal resistance's error against Brenner's at ``gap`` force spacings, the sphere turned ``turn`` degrees
    about x, and the step's state."""
    half = math.radians(turn) / 2
    out = stepped(built, pose=(0.0, 0.0, RADIUS + gap * SPACING, math.cos(half), math.sin(half), 0.0, 0.0))
    exact = 6 * math.pi * VISCOSITY * RADIUS * brenner(gap * SPACING / RADIUS)
    return float(out["resistance"][2, 2] / exact - 1.0), out


@pytest.fixture(scope="module")
def uniform_sphere():
    return sphere(0.5, blob=0.1, graded=False).build()


@pytest.fixture(scope="module")
def graded_sphere():
    return sphere(0.5, blob=0.1, graded=True).build()


def test_the_graded_contact_lies_nearer_brenners_answer_than_the_uniform_sampling(uniform_sphere, graded_sphere):
    for gap in GAPS:
        uniform, _ = normal_error(uniform_sphere, gap)
        graded, _ = normal_error(graded_sphere, gap)
        assert abs(graded) < abs(uniform), (gap, graded, uniform)
    assert abs(normal_error(graded_sphere, 0.5)[0]) < 0.07
    assert drag_node(graded_sphere).sample_count > drag_node(uniform_sphere).sample_count


@pytest.fixture(scope="module")
def count_matched_sphere(uniform_sphere, graded_sphere):
    """The uniform sampling of the graded sphere's count (design #329's count-matched arm): its force spacing the
    module's times the root of the two counts' ratio, its blob the same share of it."""
    ratio = drag_node(uniform_sphere).sample_count / drag_node(graded_sphere).sample_count
    return sphere(0.5, blob=0.1, graded=False, spacing=SPACING * math.sqrt(ratio)).build()


def test_the_graded_contact_lies_nearer_brenners_answer_than_a_uniform_sampling_of_its_count(graded_sphere,
                                                                                            count_matched_sphere):
    """The count-matched uniform arm, the standard control for a sampling claim (design #329): a graded sampling holds
    more force points, so that it helps beyond its count is shown against a uniform sampling of as many, within a tenth
    (measured before: +5.31 % graded against +10.16 % at the graded count, half a force spacing, blob 0.1)."""
    graded, matched = drag_node(graded_sphere).sample_count, drag_node(count_matched_sphere).sample_count
    errors = normal_error(graded_sphere, 0.5)[0], normal_error(count_matched_sphere, 0.5)[0]
    print(f"\ncounts: graded {graded}, count-matched uniform {matched}; errors at half a spacing {errors}")
    assert abs(matched - graded) <= 0.1 * graded
    assert abs(errors[0]) < abs(errors[1])


def test_the_wrong_way_control_fails_the_line_the_graded_contact_passes():
    """At blob 0.025 the contact sampled coarser than uniform, at the uniform count, misses Brenner's answer by more
    than the uniform sampling at both gaps."""
    uniform = sphere(0.5, blob=0.025, graded=False).build()
    with wrong_way():
        inverted = sphere(0.5, blob=0.025, graded=True).build()
    assert drag_node(inverted).sample_count == drag_node(uniform).sample_count
    for gap in GAPS:
        assert abs(normal_error(inverted, gap)[0]) > abs(normal_error(uniform, gap)[0]), gap


@pytest.mark.parametrize(("turn", "raised"), [(0.0, 0.0), (30.0, 0.0), (40.0, 1.0), (46.0, 1.0), (48.0, 1.0),
                                             (60.0, 1.0), (90.0, 1.0), (180.0, 1.0)])
def test_the_flag_rises_where_the_nearest_points_leave_the_patch(graded_sphere, turn, raised):
    _, near = normal_error(graded_sphere, 0.5, turn)
    assert float(near["off_patch"]) == raised
    _, far = normal_error(graded_sphere, 2.0, turn)
    assert float(far["off_patch"]) == 0.0 and float(far["near_wall"]) == 0.0


@pytest.mark.parametrize(("gap", "raised"), [(0.7, 1.0), (1.2, 0.0)])
def test_the_flags_band_is_one_force_spacing_from_the_wall(graded_sphere, gap, raised):
    _, out = normal_error(graded_sphere, gap, 90.0)
    assert float(out["off_patch"]) == raised


#: The box whose bottom face is graded as a held contact: its size (m).
BOX = (0.6e-3, 0.9e-3, 1.3e-3)


def lowest_at(built, turn: float, axis: tuple[float, float, float], gap: float) -> tuple[float, ...]:
    """The pose turned ``turn`` degrees about ``axis`` with the lowest point of the node's quadrature ``gap`` (m) above
    the plane."""
    node = drag_node(built)
    leaves = parameter_leaves(built.graph)["drag:ball"]
    points = np.asarray(node.body({k: leaves[k] for k in node.body_leaf_names()}).points)
    half = math.radians(turn) / 2
    q = (math.cos(half), *(math.sin(half) * c for c in axis))
    rotation = np.asarray(quaternion.rotation_matrix(jnp.asarray(q, dtype=jnp.float64)))
    return (0.0, 0.0, gap - float(np.min((points @ rotation.T)[:, 2])), *q)


#: The box's turns towards the wall and whether each raises the flag: held, or turned within the patch, it raises none;
#: at 89 and 89.5 degrees the least gap is the old bottom edge's, in the patch, while most of the face at the wall is
#: off it, and at 90 the face is a tie.
BOX_TURNS = ((0.0, 0.0), (45.0, 0.0), (80.0, 0.0), (89.0, 1.0), (89.5, 1.0), (90.0, 1.0))


def test_a_box_face_turned_to_the_wall_raises_the_flag_where_a_tie_or_near_tie_leaves_the_patch():
    built = above_plane(Box(size=m(BOX)), (0.0, 0.0, 5e-3), (1.0, 0.0, 0.0, 0.0), blob=0.1, name="graded_box",
                        contact=(0.0, 0.0, -1.0), contact_gap=m(0.5 * SPACING)).build()
    for turn, raised in BOX_TURNS:
        for axis in ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0)):
            out = stepped(built, pose=lowest_at(built, turn, axis, 0.5 * SPACING))
            assert float(out["off_patch"]) == raised, (turn, axis)


#: The graded integrated sphere's normal entry at half a force spacing, blob 0.1, against Brenner, by the wall measure
#: ``GRADED_WALL_MEASURE`` selects (decisions #308 and #329): measured through the node, the uniform sampling +1.466 %
#: there.
MEASURED_308 = {"cell": 0.052435, "force_spacing": 0.033329}


@pytest.mark.parametrize("measure", sorted(MEASURED_308))
def test_each_value_of_the_graded_wall_measures_switch_gives_its_measured_error(measure, monkeypatch):
    """The switch's two values, each built afresh (the step traced with it): the shipped one and the alternative the
    maintainer may flip it to, each at its measured error, so that the code cannot take either measure but the one the
    switch names."""
    monkeypatch.setattr(mrs_node, "GRADED_WALL_MEASURE", measure)
    jax.clear_caches()
    built = sphere(0.5, blob=0.1, graded=True, wall_refinement=2).build()
    error, _ = normal_error(built, 0.5)
    assert abs(error - MEASURED_308[measure]) < 5e-5, (measure, error)
    assert f"distance in {'its own size' if measure == 'cell' else 'the force spacing'}" in built.report.text(details=True)


def test_the_shipped_graded_wall_measure_is_decision_329s():
    """The force spacing, as decided (#329, superseding #308's cell): flipping the switch is the maintainer's, and moves
    this line with it."""
    assert GRADED_WALL_MEASURE == "force_spacing"


def _cells_short(make, dtype: str) -> tuple[int, int, int]:
    """The steps of a six-step run of ``make``'s graded body that raise ``wall_cells_short``, the steps, and its
    budget, in ``dtype``."""
    from mime.units import ms

    held_x64 = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", dtype == "float64")
    try:
        built = make().build()
        drag = next(name for name in built.graph.node_names if name.startswith("drag:"))
        held = np.asarray(built.run(ms(6.0), history=True).history[drag]["wall_cells_short"])
        return int(np.sum(held > 0)), int(held.shape[0]), int(built.graph.get_node(drag).wall_budget)
    finally:
        jax.config.update("jax_enable_x64", held_x64)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("case", ["sphere", "cylinder"])
def test_a_default_graded_run_raises_no_wall_cells_short_its_budget_following_the_graded_density(case, dtype,
                                                                                                 monkeypatch):
    """Decision #353: a graded body's wall budget follows its density, every cell of its graded patch and
    ``WALL_CELL_SHARE`` of the rest, so the gates' graded sphere (half a spacing from a plane, ``wall_refinement`` 2)
    and ``mrs_graded``'s cylinder raise ``wall_cells_short`` on no step; the flag is kept for a real shortfall. The
    control, a budget cut planted (the share of every cell, the patch not counted whole, the budget before #353),
    raises it on every step."""
    from mrs_cases import mrs_graded

    make = (lambda: sphere(0.5, blob=0.1, graded=True, wall_refinement=2)) if case == "sphere" else mrs_graded
    short, steps, budget = _cells_short(make, dtype)
    assert steps == 6 and short == 0, (case, dtype, short, budget)
    monkeypatch.setattr(mrs_node.MRSDragNode, "graded_cells", property(lambda self: 0))
    cut, _, smaller = _cells_short(make, dtype)
    assert smaller < budget and cut == steps, (case, dtype, cut, smaller, budget)


def test_the_contact_is_turned_into_a_turned_parts_frame_so_its_patch_faces_the_wall():
    """A capsule part turned 90 degrees about x (its axis along y in the body), contact=(0, 0, -1) in the body's frame:
    the graded patch lies on the capsule's side facing -z in the body's frame, not +z."""
    half = math.radians(90.0) / 2
    rotation = np.asarray(quaternion.rotation_matrix(jnp.asarray((math.cos(half), math.sin(half), 0.0, 0.0),
                                                                 dtype=jnp.float64)))
    kind = Capsule(radius=m(0.5e-3), height=m(2e-3), axis=(0.0, 1.0, 0.0))
    settings = {"radius": 0.5e-3, "height": 2e-3, "axis": (0.0, 1.0, 0.0)}
    hull = Hull("ball", "shell", kind, settings, (0.0, 0.0, 0.0), tuple(tuple(float(c) for c in row) for row in rotation),
                {})
    drag = Drag.MRS(near_contact=False, spacing=m(0.25e-3), blob=0.25, wall_kind="plane",
                    sampling=ForceSampling(contact=(0.0, 0.0, -1.0), contact_gap=m(0.1e-3)))
    density = drag._density({"sampling.contact_gap": 0.1e-3}, hull, 0.25e-3)  # noqa: SLF001 - the drag's own density
    assert density is not None
    patches = tuple(kind.patches)
    points = quadrature_arrays(kind, settings, patches, quadrature_counts(kind, settings, 0.05e-3, patches))[0]
    inside = graded_patch(density, points)
    height = (points @ rotation.T)[:, 2]
    assert 0 < np.count_nonzero(inside) < len(points)
    assert float(np.max(height[inside])) < 0.0 < float(np.max(height[~inside])), "the patch faces -z in the body"


def test_the_flags_raised_steps_lie_farther_from_brenners_answer_than_the_held_pose(graded_sphere):
    held, _ = normal_error(graded_sphere, 0.5)
    for turn in (60.0, 90.0, 180.0):
        assert abs(normal_error(graded_sphere, 0.5, turn)[0]) > abs(held), turn


def test_the_uniform_samplings_node_holds_no_patch_flag(uniform_sphere):
    assert "off_patch" not in stepped(uniform_sphere)
    assert not drag_node(uniform_sphere).graded


def test_a_graded_body_in_free_space_holds_the_flag_at_zero():
    can = Cylinder(radius=mm(1.0), length=mm(2.0), axis=(0.0, 0.0, 1.0))
    built = free_space("graded_free", {"can": can}, mrs(mm(0.6), sampling=ForceSampling(grading="features"))).build()
    node = built.graph.get_node("drag:can")
    assert node.graded
    from mime.build._maddening import node_state, parameter_leaves

    out = node.update(node_state(built.graph, "drag:can"), {"pose": np.asarray((0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))},
                      1e-3, params=parameter_leaves(built.graph)["drag:can"])
    assert float(out["off_patch"]) == 0.0


def codes(experiment) -> set[str]:
    with pytest.raises(BuildRefused) as refused:
        experiment.plan()
    return set(refused.value.codes())


def test_features_asked_of_a_kind_that_declares_none_are_refused():
    assert codes(free_space("smooth", {"ball": BALL}, mrs(mm(0.5), sampling=ForceSampling(grading="features")))) == {
        "drag.grading-features"}


def test_a_kept_version_is_refused_grading():
    kept = next(cls for (name, _), (cls, _) in KEPT.items() if name == "sphere")
    assert codes(free_space("kept", {"ball": kept(radius=mm(1.0))},
                            mrs(mm(0.5), sampling=ForceSampling(contact=(0.0, 0.0, -1.0), contact_gap=mm(0.1))))) >= {
        "drag.grading-kind"}


def test_a_contact_in_free_space_is_refused():
    assert "drag.grading-contact" in codes(free_space("free", {"ball": BALL}, mrs(mm(0.5), sampling=ForceSampling(
        contact=(0.0, 0.0, -1.0), contact_gap=mm(0.1)))))


def test_a_contact_graded_for_a_gap_it_cannot_refine_is_refused():
    """At 1.5 force spacings and beyond the contact's spacing, its gap over CONTACT_SPACINGS, is no finer than the
    force spacing: nothing is graded. The control: just inside, it plans."""
    with pytest.raises(BuildRefused) as refused:
        sphere(0.5, blob=0.1, graded=False, contact=(0.0, 0.0, -1.0), contact_gap=m(1.5 * SPACING)).plan()
    assert refused.value.codes() == ["drag.grading-gap"]
    sphere(0.5, blob=0.1, graded=False, contact=(0.0, 0.0, -1.0), contact_gap=m(1.4 * SPACING)).plan()


def held_off_the_tube(**sampling) -> Experiment:
    """A ball of 0.5 mm in a tube of 1.5 mm, a contact law holding it 0.1 mm off the tube's wall, its drag grading the
    side facing the wall: ``ForceSampling(contact=(0, 0, -1), **sampling)``."""
    experiment = Experiment("held_off", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(1))
    experiment.add_structure(Structure("tube", geometry=Tube(axis=(1.0, 0.0, 0.0), inner_radius=mm(1.5), wall=mm(0.2),
                                                             length=mm(40.0))))
    experiment.add_medium(Medium("water", fills="tube.lumen"))
    experiment.declare("water.viscosity", mPa_s(1.0), reason="water")
    experiment.add_body(Body("ball", dynamics=RigidInertial(), parts={"shell": Part(Sphere(radius=mm(0.5)))},
                             initial=State(position=mm((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0),
                                           velocity=m_per_s((0.0, 0.0, 0.0)),
                                           angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    experiment.declare("ball.mass", kg(1e-6), reason="a millimetre body's order")
    experiment.declare("ball.inertia", kg_m2((1e-13, 1e-13, 1e-13)), reason="a millimetre body's order")
    experiment.attach(Drag.MRS(spacing=mm(0.2), blob=0.25, wall_kind="tube", near_contact=False,
                               sampling=ForceSampling(contact=(0.0, 0.0, -1.0), **sampling)),
                      body="ball", medium="water")
    experiment.attach(Contact.Projection(margin=mm(0.1)), body="ball", surface="tube.inner_surface")
    return experiment


def test_a_contacts_gap_left_out_is_the_closest_approach_a_contact_law_holds_the_body_off_by():
    """``ForceSampling``'s ``contact_gap``, a default computed inside a group by the drag's own rule
    (``Drag.default_contact_gap``): the plan takes the contact law's margin, records it as the group's setting's
    determined default, and grades for it. The control: the gap written, no default is taken and the written one is
    graded for."""
    plan = held_off_the_tube().plan()
    taken = {default.path: default for default in plan.profile.defaults}
    held = taken["drag:ball.sampling.contact_gap"]
    assert held.kind.value == "determined" and float(held.value.value) == pytest.approx(1e-4, rel=1e-12)
    written = held_off_the_tube(contact_gap=mm(0.05)).plan()
    assert "drag:ball.sampling.contact_gap" not in {default.path for default in written.profile.defaults}


def test_a_contact_with_no_gap_to_grade_for_is_refused():
    with pytest.raises(BuildRefused) as refused:
        sphere(0.5, blob=0.1, graded=False, contact=(0.0, 0.0, -1.0)).plan()
    assert refused.value.codes() == ["drag.grading-approach"]
    assert "contact_gap=" in refused.value.refusals[0].action


@pytest.mark.parametrize(("settings", "said"), [
    ({"grading": "edges"}, "not one of"),
    ({"contact": (0.0, 0.0, 0.0)}, "not a direction"),
    ({"contact_gap": mm(0.1)}, "no contact is written"),
])
def test_the_settings_own_problems_are_refused(settings, said):
    with pytest.raises(BuildRefused) as refused:
        free_space("bad", {"ball": BALL}, mrs(mm(0.5), sampling=ForceSampling(**settings))).plan()
    assert any(said in refusal.message for refusal in refused.value.refusals)


def test_the_rft_rung_reads_no_grading():
    with pytest.raises(BuildRefused) as refused:
        free_space("rft", {"rod": Cylinder(radius=mm(0.1), length=mm(2.0), axis=(1.0, 0.0, 0.0))},
                   Drag.RFT(wall_kind="free_space", sampling=ForceSampling(grading="features"))).plan()
    assert any("sampling= is a setting of the mrs rung" in refusal.message for refusal in refused.value.refusals)


def test_graded_bodies_in_a_shared_medium_are_refused():
    can = Cylinder(radius=mm(0.5), length=mm(1.0), axis=(0.0, 0.0, 1.0))
    experiment = Experiment("pair", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(1))
    experiment.add_medium(Medium("water"))
    experiment.declare("water.viscosity", mPa_s(1.0), reason="water")
    for name, x in (("a", 0.0), ("b", 2e-3)):
        experiment.add_body(Body(name, dynamics=RigidInertial(), parts={"shell": Part(can)},
                                 initial=State(position=m((x, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0),
                                               velocity=m_per_s((0.0, 0.0, 0.0)),
                                               angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
        experiment.declare(f"{name}.mass", kg(1e-6), reason="x")
        experiment.declare(f"{name}.inertia", kg_m2((1e-13, 1e-13, 1e-13)), reason="x")
        experiment.attach(Drag.MRS(spacing=mm(0.4), blob=0.1, wall_kind="free_space",
                                   sampling=ForceSampling(grading="features") if name == "a" else None), body=name,
                          medium="water")
    assert "drag.grading-near-field" in codes(experiment)


def test_a_graded_experiment_saves_and_loads_with_its_settings(tmp_path):
    experiment = cylinder(0.5, blob=0.5, graded=True, contact=(0.0, -1.0, 0.0), contact_gap=mm(0.12))
    built = experiment.build()
    built.save(tmp_path / "graded")
    again = Experiment.load(tmp_path / "graded").build()
    settings = [entry["settings"] for entry in again.record.resolved["entries"].values()
                if entry["name"] == "drag"][0]
    sampled = settings["sampling"]["settings"]
    assert sampled["grading"] == "features" and tuple(sampled["contact"]) == (0.0, -1.0, 0.0)
    node, rebuilt = drag_node(built), drag_node(again)
    assert node.sample_count == rebuilt.sample_count
    np.testing.assert_array_equal(node.constants()["sampling_chosen"], rebuilt.constants()["sampling_chosen"])
    np.testing.assert_array_equal(node.constants()["graded_patch"], rebuilt.constants()["graded_patch"])
