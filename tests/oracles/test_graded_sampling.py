"""Graded sampling on the MRS rung (drag 0.5.0's ``grading`` and ``contact``, design #296), against exact answers and
its controls.

- **The sphere's wall contact, held facing the plane** (``contact=(0, 0, -1)``, graded for half a force spacing),
  against Brenner's exact normal resistance at half and one force spacing: at blob 0.1 the graded normal entry lies
  within 7 % at half a spacing and nearer the exact answer than the uniform sampling's at both gaps (the gates' line,
  predicted before it ran; the uniform one is +14.0 % and +3.6 %, W1's map).
- **The wrong-way control** (coarser at the contact, the uniform count) at blob 0.025, where the gates measured it worse
  than uniform at both gaps every time it ran (nothing is drawn): it fails the line the graded sampling passes.
- **The flag when the body's nearest point leaves the patch** (``off_patch``): the graded sphere held at the closest gap
  raises none at its held orientation and turned 30 degrees, and raises it turned 60, 90 and 180 degrees, where its
  nearest point lies outside the patch (its edge near 46 degrees); a step two force spacings off raises none at any turn
  (outside the near-wall band). The uniform sampling's node holds no such field.
- **The plan's refusals**, each by its code, and the settings' own.
- **A graded experiment saves and loads** with its settings, and builds the same sampling again.

Float64 (``tests/oracles/conftest.py``)."""

import math

import numpy as np
import pytest
from graded_cases import RADIUS, SPACING, VISCOSITY, cylinder, sphere, wrong_way
from mrs_cases import BALL, free_space, mrs
from plane_cases import drag_node, stepped
from plane_oracles import brenner

from mime.build import BuildRefused
from mime.dynamics import RigidInertial
from mime.experiment import Experiment
from mime.interactions import Drag
from mime.scene import KEPT, Cylinder, Scene, Z
from mime.things import Body, Medium, Part, State
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


def test_the_wrong_way_control_fails_the_line_the_graded_contact_passes():
    """At blob 0.025 the contact sampled coarser than uniform, at the uniform count, misses Brenner's answer by more
    than the uniform sampling at both gaps."""
    uniform = sphere(0.5, blob=0.025, graded=False).build()
    with wrong_way():
        inverted = sphere(0.5, blob=0.025, graded=True).build()
    assert drag_node(inverted).sample_count == drag_node(uniform).sample_count
    for gap in GAPS:
        assert abs(normal_error(inverted, gap)[0]) > abs(normal_error(uniform, gap)[0]), gap


@pytest.mark.parametrize(("turn", "raised"), [(0.0, 0.0), (30.0, 0.0), (60.0, 1.0), (90.0, 1.0), (180.0, 1.0)])
def test_the_flag_rises_where_the_nearest_point_leaves_the_patch(graded_sphere, turn, raised):
    _, near = normal_error(graded_sphere, 0.5, turn)
    assert float(near["off_patch"]) == raised
    _, far = normal_error(graded_sphere, 2.0, turn)
    assert float(far["off_patch"]) == 0.0 and float(far["near_wall"]) == 0.0


def test_the_flags_raised_steps_lie_farther_from_brenners_answer_than_the_held_pose(graded_sphere):
    held, _ = normal_error(graded_sphere, 0.5)
    for turn in (60.0, 90.0, 180.0):
        assert abs(normal_error(graded_sphere, 0.5, turn)[0]) > abs(held), turn


def test_the_uniform_samplings_node_holds_no_patch_flag(uniform_sphere):
    assert "off_patch" not in stepped(uniform_sphere)
    assert not drag_node(uniform_sphere).graded


def test_a_graded_body_in_free_space_holds_the_flag_at_zero():
    can = Cylinder(radius=mm(1.0), length=mm(2.0), axis=(0.0, 0.0, 1.0))
    built = free_space("graded_free", {"can": can}, mrs(mm(0.6), grading="features")).build()
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
    assert codes(free_space("smooth", {"ball": BALL}, mrs(mm(0.5), grading="features"))) == {"drag.grading-features"}


def test_a_kept_version_is_refused_grading():
    kept = next(cls for (name, _), (cls, _) in KEPT.items() if name == "sphere")
    assert codes(free_space("kept", {"ball": kept(radius=mm(1.0))},
                            mrs(mm(0.5), contact=(0.0, 0.0, -1.0), contact_gap=mm(0.1)))) >= {"drag.grading-kind"}


def test_a_contact_in_free_space_is_refused():
    assert "drag.grading-contact" in codes(free_space("free", {"ball": BALL},
                                                      mrs(mm(0.5), contact=(0.0, 0.0, -1.0), contact_gap=mm(0.1))))


def test_a_contact_with_no_gap_to_grade_for_is_refused():
    with pytest.raises(BuildRefused) as refused:
        sphere(0.5, blob=0.1, graded=False, contact=(0.0, 0.0, -1.0)).plan()
    assert refused.value.codes() == ["drag.grading-approach"]
    assert "contact_gap=" in refused.value.refusals[0].action


@pytest.mark.parametrize(("settings", "said"), [
    ({"grading": "edges"}, "not one of"),
    ({"contact": (0.0, 0.0, 0.0)}, "not a direction"),
    ({"contact_gap": mm(0.1)}, "no contact= is written"),
])
def test_the_settings_own_problems_are_refused(settings, said):
    with pytest.raises(BuildRefused) as refused:
        free_space("bad", {"ball": BALL}, mrs(mm(0.5), **settings)).plan()
    assert any(said in refusal.message for refusal in refused.value.refusals)


def test_the_rft_rung_reads_no_grading():
    with pytest.raises(BuildRefused) as refused:
        free_space("rft", {"rod": Cylinder(radius=mm(0.1), length=mm(2.0), axis=(1.0, 0.0, 0.0))},
                   Drag.RFT(wall_kind="free_space", grading="features")).plan()
    assert any("grading= is a setting of the mrs rung" in refusal.message for refusal in refused.value.refusals)


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
                                   grading="features" if name == "a" else None), body=name, medium="water")
    assert "drag.grading-near-field" in codes(experiment)


def test_a_graded_experiment_saves_and_loads_with_its_settings(tmp_path):
    experiment = cylinder(0.5, blob=0.1, graded=True, contact=(0.0, -1.0, 0.0), contact_gap=mm(0.12))
    built = experiment.build()
    built.save(tmp_path / "graded")
    again = Experiment.load(tmp_path / "graded")
    settings = [entry["settings"] for entry in again.build().record.resolved["entries"].values()
                if entry["name"] == "drag"][0]
    assert settings["grading"] == "features" and tuple(settings["contact"]) == (0.0, -1.0, 0.0)
    node, rebuilt = drag_node(built), drag_node(again.build())
    assert node.sample_count == rebuilt.sample_count
    np.testing.assert_array_equal(node.constants()["sampling_chosen"], rebuilt.constants()["sampling_chosen"])
    np.testing.assert_array_equal(node.constants()["graded_patch"], rebuilt.constants()["graded_patch"])
