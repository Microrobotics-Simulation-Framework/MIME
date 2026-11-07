"""A body's inertia as its full tensor (phase I1, F0-3 items 1 to 5): the parts give the 3 x 3 with its products, and
every profile builds it as it is (``parts.inertia-axes`` and exploratory's waiver of the products are gone);
``rigid_inertial`` 0.4.0's node takes the nine numbers; a declared tensor is carried as declared and refused at build
where it is not symmetric or not positive definite; research's option of declaring the diagonal with a reason stays.

**Controls.** A plain body (a sphere) gives a tensor whose products are exactly zero, so a product seen here is the
parts', not the resolver's rounding. ``rigid_inertial`` 0.3.0, kept for saved folders, refuses a tensor with products
(and takes one whose products it took as none as its diagonal), so the products reach the entry and are not dropped on
the way; the kept versions build their own nodes from three moments."""

import math

import numpy as np
import pytest
from hull_cases import SPACING, placed_at, turned
from profile_cases import parts_screw

from mime import Body, Experiment, State, Structure
from mime.build import BuildRefused
from mime.dynamics import RigidInertial
from mime.dynamics.body import KEPT, PRODUCTS_TAKEN_AS_NONE, RigidInertialExplicit, RigidInertialMoments
from mime.nodes.body import RigidInertialMidpointNode, RigidInertialNode, RigidInertialTensorNode
from mime.interactions import Contact
from mime.scene import Scene, Screw, Sphere, Tube, X, Z
from mime.things import Part
from mime.units import dimensionless, kg, kg_m2, kg_per_m3, m, m_per_s, m_per_s2, mm, ms, rad_per_s, s

AT_REST = dict(position=mm((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
               angular_velocity=rad_per_s((0.0, 0.0, 0.0)))


def node_inertia(built, body: str = "umr") -> np.ndarray:
    return np.asarray(built.graph.get_node(body).constants()["inertia"], dtype=np.float64)


def declared_body(inertia, dynamics=None) -> Experiment:
    exp = Experiment("declared", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(1.0))
    exp.add_body(Body("rod", dynamics=dynamics or RigidInertial(), initial=State(**AT_REST)))
    exp.declare("rod.mass", kg(1e-6), reason="a millimetre rod")
    exp.declare("rod.inertia", kg_m2(inertia), reason="a millimetre rod's tensor, the test's")
    return exp


@pytest.mark.parametrize("profile", ["exploratory", "research"])
def test_every_profile_builds_the_parts_full_tensor_with_its_products_and_stamps_nothing_for_them(profile):
    built = parts_screw(profile).build()
    tensor = node_inertia(built).reshape(3, 3)
    assert type(built.graph.get_node("umr")) is RigidInertialTensorNode
    assert np.array_equal(tensor, tensor.T) and np.all(np.linalg.eigvalsh(tensor) > 0.0)
    assert tensor[1, 2] != 0.0 and tensor[0, 1] == 0.0 == tensor[0, 2], "FL-9's shape turns about its axis only"
    resolved = built.record.resolved["values"]["umr.inertia"]
    assert resolved["source"] == "derived from its parts" and len(resolved["value"]) == 9
    assert np.array_equal(np.asarray(resolved["value"]), tensor.reshape(9))
    profile_record = built.record.resolved["profile"]
    assert not any(item["target"] == "umr.inertia" for item in [*profile_record["pending"], *profile_record["listed"]])


def test_research_may_declare_the_diagonal_with_a_reason():
    tensor = node_inertia(parts_screw("research").build()).reshape(3, 3)
    exp = parts_screw("research")
    exp.declare("umr.inertia", kg_m2(tuple(float(c) for c in np.diag(tensor))),
                reason="the parts' diagonal: a modelling choice, the products left out")
    built = exp.build()
    assert np.array_equal(node_inertia(built).reshape(3, 3), np.diag(np.diag(tensor)))
    source = built.record.resolved["values"]["umr.inertia"]["source"]
    named = source.removeprefix("declared in the experiment, in place of the value its parts give, (")
    assert named != source and named.endswith(") kg m^2") and len(named.split(", ")) == 9, \
        "the parts' nine numbers, named in the source"


def test_control_a_sphere_gives_a_tensor_with_no_products():
    exp = Experiment("ball", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(1.0))
    exp.add_body(Body("ball", dynamics=RigidInertial(), initial=State(**AT_REST),
                      parts={"shell": Part(Sphere(radius=mm(1.0)))}))
    exp.declare("ball.parts.shell.density", kg_per_m3(1000.0), reason="a test sphere")
    tensor = node_inertia(exp.build(), "ball").reshape(3, 3)
    assert np.all(tensor[~np.eye(3, dtype=bool)] == 0.0) and np.all(np.diag(tensor) > 0.0)


def test_a_declared_tensor_is_carried_as_declared_and_three_moments_as_their_diagonal():
    tensor = (1e-13, 2e-15, -1e-15, 2e-15, 2e-13, 3e-15, -1e-15, 3e-15, 3e-13)
    built = declared_body(tensor).build()
    assert node_inertia(built, "rod").tolist() == list(tensor)
    assert built.record.resolved["values"]["rod.inertia"]["value"] == list(tensor)
    moments = declared_body((1e-13, 2e-13, 3e-13))
    built = moments.build()
    assert node_inertia(built, "rod").tolist() == [1e-13, 0.0, 0.0, 0.0, 2e-13, 0.0, 0.0, 0.0, 3e-13]
    assert built.record.resolved["values"]["rod.inertia"]["value"] == [1e-13, 2e-13, 3e-13], "as declared"


@pytest.mark.parametrize("tensor, said", [
    ((1e-13, 2e-15, 0.0, -2e-15, 2e-13, 0.0, 0.0, 0.0, 3e-13), "is not symmetric"),
    ((1e-13, 2e-13, 0.0, 2e-13, 1e-13, 0.0, 0.0, 0.0, 3e-13), "is not positive definite"),
    ((1e-13, -2e-13, 3e-13), "are not all above zero"),
])
def test_a_tensor_that_is_not_symmetric_or_not_definite_is_refused_at_build(tensor, said):
    with pytest.raises(BuildRefused) as refused:
        declared_body(tensor).build()
    (found,) = refused.value.refusals
    assert found.code == "value.not-positive" and said in found.message and "exp.declare" in found.action


def kept_screw() -> Experiment:
    """``parts_screw``'s body, its parts' tensor with its product, on the kept ``rigid_inertial`` 0.3.0."""
    exp = Experiment("screw", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(1.0))
    exp.add_body(Body("umr", dynamics=RigidInertialMoments(), initial=State(**AT_REST), parts={"shell": Part(Screw(
        radius=mm(1.56), length=mm(7.47), wavenumber=dimensionless(2.33), modulation=dimensionless(0.33),
        starts=2))}))
    exp.declare("umr.parts.shell.density", kg_per_m3(1410.0), reason="FL-9 resin plus magnets, mean density")
    return exp


def test_control_the_kept_0_3_0_refuses_the_parts_products_so_they_reach_the_entry():
    tensor = node_inertia(parts_screw("exploratory").build()).reshape(3, 3)
    assert abs(tensor[1, 2]) / np.max(np.diag(tensor)) > PRODUCTS_TAKEN_AS_NONE
    with pytest.raises(BuildRefused) as refused:
        kept_screw().build()
    (found,) = refused.value.refusals
    assert found.code == "value.products-of-inertia" and "products of inertia" in found.message
    assert "dynamics=RigidInertial()" in found.action


def test_the_kept_versions_take_a_tensor_with_products_they_took_as_none_as_its_diagonal():
    below = (1e-10, 0.0, 0.0, 0.0, 2e-10, 1e-23, 0.0, 1e-23, 3e-10)
    for kept in (RigidInertialMoments(), RigidInertialExplicit()):
        assert kept.value_problems({"inertia": below}) == []
        assert kept.node_inertia(below) == (1e-10, 2e-10, 3e-10)
        assert kept.node_inertia((1e-10, 2e-10, 3e-10)) == (1e-10, 2e-10, 3e-10)


def test_the_kept_versions_build_their_own_nodes_from_three_moments():
    assert KEPT[("rigid_inertial", "0.3.0")] == (RigidInertialMoments, RigidInertial.version)
    assert KEPT[("rigid_inertial", "0.2.0")] == (RigidInertialExplicit, RigidInertialMoments.version)
    for dynamics, node in ((RigidInertialMoments(), RigidInertialMidpointNode),
                           (RigidInertialExplicit(), RigidInertialNode)):
        built = declared_body((1e-13, 2e-13, 3e-13), dynamics).build()
        assert type(built.graph.get_node("rod")) is node
        assert node_inertia(built, "rod").tolist() == [1e-13, 2e-13, 3e-13]


def spinning_screw(inertia=None) -> Experiment:
    """FL-9's shape alone in free space, torque-free, spinning about axes that mix its transverse ones, its density
    declared and its inertia the parts' own, or ``inertia`` declared in its place."""
    exp = Experiment("spinning_screw", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(1.0))
    exp.add_body(Body("umr", dynamics=RigidInertial(), initial=State(
        position=mm((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((30.0, 60.0, 40.0))), parts={"shell": Part(Screw(
            radius=mm(1.56), length=mm(7.47), wavenumber=dimensionless(2.33), modulation=dimensionless(0.33),
            starts=2))}))
    exp.declare("umr.parts.shell.density", kg_per_m3(1410.0), reason="FL-9 resin plus magnets, mean density")
    if inertia is not None:
        exp.declare("umr.inertia", kg_m2(inertia), reason="the parts' diagonal, the products left out")
    return exp


def test_a_body_whose_parts_give_products_runs_end_to_end_and_the_products_turn_it():
    """The first released-path run with products of inertia (a run, not a node or a saved fixture): FL-9's shape,
    torque-free, spinning across its transverse axes for 0.2 s, its inertia the parts' own. The node is 0.4.0's, the
    tensor it holds has the parts' product, and the run is finite. **The control:** the same body with the parts'
    diagonal declared in the tensor's place, nothing else changed, ends at another attitude (0.028 apart here, the
    product being 0.1 % of the moment it couples), so the run did carry the product. No contact is attached: the
    centre-held contact would flag FL-9's tensor (the containment watch) and the hull contact's compile alone is over
    the test budget."""
    built = spinning_screw().build()
    tensor = node_inertia(built).reshape(3, 3)
    assert type(built.graph.get_node("umr")) is RigidInertialTensorNode and tensor[1, 2] != 0.0
    carried = built.run(s(0.2), history=True)
    left_out = spinning_screw(tuple(float(c) for c in np.diag(tensor))).build().run(s(0.2), history=True)
    pose, control = (np.asarray(each.history["umr"]["pose"]) for each in (carried, left_out))
    assert np.all(np.isfinite(pose)) and np.all(np.isfinite(np.asarray(carried.history["umr"]["twist"])))
    assert carried.containment == () and not [read.path for read in carried.flags if read.raised]
    assert np.max(np.abs(pose[:, 3:] - control[:, 3:])) > 1e-3, "the product turned the body"


def test_the_saved_forms_copy_of_the_line_0_3_0_drew_is_the_kept_versions():
    """``mime.saved._versions`` cannot import ``mime.dynamics``; its copy of the line is held equal here."""
    from mime.saved import _versions

    assert _versions._PRODUCTS_TAKEN_AS_NONE == PRODUCTS_TAKEN_AS_NONE  # pyright: ignore[reportPrivateUsage]


TUBE_RADIUS, STEPS, ONE_ULP_OF_30 = 3.175e-3, 3, float(np.nextafter(np.float32(30.0), np.float32(np.inf)))


def screw_held_in_tube(inertia=None, spin_x: float = 30.0, contact: bool = True) -> Experiment:
    """``spinning_screw``'s body in script A's tube, turned 12 degrees across it and lowered until its hull's nearest
    point is 1e-7 m from the wall less the margin, so the hull contact holds it from the first step; ``spin_x``, the
    angular velocity's x component (rad/s), for the resolution's one-ulp change; ``inertia`` as in ``spinning_screw``;
    ``contact=False``, the same body free in the tube."""
    quaternion = turned(math.radians(12.0))
    start = placed_at(1e-7, quaternion)
    exp = Experiment("screw_held_in_tube", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(1.0))
    tube = exp.add_structure(Structure("tube", geometry=Tube(axis=+X, inner_radius=m(TUBE_RADIUS), wall=mm(1.6),
                                                             length=mm(100.0))))
    exp.add_body(Body("umr", dynamics=RigidInertial(), initial=State(
        position=m(tuple(float(c) for c in start)), orientation=quaternion, velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((spin_x, 60.0, 40.0))), parts={"shell": Part(Screw(
            radius=mm(1.56), length=mm(7.47), wavenumber=dimensionless(2.33), modulation=dimensionless(0.33),
            starts=2))}))
    exp.declare("umr.parts.shell.density", kg_per_m3(1410.0), reason="FL-9 resin plus magnets, mean density")
    if inertia is not None:
        exp.declare("umr.inertia", kg_m2(inertia), reason="the parts' diagonal, the products left out")
    if contact:
        exp.attach(Contact.Projection(margin=m(1e-4), spacing=m(SPACING)), body="umr", surface=tube.inner_surface)
    return exp


def test_a_body_whose_parts_give_products_runs_under_the_hull_contact_and_the_products_reach_the_run():
    """The hull contact's first end-to-end run of a tensor with products (the follow-up to #186), the body being
    ``parts_screw``'s: FL-9, whose parts give a product of inertia of 4.98e-13 kg m^2, 0.107 % of its largest moment
    (4.63e-10), torque-free in script A's tube, spinning across its transverse axes, held by its hull's points from the
    first step. Three steps of 1 ms: the hull's nearest point starts 1e-7 m inside the margin, so the contact acts from the
    first step; the products' effect on the pose grows with each step, and three put it well above the resolution.
    The containment watch reads nothing for a hull the contact holds; the only flag raised is that the
    contact touched.

    **Controls.** (1) The same body with the parts' diagonal declared in the tensor's place ends apart from it by more
    than a hundred times the run's own resolution, which is the pose's divergence under a one-ulp change of the start's
    angular velocity (measured: 7.5e-9, against 3.0e-5 for the diagonal, 4 000 times it, in float32 on CPU). (2) The
    same body without the contact ends 0.046 from the held one (six million times the resolution), so the contact is in
    play. A resolution that measures zero would make the first control prove nothing, and is refused."""
    built = screw_held_in_tube().build()
    tensor = node_inertia(built).reshape(3, 3)
    assert type(built.graph.get_node("umr")) is RigidInertialTensorNode
    resolved = built.record.resolved["values"]["umr.inertia"]
    assert resolved["source"] == "derived from its parts"
    assert np.array_equal(np.asarray(resolved["value"]), tensor.reshape(9))
    assert tensor[1, 2] != 0.0 and abs(tensor[1, 2]) / np.max(np.diag(tensor)) > PRODUCTS_TAKEN_AS_NONE

    carried = built.run(ms(STEPS), history=True)
    diagonal = tuple(float(c) for c in np.diag(tensor))
    left_out = screw_held_in_tube(diagonal).build().run(ms(STEPS), history=True)
    nudged = screw_held_in_tube(spin_x=ONE_ULP_OF_30).build().run(ms(STEPS), history=True)
    free = screw_held_in_tube(contact=False).build().run(ms(STEPS), history=True)
    pose, control, one_ulp, uncontacted = (np.asarray(each.history["umr"]["pose"])
                                           for each in (carried, left_out, nudged, free))

    assert pose.shape == (STEPS, 7) and np.all(np.isfinite(pose))
    assert np.all(np.isfinite(np.asarray(carried.history["umr"]["twist"])))
    assert carried.containment == ()
    raised = {read.path for read in carried.flags if read.raised}
    assert raised == {"umr.touching"}, "the contact held the hull and nothing else was raised"

    resolution = float(np.max(np.abs(one_ulp - pose)))
    assert resolution > 0.0, "a one-ulp change that moves nothing cannot be the run's resolution"
    assert np.max(np.abs(control - pose)) > 100.0 * resolution, "the products reached the run"
    assert np.max(np.abs(uncontacted - pose)) > 100.0 * resolution, "the contact was in play"
