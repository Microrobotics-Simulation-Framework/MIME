"""A body's inertia as its full tensor (phase I1, F0-3 items 1 to 5): the parts give the 3 x 3 with its products, and
every profile builds it as it is (``parts.inertia-axes`` and exploratory's waiver of the products are gone);
``rigid_inertial`` 0.4.0's node takes the nine numbers; a declared tensor is carried as declared and refused at build
where it is not symmetric or not positive definite; research's option of declaring the diagonal with a reason stays.

**Controls.** A plain body (a sphere) gives a tensor whose products are exactly zero, so a product seen here is the
parts', not the resolver's rounding. ``rigid_inertial`` 0.3.0, kept for saved folders, refuses a tensor with products
(and takes one whose products it took as none as its diagonal), so the products reach the entry and are not dropped on
the way; the kept versions build their own nodes from three moments."""

import numpy as np
import pytest
from profile_cases import parts_screw

from mime import Body, Experiment, State
from mime.build import BuildRefused
from mime.dynamics import RigidInertial
from mime.dynamics.body import KEPT, PRODUCTS_TAKEN_AS_NONE, RigidInertialExplicit, RigidInertialMoments
from mime.nodes.body import RigidInertialMidpointNode, RigidInertialNode, RigidInertialTensorNode
from mime.scene import Scene, Screw, Sphere, Z
from mime.things import Part
from mime.units import dimensionless, kg, kg_m2, kg_per_m3, m_per_s, m_per_s2, mm, ms, rad_per_s, s

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


def test_a_body_whose_parts_give_products_runs_end_to_end_under_the_hull_contact_with_nothing_contained_or_flagged():
    """The first released-path run with products of inertia: FL-9 in script A's tube, held at its hull's points from
    the first step (the hull contact, so the containment watch of the centre-held contact does not apply: FL-9's tensor
    under that contact turns the screw across the tube within 0.35 s), its inertia the parts' own, 20 steps of 0.5 ms.
    The node is 0.4.0's, the tensor it holds has the parts' product, and the run is finite, with nothing flagged but
    the touch."""
    from hull_cases import hull_contact

    experiment = hull_contact("products")
    experiment.remove("umr.inertia", kind="value")
    experiment.declare("umr.parts.shell.density", kg_per_m3(1410.0), reason="FL-9 resin plus magnets, mean density")
    built = experiment.build()
    node = built.graph.get_node("umr")
    tensor = node_inertia(built).reshape(3, 3)
    assert type(node) is RigidInertialTensorNode and tensor[1, 2] != 0.0
    result = built.run(s(0.01), history=True)
    assert result.containment == ()
    assert {read.path for read in result.flags if read.raised} <= {"umr.touching"}, "the hull is at the wall by design"
    for name in ("pose", "twist"):
        assert np.all(np.isfinite(np.asarray(result.history["umr"][name])))
    assert np.max(np.abs(np.asarray(result.history["umr"]["twist"])[:, 3:])) > 1.0, "the field turned the body"


def test_the_saved_forms_copy_of_the_line_0_3_0_drew_is_the_kept_versions():
    """``mime.saved._versions`` cannot import ``mime.dynamics``; its copy of the line is held equal here."""
    from mime.saved import _versions

    assert _versions._PRODUCTS_TAKEN_AS_NONE == PRODUCTS_TAKEN_AS_NONE  # pyright: ignore[reportPrivateUsage]
