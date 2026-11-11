"""A tube tree's flow as a script declares it: ``TubeTree`` filled by a medium whose flow is the FVM on a periodic box.
The build holds the tree in the box (its points moved into the box's frame, its spacing from the narrowest lumen) and
steps it by IMEX Euler or its steady limit; RK3 on a tree is refused (``flow.tree-scheme``), and so is a tree that does
not repeat over the box's period where the wall's band reads it (``flow.tree-period``), each with its route. The tree's
free ends are named by geometry: ``root`` and ``end_0``, ``end_1``, ... or the names given, each a surface and an
opening of its lumen, reached through the structure's handle. A tree the fillet fuses is refused (row #353)."""

from __future__ import annotations

import numpy as np
import pytest

from mime.build import BuildRefused
from mime.dynamics import FVM, Scheme
from mime.experiment import Experiment
from mime.interactions import BodyForce
from mime.scene import PeriodicBox, Scene, TubeTree, Z
from mime.things import Medium, Structure
from mime.units import Pa_s, kg_per_m3, m_per_s2, mm, ms

#: A vessel narrowing to 0.8 of its radius over half its 1 mm period and back, the parent reaching past both ends.
PERIODIC = dict(points=mm((0.0, 0.0, -0.5, 0.0, 0.0, 0.25, 0.0, 0.0, 0.5, 0.0, 0.0, 0.75, 0.0, 0.0, 1.5)),
                segments=((0, 1), (1, 2), (2, 3), (3, 4)), radii=mm((0.5, 0.5, 0.5, 0.4, 0.4, 0.5, 0.5, 0.5)))


def tree_experiment(tree: TubeTree, scheme=None, length=mm(1.0)) -> Experiment:
    experiment = Experiment("tree", scene=Scene.new(up=Z, units="m"), dt=ms(5.0))
    experiment.add_structure(Structure("vessel", geometry=tree))
    flow = FVM(domain=PeriodicBox(length=length, margin=4), cells_across=8, stokes=True,
               scheme=scheme if scheme is not None else Scheme.Steady())
    experiment.add_medium(Medium("water", fills="vessel.lumen", fields={"flow": flow}))
    experiment.declare("water.viscosity", Pa_s(1e-3), reason="water near 20 C")
    experiment.declare("water.density", kg_per_m3(1000.0), reason="water near 20 C")
    experiment.attach(BodyForce(along=(0.0, 0.0, 1.0), steady=m_per_s2(0.01)), medium="water")
    return experiment


def _codes(refused: pytest.ExceptionInfo[BuildRefused]) -> list[str]:
    return [each.code for each in refused.value.refusals]


def test_the_build_holds_the_tree_in_its_box(float64):
    built = tree_experiment(TubeTree(**PERIODIC)).build()
    c = built.graph.get_node("water.flow").constants()
    assert c["tree_segments"] == ((0, 1), (1, 2), (2, 3), (3, 4))
    assert c["spacing"] == pytest.approx(0.8e-3 / 8, rel=1e-3)
    assert c["cells_z"] * c["spacing"] == pytest.approx(1e-3, rel=1e-12)
    points = np.asarray(c["tree_points"])
    assert np.all(points[:, 0] > 0.0) and np.all(points[:, 1] > 0.0)


def test_rk3_on_a_tree_is_refused_with_its_route():
    with pytest.raises(BuildRefused) as refused:
        tree_experiment(TubeTree(**PERIODIC), scheme=Scheme.RK3()).build()
    assert "flow.tree-scheme" in _codes(refused)


def test_a_tree_that_does_not_repeat_over_the_period_is_refused_with_its_route():
    """The same tree in a box of 2.4 mm: the box's ends lie past the parent's ends, inside their closing balls, which
    do not repeat; a period a tenth longer (1.1 mm) repeats, the parent straight across both ends."""
    tree_experiment(TubeTree(**PERIODIC), length=mm(1.1)).build()
    with pytest.raises(BuildRefused) as refused:
        tree_experiment(TubeTree(**PERIODIC), length=mm(2.4)).build()
    assert "flow.tree-period" in _codes(refused)


def test_the_trees_free_ends_are_named_by_geometry():
    y = dict(points=mm((0.0, 0.0, -3.0, 0.0, 0.0, 0.0, 1.5, 0.0, 2.6, -1.5, 0.0, 2.6)),
             segments=((0, 1), (1, 2), (1, 3)), radii=mm((0.5, 0.5, 0.4, 0.4, 0.3, 0.3)))
    assert TubeTree(**y).end_names() == (("root", 0), ("end_0", 2), ("end_1", 3))
    assert TubeTree(**y, ends=("left", "right")).end_names() == (("root", 0), ("left", 2), ("right", 3))


def test_each_free_end_is_a_surface_its_plane_facing_the_lumen():
    """Each free end is a surface of the tree's own (``surface_names``), the plane normal to its segment at its point,
    its normal into the segment: the root's along the parent, an end's back along its daughter."""
    y = TubeTree(points=mm((0.0, 0.0, -3.0, 0.0, 0.0, 0.0, 3.0, 0.0, 4.0, -3.0, 0.0, 4.0)),
                 segments=((0, 1), (1, 2), (1, 3)), radii=mm((0.5, 0.5, 0.4, 0.4, 0.3, 0.3)), ends=("left", "right"))
    assert y.surface_names() == {"root": "half_space", "left": "half_space", "right": "half_space"}
    settings = {"points": y.points.value, "radii": y.radii.value}
    root = y.surface_parameters("root", settings)
    assert root["normal"] == pytest.approx((0.0, 0.0, 1.0)) and root["offset"] == pytest.approx(-3e-3)
    left = y.surface_parameters("left", settings)
    assert left["normal"] == pytest.approx((-0.6, 0.0, -0.8)) and left["offset"] == pytest.approx(-0.6 * 3e-3 - 0.8 * 4e-3)
    with pytest.raises(KeyError):
        y.surface_parameters("end_0", settings)


def test_the_trees_ends_are_openings_of_its_lumen_reachable_through_its_handle():
    """The tree's ends are its lumen's declared openings (``Geometry.opening_names``), root first, which F2b's open
    trees read; a script reaches each through the structure's handle, and the printed tree lists them."""
    experiment = Experiment("ends", scene=Scene.new(up=Z, units="m"), dt=ms(5.0))
    vessel = experiment.add_structure(Structure("vessel", geometry=TubeTree(**PERIODIC)))
    experiment.add_medium(Medium("water", fills=vessel.lumen,
                                 fields={"flow": FVM(domain=PeriodicBox(length=mm(1.0), margin=4), cells_across=8,
                                                     stokes=True, scheme=Scheme.Steady())}))
    assert (str(vessel.root), str(vessel.end_0)) == ("vessel.root", "vessel.end_0")
    assert TubeTree(**PERIODIC).opening_names() == {"lumen": ("root", "end_0")}
    assert "root" in experiment.tree().text() and "end_0" in experiment.tree().text()
    built = tree_experiment(TubeTree(**PERIODIC))
    (flow,) = [entry for entry in built.plan().entries if "medium" in entry.regions]
    assert flow.regions["medium"].openings == ("vessel.root", "vessel.end_0")


def test_an_end_named_as_a_region_is_refused():
    with pytest.raises(ValueError, match="a region's name"):
        TubeTree(**{**PERIODIC, "ends": ("lumen",)})


#: Two parallel tubes sharing no point, each meeting a cross tube at its middle: the solid between their walls is
#: 0.12 mm thick at a gap of the axes of 1.12 mm (radii 0.5 mm).
def _ladder(gap: float) -> dict:
    return dict(points=mm((0.0, 0.0, -0.5, 0.0, 0.0, 1.5, gap, 0.0, -0.5, gap, 0.0, 1.5, 0.0, 0.0, 0.5,
                           gap, 0.0, 0.5)),
                segments=((0, 4), (4, 1), (2, 5), (5, 3), (4, 5)),
                radii=mm((0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.3, 0.3)))


def test_a_tree_the_fillet_fuses_is_refused_naming_a_smaller_fillet_or_a_thicker_wall():
    """Row #353: the fillet's closing would fill solid thinner than its diameter between tubes that share no point,
    joining their lumens, so the plan refuses it (``setting.value``), naming a smaller fillet or a thicker wall. The
    same tubes with the fillet under the route's share, or the wall thicker, plan."""
    tree = TubeTree(**_ladder(1.12), fillet=0.5)
    assert tree.wall_features().fused, "the plant: the 0.12 mm wall under the fillet's 0.3 mm diameter"
    with pytest.raises(BuildRefused) as refused:
        tree_experiment(tree).plan()
    (message,) = [r.message + " " + r.action for r in refused.value.refusals if r.code == "setting.value"]
    assert "fuses" in message and "smaller fillet" in message and "thicker wall" in message
    for clear in (TubeTree(**_ladder(1.12), fillet=0.1), TubeTree(**_ladder(1.5), fillet=0.5)):
        assert not clear.wall_features().fused
        assert not [r for r in _plan_refusals(clear) if r.code == "setting.value"]


def _plan_refusals(tree: TubeTree) -> list:
    try:
        tree_experiment(tree).plan()
    except BuildRefused as refused:
        return list(refused.refusals)
    return []
