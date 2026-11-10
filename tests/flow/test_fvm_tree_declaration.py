"""A tube tree's flow as a script declares it: ``TubeTree`` filled by a medium whose flow is the FVM on a periodic box.
The build holds the tree in the box (its points moved into the box's frame, its spacing from the narrowest lumen) and
steps it by IMEX Euler or its steady limit; RK3 on a tree is refused (``flow.tree-scheme``), and so is a tree that does
not repeat over the box's period where the wall's band reads it (``flow.tree-period``), each with its route. The tree's
free ends are named by geometry: ``root`` and ``end_0``, ``end_1``, ... or the names given."""

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
