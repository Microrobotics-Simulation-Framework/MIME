"""The build's near field (drag 0.5.0, C1, 4 and Bundle 2): a medium's mrs bodies grouped into one joint node, their
reads and deliveries its; the mass matrix each body publishes only where the near field reads it; one declared mass
read by both the body and the joint node (an override of it moves the node's gain); the build's refusals (a medium
mixing a kept, uncoupled drag with the near field; a cluster past the dense route's size); and a saved folder of a near
field that loads and reproduces. The energy ledger beside a near field is ``test_ledger.py``'s."""

from __future__ import annotations

import numpy as np
import pytest
from near_field_cases import NODE, bodies, joint_node, mrs, precision, spheres, stepped

from mime.build import BuildRefused
from mime.experiment import Experiment
from mime.interactions.hydrodynamic import UncoupledDrag
from mime.nodes.hydro.mrs import MRSDragNode
from mime.units import kg, m, ms

SPACING = 0.6e-3


def cluster(count: int = 2, gap: float = 0.4e-3, **settings):
    return bodies("near", spheres(gap, count), mrs(SPACING), **settings)


def test_a_mediums_mrs_bodies_are_one_joint_node_reading_each_pose_twist_and_mass_and_delivering_each_load():
    built = cluster(3).build()
    assert sorted(built.graph.node_names) == [NODE, "s0", "s1", "s2"]
    node = joint_node(built)
    assert node.members == ("s0", "s1", "s2")
    from mime.build._maddening import graph_edges

    edges = {(e[0], e[1], e[2], e[3]) for e in graph_edges(built.graph)}
    for i, body in enumerate(node.members):
        for field in ("pose", "twist", "mass_matrix"):
            assert (body, field, NODE, f"{field}_{i}") in edges
        for field in ("resistance", "background_load"):
            assert (NODE, f"{field}_{i}", body, field) in edges
    lines = built.report.checked.get(NODE, ())
    print("\n" + "\n".join(lines))
    assert any("couples s0, s1, s2 in one joint system" in line for line in lines)


def test_a_body_publishes_its_mass_matrix_only_where_the_near_field_reads_it():
    near = cluster(2).build()
    alone = bodies("alone", spheres(0.4e-3, 1), mrs(SPACING)).build()
    assert isinstance(alone.graph.get_node("drag:s0"), MRSDragNode)
    assert "mass_matrix" not in alone.graph.get_node("s0").initial_state()
    assert "publishes_mass_matrix" not in alone.graph.get_node("s0").constants()
    held = near.graph.get_node("s0").initial_state()["mass_matrix"]
    assert np.allclose(np.diag(np.asarray(held))[:3], 4.19e-6)


def test_an_override_of_a_bodys_mass_moves_the_joint_nodes_gain_and_the_body_publishes_the_new_mass():
    """One declared mass, read by the body's step and, through the mass matrix it publishes, by the joint node's
    gain: an override of it moves both, so no copy can go stale. Bodies whose relaxation time is about the step (tau/dt
    near 1), where the gain falls steeply with the mass, so that one ten times larger lowers it by far more than its
    rounding."""
    with precision("float64"):
        base = cluster(2, mass=1.9e-8, inertia=7.5e-15).build().run(ms(2.0), history=True)
        heavier = cluster(2, mass=1.9e-8, inertia=7.5e-15).build().run(
            ms(2.0), history=True, override={"s0.mass": kg(1.9e-7)})
    gains = [float(np.asarray(run.history[NODE]["gain"])[-1]) for run in (base, heavier)]
    masses = [float(np.asarray(run.history["s0"]["mass_matrix"])[-1][0, 0]) for run in (base, heavier)]
    print(f"\nthe gain {gains[0]:.6g} at the declared mass, {gains[1]:.6g} with s0 ten times heavier; s0's published "
          f"mass {masses[0]:.3g}, {masses[1]:.3g} kg")
    assert masses[1] == pytest.approx(10.0 * masses[0], rel=1e-12)
    assert gains[1] < 0.9 * gains[0]


def test_a_medium_mixing_the_kept_uncoupled_drag_with_the_near_field_is_refused():
    placed = spheres(0.4e-3, 3)
    laws = {"s0": mrs(SPACING), "s1": mrs(SPACING), "s2": UncoupledDrag.MRS(spacing=m(SPACING), wall_kind="free_space")}
    with pytest.raises(BuildRefused) as refused:
        bodies("mixed", placed, laws).build()
    assert "drag.near-field-mixed" in refused.value.codes()


def test_a_cluster_past_the_dense_routes_size_is_refused(monkeypatch):
    from mime.nodes.hydro.near_field import _node

    monkeypatch.setattr(_node, "DENSE_UNKNOWNS", 100)
    with pytest.raises(BuildRefused) as refused:
        cluster(2).build()
    assert "drag.near-field-size" in refused.value.codes()


def test_a_saved_near_field_loads_and_reproduces_its_run(tmp_path):
    with precision("float64"):
        experiment = cluster(2)
        built = experiment.build()
        first = built.run(ms(3.0), history=True)
        built.save(tmp_path / "near", steps=3)
        again = Experiment.load(tmp_path / "near").build()
        second = again.run(ms(3.0), history=True)
    assert sorted(again.graph.node_names) == sorted(built.graph.node_names)
    for node in (NODE, "s0", "s1"):
        for field, value in first.history[node].items():
            assert np.array_equal(np.asarray(value), np.asarray(second.history[node][field])), (node, field)


def test_the_joint_nodes_step_delivers_each_body_its_own_block_and_the_others_lagged_load():
    with precision("float64"):
        built = bodies("moving", spheres(0.4e-3, 2), mrs(SPACING), velocity=(1e-3, 0.0, 0.0)).build()
        out = stepped(built)
    grand = out["resistance"]
    for i in range(2):
        assert np.array_equal(out[f"resistance_{i}"], grand[6 * i:6 * i + 6, 6 * i:6 * i + 6])
    twist = np.array([1e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert np.allclose(out["background_load_0"], -grand[0:6, 6:12] @ twist, rtol=1e-12, atol=0.0)
    assert np.allclose(out["background_load_1"], -grand[6:12, 0:6] @ twist, rtol=1e-12, atol=0.0)
