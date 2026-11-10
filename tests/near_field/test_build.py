"""The build's near field (drag 0.5.0, C1, 4 and Bundle 2): a medium's mrs bodies grouped into one joint node, their
reads and deliveries its; the mass matrix each body publishes only where the near field reads it; one declared mass
read by both the body and the joint node (an override of it moves the node's gain); the build's refusals (a medium
mixing a kept, uncoupled drag with the near field; a cluster whose run will not fit in the memory a run can have, on
either route); a cluster past the takeover line built on the iterative route, the build reporting its route, its memory
and its starting solve; and a saved folder of a near field, on either route, that loads and reproduces. The energy
ledger beside a near field is ``test_ledger.py``'s."""

from __future__ import annotations

import numpy as np
import pytest
from near_field_cases import NODE, assembled, bodies, joint_node, line, mrs, precision, spheres, stepped

from mime.build import BuildRefused
from mime.experiment import Experiment
from mime.interactions import NearField
from mime.interactions.hydrodynamic import UncoupledDrag
from mime.nodes.hydro.mrs import MRSDragNode
from mime.units import kg, m, ms

SPACING = 0.6e-3


def cluster(count: int = 2, gap: float = 0.4e-3, law=None, **settings):
    return bodies("near", spheres(gap, count), law if law is not None else mrs(SPACING), **settings)


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


@pytest.mark.parametrize("route", ["dense", "gmres"])
def test_a_cluster_whose_run_will_not_fit_is_refused_on_either_route(route, monkeypatch):
    """The joint node's memory estimate against a planted memory a byte under it is refused before any step
    (``drag.near-field-memory``), and a byte over it builds (the control), on either route."""
    from mime.nodes.hydro.near_field import _node

    law = mrs(SPACING, **({"near_field": NearField(solve="gmres")} if route == "gmres" else {}))
    needed = joint_node(cluster(2, law=law).build()).memory_estimate()
    assert needed > 0.0
    for room, fits in ((int(needed) - 1, False), (int(needed) + 1, True)):
        monkeypatch.setattr(_node, "device_memory", lambda room=room: ("a planted memory", room))
        if fits:
            assert joint_node(cluster(2, law=law).build()).route == route
            continue
        with pytest.raises(BuildRefused) as refused:
            cluster(2, law=law).build()
        found = [each for each in refused.value.refusals if each.code == "drag.near-field-memory"]
        assert len(found) == 1 and "a planted memory" in found[0].message
        assert f"on the {route} route" in found[0].message


@pytest.mark.parametrize(("route", "mode"), [("dense", "forward"), ("dense", "reverse"), ("gmres", "reverse")])
def test_a_derivative_whose_memory_will_not_fit_is_refused_before_the_study_compiles_it(route, mode, monkeypatch):
    """M3 of #240's review: the memory check covers a gradient study's derivative. With a planted memory the run fits
    and the derivative's estimate (``DERIVATIVE_COUNTS``, times ``MEMORY_MARGIN``, against the memory less
    ``MEMORY_HEADROOM``) does not, the build passes and ``Study.gradient`` is refused (``drag.near-field-memory``, the
    mode named) before anything is compiled; a byte more than the derivative's line and the node admits it (the
    control). A derivative holds more than its run, reverse mode more than forward."""
    from mime.nodes.hydro.near_field import _node
    from mime.study import Study, along
    from mime.study import _gradient
    from mime.units import mPa_s

    law = mrs(SPACING, **({"near_field": NearField(solve="gmres")} if route == "gmres" else {}))
    node = joint_node(cluster(2, law=law).build())
    run, forward, reverse = (node.memory_estimate(each) for each in (_node.RUN, "forward", "reverse"))
    assert 0.0 < run < forward < reverse
    needed = _node.MEMORY_MARGIN * node.memory_estimate(mode)
    line = int(needed / (1.0 - _node.MEMORY_HEADROOM))
    monkeypatch.setattr(_node, "device_memory", lambda: ("a planted memory", line - 1))
    experiment = cluster(2, law=law)
    experiment.build()

    def compiled(*_: object, **__: object) -> None:
        raise AssertionError("the derivative was compiled")

    monkeypatch.setattr(_gradient, "prepare", compiled)
    with pytest.raises(BuildRefused) as refused:
        Study(experiment).sweep(along({"water.viscosity": [mPa_s(1.0)]}), duration=ms(2.0)).gradient(
            lambda final: final["s1"]["twist"][0], mode=mode)
    found = [each for each in refused.value.refusals if each.code == "drag.near-field-memory"]
    assert len(found) == 1 and f"derivative in {mode} mode" in found[0].message and "a planted memory" in found[0].message
    monkeypatch.setattr(_node, "device_memory", lambda: ("a planted memory", line + 2))
    assert node.derivative_problem(mode, 1, 1, 2) is None and node.memory_problem() is None


#: The fit's own configuration, two of MIME's spheres at 882 force points each a radius apart (5,292 unknowns), and what
#: its fit gave (:data:`MEMORY_COUNTS`) as the estimate in GiB, written out here rather than derived from the counts
#: (a golden value: a count edited without a refit, or a term dropped, moves it), with the two peaks past the fixed part
#: the fit's own runs measured there (a radius apart, then a twentieth of one, in the fix round's measurements).
FITTED_ESTIMATE_GIB = {("dense", "float64"): 1.0454, ("gmres", "float64"): 1.5148,
                       ("dense", "float32"): 0.7835, ("gmres", "float32"): 1.0537}
MEASURED_PEAK_GIB = {("dense", "float64"): (1.048, 1.071), ("gmres", "float64"): (1.602, 1.492),
                     ("dense", "float32"): (0.884, 0.760), ("gmres", "float32"): (1.143, 1.007)}
#: The fit's residual band over its 882-point cases (the measured peak over the estimate: 0.92 to 1.13).
FIT_BAND = (0.92, 1.13)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_the_memory_estimate_is_what_the_fit_measured_and_gmres_holds_the_inverses_besides(dtype):
    """The estimate on the fit's own configuration, both routes: the golden value of the fit, the fit's measured peaks
    inside its own residual band of it, and the GMRES route's estimate over the dense route's by at least the held
    blocks' bytes (``8 sum (3 N)^2``, here counted from the bodies' force points). The refusal test above takes its
    control from the estimate itself, so it cannot see the estimate move; this one can. Measured, not a bound: the
    band is the fit's residual over that one configuration. And a node of eight such spheres, 21,168 unknowns, past
    the withdrawn line of 16,000, takes the dense route (decision #330)."""
    with precision(dtype):
        built = bodies("fit", line(1e-3, 2), UncoupledDrag.MRS(spacing=m(0.15e-3), wall_kind="free_space")).build()
        names = ["s0", "s1"]
        members = [built.graph.get_node(f"drag:{body}") for body in names]
        estimate = {route: assembled(members, names, solve=route).memory_estimate() for route in ("dense", "gmres")}
        unknowns = assembled(members, names).unknowns
        large = assembled(members * 4, [f"s{i}" for i in range(8)])  # nothing made: the node's route and size alone
    assert unknowns == 2 * 3 * 882
    assert large.unknowns == 8 * 3 * 882 > 16_000 and large.route == "dense" and not large.iterative, (
        "no size line (decision #330): a cluster past the withdrawn 16,000 unknowns takes the dense route")
    held_blocks = 8 * 2 * (3 * 882) ** 2
    for route, needed in estimate.items():
        assert needed / 2 ** 30 == pytest.approx(FITTED_ESTIMATE_GIB[(route, dtype)], rel=1e-3), (route, dtype)
        for peak in MEASURED_PEAK_GIB[(route, dtype)]:
            assert FIT_BAND[0] <= peak / (needed / 2 ** 30) <= FIT_BAND[1], (route, dtype, peak)
    assert estimate["gmres"] - estimate["dense"] >= held_blocks


def test_the_route_is_dense_unless_gmres_is_declared():
    """Decision #330: the route is the drags' declaration, never the cluster's size (the size itself is the memory
    estimate's test above, a cluster past the withdrawn line of 16,000 unknowns taking the dense route): by GMRES only
    where the drags declare it, which the build report says."""
    with precision("float64"):
        dense = cluster(2).build()
        declared = cluster(2, law=mrs(SPACING, near_field=NearField(solve="gmres"))).build()
    assert joint_node(dense).route == "dense" and joint_node(declared).route == "gmres"
    assert "solve_report" in joint_node(declared).initial_state()
    assert "solve_report" not in joint_node(dense).initial_state()
    lines = declared.report.checked.get(NODE, ())
    print("\n" + "\n".join(lines))
    assert any("as its drags declare (NearField(solve='gmres'))" in line for line in lines)
    assert any("the run's memory on the gmres route about" in line for line in lines)
    assert any(line.startswith("at the start: the joint solve took") and "every column converged" in line
               for line in lines)
    assert any("the dense route at every size" in line for line in dense.report.checked.get(NODE, ()))


def test_a_cluster_takes_gmres_only_where_every_members_drag_declares_it():
    """Members that differ: the more accurate route any asks for, the dense one, unless every member's drag declares
    GMRES; the tolerance the smallest any writes."""
    gmres = mrs(SPACING, near_field=NearField(solve="gmres", tolerance=1e-9))
    with precision("float64"):
        mixed = bodies("mixed", spheres(0.4e-3, 2), {"s0": gmres, "s1": mrs(SPACING)}).build()
        both = bodies("both", spheres(0.4e-3, 2),
                      {"s0": gmres, "s1": mrs(SPACING, near_field=NearField(solve="gmres"))}).build()
    assert joint_node(mixed).route == "dense"
    node = joint_node(both)
    assert node.route == "gmres" and float(node.constants()["tolerance"]) == 1e-9


@pytest.mark.parametrize("route", ["dense", "iterative"])
def test_a_saved_near_field_loads_and_reproduces_its_run(tmp_path, route):
    with precision("float64"):
        gmres = {"near_field": NearField(solve="gmres")} if route == "iterative" else {}
        experiment = cluster(2, law=mrs(SPACING, **gmres))
        built = experiment.build()
        first = built.run(ms(3.0), history=True)
        built.save(tmp_path / "near", steps=3)
        again = Experiment.load(tmp_path / "near").build()
        second = again.run(ms(3.0), history=True)
    assert sorted(again.graph.node_names) == sorted(built.graph.node_names)
    assert joint_node(again).iterative == (route == "iterative")
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
