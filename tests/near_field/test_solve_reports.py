"""Gate 10, never-silent solves on the near field's iterative route (the maintainer's item 2; F1a's design, the FVM's
fields, so that the recheck reads them as it reads the FVM's): each step's primal report a state field with its running
sum and a flag the run tallies and warns of with its count; the tangent solves' report as the tangents of those fields;
the transpose solves' as the gradient of the zero ``solve_report`` leaf, summed over the run or a row a step.

The control, every run: a cap of one iteration planted (the node's stop built with ``cap=1``) is at the cap on every step
of every run, forward (each step's report, the flag on every step, the warning's count) and backward (each step's row
of the transpose reports, and the tangent's), through the node's own scan and through MIME's chunked run; the default
stop raises none. And a tolerance below the dtype's floor (1e-30, written on the drag) stagnates every step's solve, in
both dtypes: the run warns with the count, each report says stagnated, before its cap."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from near_field_cases import NODE, bodies, joint_node, mrs, precision, rebuilt, spheres, start_inputs

from mime.build._maddening import chunk_function, full_state, graph_params
from mime.nodes.hydro.near_field import _node
from mime.nodes.hydro.near_field._gmres import REPORT_FIELDS, Stopping
from mime.units import ms

SPACING = 0.4e-3
STEPS = 4
ITERATIONS, AT_CAP, STAGNATED = (REPORT_FIELDS.index(name) for name in ("iterations", "at_cap", "stagnated"))
LEAF = "m1_shape_radius"


def pair(**settings):
    return bodies("reported", spheres(0.3e-3, 2), mrs(SPACING, near_field_dense=0, **settings),
                  velocity=(1e-4, 0.0, 0.0))


def capped(monkeypatch) -> None:
    """The plant: every GMRES of the joint node built with a cap of one iteration."""
    monkeypatch.setattr(_node, "Stopping", lambda tolerance, restart, cap: Stopping(tolerance, restart=restart, cap=1))


def node_scan(node, inputs, params):
    """``STEPS`` steps of the node alone at fixed inputs: the final state and each step's state."""
    p = node._step_constants(params)  # noqa: SLF001

    def body(state, _):
        ended = node.advance(p, state, inputs, node.delta_t)
        return ended, ended

    return jax.lax.scan(body, node.initial_state(), None, length=STEPS)


@pytest.mark.parametrize("plant", [False, True])
def test_each_steps_reports_forward_tangent_and_transpose_and_a_planted_cap_at_every_step(plant, monkeypatch):
    if plant:
        capped(monkeypatch)
    with precision("float64"):
        built = pair().build()
        node = rebuilt(joint_node(built), report_rows=STEPS)
        inputs = {key: jnp.asarray(value) for key, value in start_inputs(built).items()}
        params = node.params_pytree()
        _, history = node_scan(node, inputs, params)

        def loss(radius, tap):
            _, steps = node_scan(node, inputs, {**params, LEAF: radius, "solve_report": tap})
            weights = jnp.arange(steps["resistance"][0].size).reshape(steps["resistance"][0].shape)
            return 1e3 * jnp.sum(steps["resistance"] * weights)  # every step's solve, so every transpose solve

        radius, tap = jnp.asarray(params[LEAF]), jnp.asarray(params["solve_report"])
        _, transposed = jax.grad(loss, argnums=(0, 1))(radius, tap)
        _, tangent = jax.jvp(lambda r: node_scan(node, inputs, {**params, LEAF: r})[0]["solve_report_sum"],
                             (radius,), (jnp.asarray(1e-3),))
    forward, backward = np.asarray(history["solve_report"]), np.asarray(transposed)
    flags = np.asarray(history["solve_flagged"])
    print(f"\n{'planted' if plant else 'clean'}: forward {forward[:, [ITERATIONS, AT_CAP]].tolist()}, backward "
          f"{backward[:, [ITERATIONS, AT_CAP]].tolist()}, tangent {np.asarray(tangent)}")
    assert forward.shape == backward.shape == (STEPS, 6)
    if plant:
        assert np.all(forward[:, AT_CAP] >= 1) and np.all(flags == 1.0)
        assert np.all(backward[:, AT_CAP] >= 1) and np.all(backward[:, ITERATIONS] == 1.0)
        assert float(tangent[AT_CAP]) >= STEPS
    else:
        assert np.all(forward[:, AT_CAP] + forward[:, STAGNATED] == 0) and np.all(flags == 0.0)
        assert np.all(backward[:, AT_CAP] + backward[:, STAGNATED] == 0) and np.all(backward[:, ITERATIONS] >= 1)
        assert float(tangent[AT_CAP] + tangent[STAGNATED]) == 0.0 and float(tangent[ITERATIONS]) >= STEPS


@pytest.mark.parametrize("plant", [False, True])
def test_the_transpose_reports_leave_mimes_chunked_run_through_the_leafs_gradient(plant, monkeypatch):
    """The path a gradient study takes (the chunk's scan, each node's run-constants made before it): the gradient of a
    loss of the final poses in the ``solve_report`` leaf is the run's sum of the transpose solves' reports."""
    if plant:
        capped(monkeypatch)
    with precision("float64"):
        built = pair().build()
        graph = built.graph
        run = chunk_function(graph, STEPS, built.held_inputs())
        state, params = full_state(graph), graph_params(graph)

        def loss(tree):
            final = run(state, tree)
            return 1e3 * (final["s1"]["pose"][0] - final["s0"]["pose"][0]) + jnp.sum(final["s0"]["pose"][3:])

        gradient = jax.grad(loss)(params)
    found = np.asarray(gradient["nodes"][NODE]["solve_report"])[0]
    print(f"\n{'planted' if plant else 'clean'}: the run's transpose reports {found}")
    if plant:
        assert found[AT_CAP] >= STEPS and found[ITERATIONS] >= STEPS
    else:
        assert found[AT_CAP] == found[STAGNATED] == 0.0 and found[ITERATIONS] >= STEPS


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_a_run_whose_solves_stagnate_is_flagged_and_warned_of_with_its_count(dtype):
    """A tolerance below the dtype's floor (planted) stagnates every step's solve: the run's result names the flag with
    its count, the run warns with it, and each step's report says how it stopped. The control is the same run at the
    default tolerance, which raises neither."""
    for tolerance, flagged in ((1e-30, True), (None, False)):
        with precision(dtype):
            built = pair(**({"near_field_tolerance": tolerance} if tolerance is not None else {})).build()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = built.run(ms(5.0))
        raised = {f.path: f for f in result.flags}
        said = [str(w.message) for w in caught if "solve_flagged" in str(w.message)]
        report = np.asarray(result.final[NODE]["solve_report"])
        if flagged:
            read = raised[f"{NODE}.solve_flagged"]
            assert read.raised == read.steps == 5 and read.was_raised
            assert len(said) == 1 and "raised at 5 of 5 steps" in said[0]
            assert report[STAGNATED] >= 1.0 and report[AT_CAP] == 0.0
        else:
            assert f"{NODE}.solve_flagged" not in raised and not said and report[AT_CAP:AT_CAP + 3].sum() == 0.0


def test_a_run_past_the_report_rows_is_refused_before_it_runs():
    with precision("float64"):
        node = rebuilt(joint_node(pair().build()), report_rows=3)
    assert node.run_problem(3) is None
    refused = node.run_problem(4)
    assert refused is not None and refused.code == "drag.near-field-report-rows"
