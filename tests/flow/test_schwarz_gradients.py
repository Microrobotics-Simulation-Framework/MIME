"""Gate 11 (F5), float64: derivatives through the coupled Schwarz solve by the implicit route (GMRES's and the walled
CG's own tangent and transpose solves), forward and reverse, against central differences, gate 3 having shown the map
converged (``test_schwarz_flow.py``, ``test_schwarz_open.py``).

- **Through the run** (``Study.gradient``, two steps): the driven sphere's axial position in the water's viscosity;
  the undriven magnetic sphere's position along the wall in the field's amplitude (the torque about the tube's axis
  moves the off-axis ball along the wall only through the wall's reflection). Each against the central difference
  of two runs built at the moved value.
- **In the body's position, at the node**: a position is no study axis here (``ball.initial.position`` writes the
  flow's start, which only the initial state reads, and a vector into the pose's seven fields is a part of a field:
  ``gradient.leaf-unread``, ``gradient.shape``), so the first preparation's handed response (its radial resistance and
  the axial load) is differentiated in the pose's x by ``jax.jvp`` and ``jax.vjp`` of ``schwarz_prepare``, against
  the central difference of two preparations. Free space's near field does not move with the pose: the whole
  derivative is the walls' reflection's.
- **Controls** (``MIME_PLANT_SCHWARZ``): ``stopped``, a ``stop_gradient`` on the reflection, which the derivatives
  whose operator moves (the viscosity's, the position's) miss; ``capped``, the columns' GMRES capped at one product,
  whose run is flagged unconverged (gate 3's catch), whatever its derivative against its own differences."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from schwarz_cases import draw, plant
from schwarz_experiments import schwarz_sphere as experiment

#: The run's derivatives against their central differences: the solves' tolerance (the columns' 1e-8) over the
#: difference's relative step moves it by about 1e-8 / h, and its truncation is about h^2 where the map curves (the
#: viscosity's, h = 1e-3: 1e-5 and 1e-6); the amplitude's map is linear but for the ball's turn in two steps (under a
#: milliradian), so its step is 1e-2 (noise 1e-6). The node's position derivative at tight solves (columns 1e-11, CG
#: 1e-12), its step 1e-7 m over a gap of 0.2 mm (truncation 2.5e-7, noise 2e-8).
LINE = 1e-4
STEPS_MS = 2.0
CASES = {"viscosity": ("water.viscosity", 1e-3), "amplitude": ("field.commands.amplitude", 1e-2)}


def _planted() -> Any:
    """The plant switched on for this run, put back by the caller (the module attribute it replaced)."""
    import mime.nodes.flow.fvm._node as flow_node
    import mime.nodes.flow.fvm._schwarz as schwarz

    if plant() == "stopped":
        held = schwarz.reflection

        def stopped(*args: Any, **kwargs: Any) -> Any:
            found, report = held(*args, **kwargs)
            return jax.lax.stop_gradient(found), report

        schwarz.reflection = stopped
        return schwarz, "reflection", held
    if plant() == "capped":
        held = flow_node.GmresStopping

        def capped(tolerance: Any, restart: int, cap: int) -> Any:
            del cap
            return held(tolerance, restart=restart, cap=1)

        flow_node.GmresStopping = capped
        return flow_node, "GmresStopping", held
    return None


def _case(name: str, value: float) -> Any:
    from mime.units import Pa_s

    if name == "viscosity":
        return experiment(viscosity=value), Pa_s(value), (lambda final: final["ball"]["pose"][2])
    from mime.units import mT

    return (experiment(magnetic=True, drive=0.0, amplitude=value * 1e3), mT(value * 1e3),
            (lambda final: final["ball"]["pose"][1]))


@pytest.mark.parametrize("mode", ["forward", "reverse"])
@pytest.mark.parametrize("name", list(CASES))
def test_a_derivative_through_the_coupled_run_matches_its_central_difference(float64, name, mode):
    """The study's derivative in the case's value (drawn within a quarter of 1 mPa s or 1 mT), forward or reverse,
    against the central difference of two runs
    built at the value moved by its relative step; the run itself unflagged (the columns converged every step)."""
    from mime.study import Study, along
    from mime.units import ms

    path, h = CASES[name]
    value = 1e-3 * float(draw().uniform(0.8, 1.25))  # Pa s, or T
    replaced = _planted()
    try:
        exp, quantity, of = _case(name, value)
        result = Study(exp).sweep(along({path: [quantity]}), duration=ms(STEPS_MS)).gradient(
            of, wrt=[path], mode=mode,
            acknowledged={"water.flow": "gate 11: the derivative is in the viscosity and the field's amplitude; the "
                                        "flow's geometry leaves are not moved"})
        derivative = float(np.asarray(result.member(0).gradient[path]))
        ends = [float(of(_case(name, value * (1 + sign * h))[0].build().run(ms(STEPS_MS)).final)) for sign in (1, -1)]
        run = _case(name, value)[0].build().run(ms(STEPS_MS))
    finally:
        if replaced is not None:
            setattr(*replaced)
    difference = (ends[0] - ends[1]) / (2.0 * h * value)
    flagged = [read.flag.field for read in run.flags if read.was_raised and "schwarz" in read.flag.field]
    miss = abs(derivative - difference) / abs(difference)
    print(f"F5 gate 11 [{name}, {mode}]: derivative {derivative:.10g}, central difference {difference:.10g}, "
          f"relative {miss:.3g}; flagged {flagged}")
    assert flagged == [], flagged
    assert miss <= LINE, (derivative, difference)


#: The pose's x step (m) for the node's central difference.
POSE_STEP = 1e-7
#: The node remade at these tolerances for the position's derivative.
TIGHT_SOLVES = {"tolerance": 1e-12, "schwarz_tolerance": 1e-11}


def test_the_first_preparation_differentiates_in_the_bodys_position(float64):
    """The handed response's radial resistance and axial load at the run's start (the pose's x drawn from the axis to
    0.1 mm, the gap 2.0 to 2.8 cells), in the pose's x: ``jax.jvp`` and
    ``jax.vjp`` of the flow's preparation against the central difference of two preparations; the preparation
    unflagged."""
    from mime.nodes.flow.fvm._schwarz import RESIDUAL_LINE
    from mime.nodes.flow.fvm._node import grid_zero_box

    replaced = _planted()
    try:
        with jax.enable_x64(True):
            made: Any = experiment().build().graph.get_node("water.flow")
            node: Any = type(made)(made.name, float(made.delta_t), **{**dict(made.params), **TIGHT_SOLVES})
            p = {**node.constants(), **node.params_pytree()}
            state = node.initial_state()
            grid = node.grid(p)
            velocity = tuple(jnp.asarray(state[f"velocity_{x}"]) for x in "xyz")
            force = node.schwarz_forcing(node.fields(p, grid), {}, jnp.asarray(p["schwarz_drive"]), jnp.float64)
            start = jnp.asarray(state["schwarz_0_at"]).at[0].set(float(draw().uniform(0.0, 1e-4)))

            def response(x: Any) -> tuple[Any, Any]:
                _, _, prepared, _, _ = node.schwarz_prepare(
                    p, velocity, jnp.asarray(0, dtype=jnp.int32), node.delta_t, force, start.at[0].set(x),
                    grid_zero_box(node.schwarz_box()), jnp.zeros(3), node.schwarz_previous(state), state)
                return jnp.stack([prepared.resistance[0, 0], prepared.load[2]]), prepared.report

            x0 = start[0]
            (found, report), (tangent, _) = jax.jvp(response, (x0,), (jnp.ones(()),))
            outputs, pullback = jax.vjp(lambda x: response(x)[0], x0)
            reverse = jnp.stack([pullback(jnp.eye(2)[k])[0] for k in range(2)])
            ends = [np.asarray(response(x0 + sign * POSE_STEP)[0]) for sign in (1, -1)]
    finally:
        if replaced is not None:
            setattr(*replaced)
    difference = (ends[0] - ends[1]) / (2.0 * POSE_STEP)
    forward, backward = np.asarray(tangent), np.asarray(reverse)
    misses = [float(np.max(np.abs(d - difference) / np.abs(difference))) for d in (forward, backward)]
    residual = float(np.max(np.asarray(report)[1::6]))
    print(f"F5 gate 11 [position, node]: radial resistance and axial load {np.asarray(found)}; forward {forward}, "
          f"reverse {backward}, central difference {difference}; relative {misses}; the columns' residual {residual:.3g}")
    assert residual <= RESIDUAL_LINE * TIGHT_SOLVES["schwarz_tolerance"], residual
    assert max(misses) <= LINE, (forward, backward, difference)
