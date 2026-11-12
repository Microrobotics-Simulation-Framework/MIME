"""The Schwarz flow's prepared response is stored state (F5; the coordinator's condition on the staggered order).

- **A restart carries it**: a run continued from where the last one ended is the uninterrupted run, every field of
  every node to its rounding (another compiled scan: bit for bit holds per program). Control: the continued run
  started from the initial preparation instead of the stored one (``MIME_PLANT_SCHWARZ=restart``), which must differ.
- **A changed parameter prepares it again**: an override of the water's viscosity on a run from the start runs the
  first step on a preparation made at the new viscosity, so the run is the experiment declared with that viscosity, to
  the solves' rounding; after the start, where the stored preparation would be stale, the override is refused by name
  (``override.initial-after-start``, naming the flow). Control: the preparation made at the build's viscosity whatever
  the override (``MIME_PLANT_SCHWARZ=stale``), which must miss the declared run."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from schwarz_cases import draw, plant
from schwarz_experiments import schwarz_sphere as experiment

from mime.build import BuildRefused
from mime.units import Pa_s, ms

#: The Schwarz flow's stored preparation (the fields the body's next step reads and the flow composes from).
PREPARED = ("schwarz_base_x", "schwarz_base_y", "schwarz_base_z", "schwarz_0_columns", "schwarz_0_resistance",
            "schwarz_0_load", "schwarz_0_margin", "schwarz_report")


#: A continued run against the uninterrupted one, each floating field on its own scale: the two compile other scan
#: programs (a chunk of one step is another program than one of several; chunks of 2 and 2 are the run of 4 bit for
#: bit), whose last bits differ (bit for bit holds per compiled program). Established in float64 on one thread: the
#: same continued program from the same saved state is equal twice, the preparation the continued run starts from is
#: the stored one bit for bit, and a restart in a fresh build from the saved state is the continued run bit for bit
#: (the gates probe's restart cause arm; a multi-threaded run is never judged bit for bit). The programs' last bits, carried through the step's solves, moved a field by at most 5.4e-14 of
#: its scale (the twist, a split of 1) with no solve's iteration count moved; the line, 1e-12, sits between that and
#: the solves' tolerance (1e-8), where a state not carried would show. In float32 a solve may stop an iteration apart
#: on the last bits, so its line is the solves' tolerance's order (the columns' 1e-4).
RESTART_LINE = {"float64": 1e-12, "float32": 1e-3}


def _relative(field: str) -> bool:
    """A field that is itself relative (a solve's residual over its right-hand side, a report's counts and residuals):
    judged on its own inputs' scale, one, never on its value, a cancellation's remainder (the composed residual at a
    split of 1 moves 3.9e-8 of its value of 3.3e-9, 1.3e-16 on its scale)."""
    return field.endswith("residual") or "report" in field


def _differing(first: dict[str, dict[str, Any]], second: dict[str, dict[str, Any]], line: float) -> list[str]:
    """Each field off its rounding line, with its miss on its own scale."""
    found = []
    for node, fields in first.items():
        for field, value in fields.items():
            a, b = np.asarray(value), np.asarray(second[node][field])
            if a.dtype.kind != "f":
                if not np.array_equal(a, b):
                    found.append(f"{node}.{field}")
                continue
            scale = 1.0 if _relative(field) else float(np.max(np.abs(a))) if a.size else 0.0
            miss = float(np.max(np.abs(a - b))) if a.size else 0.0
            if miss > line * scale:
                found.append(f"{node}.{field} {miss / scale if scale else miss:.3g}")
    return found


def test_a_continued_run_is_the_uninterrupted_run(dtype):
    """Four steps run as one run, and as a run of ``k`` steps continued for the rest (``k`` drawn): every field of every
    node equal to its rounding, the stored preparation among them. The plant starts the continued run from the initial
    preparation."""
    split = int(draw().integers(1, 4))
    whole = experiment().build().run(ms(4.0)).final
    built = experiment().build()
    built.run(ms(float(split)))
    if plant() == "restart":
        from mime.build._maddening import node_state, set_node_state

        held = node_state(built.graph, "water.flow")
        initial = built.graph.get_node("water.flow").initial_state()
        set_node_state(built.graph, "water.flow", {**held, **{key: initial[key] for key in PREPARED}})
    continued = built.run(ms(4.0 - split)).final
    differing = _differing(whole, continued, RESTART_LINE[dtype])
    print(f"F5 restart [{dtype}]: split at {split} of 4 steps, {len(differing)} fields off the line")
    assert differing == [], differing[:8]


#: The override's run against the experiment declared with the value: the same compiled arithmetic on other leaves,
#: within the solves' rounding (the columns' GMRES at 1e-8 and the CG at the flow's tolerance, float64).
DECLARED_LINE = 1e-9


def test_an_override_of_the_viscosity_prepares_the_first_step_again(float64):
    """The water's viscosity overridden on a run from the start (a factor drawn from 1.5 to 3): the flow's initial
    preparation is made at it, so the run's body pose equals the experiment declared with that viscosity; the plant
    makes the preparation at the build's viscosity whatever the override."""
    from mime.nodes.flow.fvm._node import ImplicitFlowNode

    factor = float(draw().uniform(1.5, 3.0))
    mu = 1e-3 * factor
    original = ImplicitFlowNode.started
    if plant() == "stale":
        def stale(self: Any, params: Any, state: Any) -> Any:
            return original(self, {**params, "viscosity": np.asarray(1e-3, dtype=np.asarray(params["viscosity"]).dtype)},
                            state)

        ImplicitFlowNode.started = stale  # type: ignore[method-assign]
    try:
        overridden = experiment().build().run(ms(2.0), override={"water.viscosity": Pa_s(mu)}).final
    finally:
        ImplicitFlowNode.started = original  # type: ignore[method-assign]
    declared = experiment(viscosity=mu).build().run(ms(2.0)).final
    base = experiment().build().run(ms(2.0)).final
    pose, want, before = (np.asarray(f["ball"]["pose"][:3]) for f in (overridden, declared, base))
    moved = float(np.linalg.norm(want - before))
    miss = float(np.linalg.norm(pose - want))
    print(f"F5 fork: viscosity x{factor:.3f}, the override's position off the declared run's by {miss:.3g} m "
          f"(relative to the viscosity's own effect {miss / moved:.3g})")
    assert moved > 0.0
    assert miss <= DECLARED_LINE * moved, (miss, moved)


def test_an_override_after_the_start_is_refused_by_name():
    """After the run's start the stored preparation is the old viscosity's; the override is refused by name."""
    built = experiment().build()
    built.run(ms(1.0))
    with pytest.raises(BuildRefused) as refused:
        built.run(ms(1.0), override={"water.viscosity": Pa_s(2e-3)})
    found = [r for r in refused.value.refusals if r.code == "override.initial-after-start"]
    assert found and "water.flow" in found[0].message, refused.value.refusals
