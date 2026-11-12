"""The Schwarz coupling's declared schemes on the mrs rung (F5): ``Drag.MRS(coupling='lagged')`` and
``Drag.MRS(one_way=True)``, on the driven sphere of ``tests/schwarz_experiments.py``, metered, three steps.

- **Lagged**: one Schwarz pass a step from the last step's tractions. Its composed residual, the reflection of the
  step's change of tractions, is reported and booked: every receiver's books close, and the law's identity (its books
  against the composed field's work at the force points) holds with the residual in it. Its position after three steps
  is near the exact one-pass run's (an estimate, the gain's line). Control: the residual book zeroed
  (``MIME_PLANT_SCHWARZ=hidden``), which the identity misses by the residual's work.
- **One way**: the body reads the flow and delivers nothing: the flow is the flow without the body's drag, the law's row
  on the flow zero, and the law, judged at one velocity, never a gift. Control: the composition kept
  (``MIME_PLANT_SCHWARZ=delivered``), which moves the flow off the bodiless one."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from schwarz_cases import plant
from schwarz_experiments import schwarz_sphere as experiment

from mime.units import ms

#: The law's identity's line, gate 5's (``test_schwarz_flow.WORKLESS``).
WORKLESS = {"float64": 1e-9, "float32": 1e-3}
#: Lagged against exact, three steps: the position's miss on the scale of the displacement (an estimate: the gain
#: about 1e-2 under IMEX at this point, the first step's pass starting from zero tractions).
LAGGED_LINE = 3e-2
#: The sphere's starting position (m), ``tests/schwarz_experiments.py``'s.
START = (0.05e-3, 0.0, 0.0)
#: The one-way flow against the bodiless flow: the same scheme's step and solves, to the flow's CG tolerance.
ONE_WAY_LINE = 1e-8


def _metered(settings: dict[str, Any] | None, *, hide: bool = False, deliver: bool = False,
             drag: bool = True) -> dict[str, Any]:
    """The metered sphere with the drag's ``settings``, three steps: the ledger's verdict, the drag's rows, the flow's
    books of the law summed and its final state, the body's final pose. ``hide`` zeroes the law's residual book,
    ``deliver`` keeps a one-way body's composition; ``drag=False`` the body with no drag (the flow alone)."""
    import mime.nodes.flow.fvm._node as flow_node
    from mime.observations import EnergyLedger
    from mime.testing import ledger_closes

    books, compose = flow_node.law_books, flow_node.composed

    def hidden(*args: Any, **kwargs: Any) -> Any:
        account, size = books(*args, **kwargs)
        return account.at[4].set(0.0), size

    def delivered(*args: Any, **kwargs: Any) -> Any:
        return compose(*args, **{**kwargs, "delivers": True})

    flow_node.law_books = hidden if hide else books
    flow_node.composed = delivered if deliver else compose
    try:
        exp = experiment(settings=settings)
        if not drag:
            exp.remove("drag:ball", kind="law")
        exp.observe(EnergyLedger())
        result = exp.build().run(ms(3.0))
    finally:
        flow_node.law_books, flow_node.composed = books, compose
    flow = result.final["water.flow"]
    held: dict[str, Any] = dict(verdict=ledger_closes(result), flow=flow, pose=np.asarray(result.final["ball"]["pose"]))
    if drag:
        assert result.energy is not None
        rows = {(law.law, law.receiver): law for law in result.energy.laws}
        held.update(body=rows[("drag:ball", "ball")], fluid=rows[("drag:ball", "water.flow")],
                    books=np.asarray(flow["schwarz_0_books_sum"]), gifts=float(flow["schwarz_0_gift_steps"]))
    return held


def _identity_miss(run: dict[str, Any]) -> tuple[float, float]:
    exchange, near, boxed, quadrature, residual = run["books"]
    fluid = run["fluid"].work
    read = exchange - near + boxed - quadrature - residual
    return abs(fluid - read), max(abs(fluid), abs(read))


def test_a_lagged_run_books_its_residual_and_its_books_close(dtype):
    """Lagged: every receiver's books close, the hand-over is workless, the law's identity holds with the residual
    booked (which is not zero: the pass's own error), and nothing is a gift. ``MIME_PLANT_SCHWARZ=hidden`` zeroes the
    residual book (the detection-rate probe's plant)."""
    run = _metered({"coupling": "lagged"}, hide=plant() == "hidden")
    assert run["verdict"].status == "PASS", run["verdict"].detail
    miss, scale = _identity_miss(run)
    residual = float(run["books"][4])
    body, exchange = run["body"].work, float(run["books"][0])
    print(f"F5 lagged [{dtype}]: identity miss {miss / scale:.3g}, residual's work {residual:.3g} J of the near "
          f"field's {float(run['books'][1]):.3g} J, residual {float(run['flow']['schwarz_residual']):.3g}")
    assert miss <= WORKLESS[dtype] * scale, (miss, scale)
    assert abs(body + exchange) <= WORKLESS[dtype] * max(abs(body), abs(exchange)), (body, exchange)
    assert residual != 0.0 and float(run["flow"]["schwarz_residual"]) > 0.0
    assert run["gifts"] == 0.0 and run["body"].gift_steps == 0 and run["fluid"].gift_steps == 0


def test_a_lagged_runs_hidden_residual_breaks_the_laws_identity(float64):
    """Lagged's control: the residual book zeroed, which the identity misses by the residual's work on every run."""
    run = _metered({"coupling": "lagged"}, hide=True)
    miss, scale = _identity_miss(run)
    assert miss > WORKLESS["float64"] * scale, (miss, scale)


def test_a_lagged_run_is_near_the_exact_run(float64):
    """Lagged against one pass, three steps: the body's position within the gain's estimate of the exact run's."""
    lagged, exact = (_metered(settings)["pose"][:3] for settings in ({"coupling": "lagged"}, None))
    moved = float(np.linalg.norm(exact - np.asarray(START)))
    miss = float(np.linalg.norm(lagged - exact))
    print(f"F5 lagged against exact: position miss {miss:.3g} m, {miss / moved:.3g} of the displacement")
    assert moved > 0.0
    assert miss <= LAGGED_LINE * moved, (miss, moved)


def test_a_one_way_body_leaves_the_flow_as_the_flow_without_it(float64):
    """One way: the flow's field after three steps is the flow's with no drag on the body (to the CG's tolerance), the
    law's row on the flow zero, the hand-over workless and never a gift. The plant keeps the composition."""
    alone = _metered(None, drag=False)["flow"]
    run = _metered({"one_way": True}, deliver=plant() == "delivered")
    flow = run["flow"]
    scale = max(float(np.max(np.abs(alone[f"velocity_{x}"]))) for x in "xyz")
    miss = max(float(np.max(np.abs(flow[f"velocity_{x}"] - alone[f"velocity_{x}"]))) for x in "xyz")
    print(f"F5 one way: the flow off the bodiless flow by {miss / scale:.3g} of its largest speed")
    assert miss <= ONE_WAY_LINE * scale, (miss, scale)
    assert run["verdict"].status == "PASS", run["verdict"].detail
    assert run["fluid"].work == 0.0
    body, exchange = run["body"].work, float(run["books"][0])
    assert abs(body + exchange) <= WORKLESS["float64"] * max(abs(body), abs(exchange)), (body, exchange)
    assert run["gifts"] == 0.0 and run["body"].gift_steps == 0


@pytest.mark.parametrize(("settings", "message"), [
    ({"one_way": "yes"}, "is a switch"),
    ({"coupling": "lazy"}, "is one of"),
    ({"one_way": True, "coupling": "lagged"}, "leave one of them out"),
    ({"kernel_width": 1.5}, "the mrs rung reads none"),
])
def test_the_schemes_settings_are_checked_as_written(settings, message):
    from mime.build import BuildRefused

    with pytest.raises(BuildRefused) as refused:
        experiment(settings=settings).build()
    assert any(message in r.message for r in refused.value.refusals), refused.value.refusals
