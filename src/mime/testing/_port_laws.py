"""The port-law family suite: the cases every law held at a flow's opening passes, on the law alone (F0's "Part D
tail", item 1, CI check (a); ``mime.nodes.flow.fvm._ports``).

- **Passivity**: the block of the rows the law gives the solver (``PortLaw.affine``'s, never a copy of it) has a
  positive semidefinite symmetric part over draws at physical scales (``mime.core.ledger.definite_at_floor``), at the
  draw's step and at the step's limit ``dt -> 0`` (a law stepped at every step a run can take: there a Windkessel's block
  is its proximal resistance, so a sign error in it fails every draw). A sign or a transposition in a law is invisible
  to its own identity, which books whatever the law does, and the same property keeps the openings' rows well posed. A
  source (a given flux or pressure) supplies work and is exempt, said so.
- **Its own books**: over a stage at drawn states, the work it received, ``dt p . q`` over its openings at the pressures
  its pressure rows give (or the fluxes its flux rows give, at drawn pressures), equals its storage's change, its loss
  and its numerical term, to rounding.

Draws (:data:`PORT_LAW_DRAWS`) are log-uniform over vascular scales in SI: resistances 1e7 to 1e11 Pa s/m^3,
compliances 1e-12 to 1e-8 m^3/Pa, steps 1e-5 to 1e-2 s, a capacitor's pressure 0 to 2e4 Pa, a flux of either sign
up to 1e-5 m^3/s at each opening and a pressure of either sign up to 2e4 Pa."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
from maddening.testing.verification import VerificationResult

from mime.core import jnp
from mime.nodes.flow.fvm._ports import PARAMETERS, PortLaw, RowKind, block, passive

_NAME = "port law"


def _log(low: float, high: float) -> Any:
    from hypothesis import strategies as st

    return st.floats(math.log(low), math.log(high)).map(math.exp)


def _windkessel_draws() -> Any:
    from hypothesis import strategies as st

    return st.tuples(_log(1e7, 1e10), _log(1e-12, 1e-8), _log(1e8, 1e11), _log(1e-5, 1e-2), st.floats(0.0, 2e4),
                     st.floats(-1e-5, 1e-5))


def _valve_draws() -> Any:
    from hypothesis import strategies as st

    return st.tuples(_log(1e-5, 1e-2), st.just(0.0), st.floats(-2e4, 2e4))


def _shared_distal_draws() -> Any:
    from hypothesis import strategies as st

    return st.tuples(_log(1e7, 1e10), _log(1e7, 1e10), _log(1e8, 1e11), _log(1e-5, 1e-2), st.just(0.0),
                     st.floats(-1e-5, 1e-5), st.floats(-1e-5, 1e-5))


#: The draws of each law's parameters (its row), its step, its state and, at each of its openings, the flux (a pressure
#: row's) or the pressure (a flux row's), by its name: ``(row..., dt, state, at each opening...)``.
PORT_LAW_DRAWS: dict[str, Callable[[], Any]] = {"windkessel": _windkessel_draws, "valve": _valve_draws,
                                                "shared_distal": _shared_distal_draws}


def _parts(law: PortLaw, drawn: tuple[float, ...]) -> tuple[Any, float, Any, Any]:
    count = len(law.parameters)
    row = list(drawn[:count]) + [0.0] * (PARAMETERS - count)
    dt, state, given = drawn[count], drawn[count + 1], drawn[count + 2:]
    held = jnp.asarray(given[0]) if law.openings == 1 else jnp.asarray(given)
    return jnp.asarray(row), dt, jnp.asarray([state]), held


def port_law_suite(law: PortLaw, *, max_examples: int = 200) -> VerificationResult:
    """``law``'s passivity and its own books on draws (the module's cases), every draw judged (none stops the run, so
    the detail counts the draws that failed: a control's rate); a source passes passivity by its exemption, said so,
    and a law with no draws fails, naming it."""
    from hypothesis import HealthCheck, Phase, given, settings

    if law.source:
        return VerificationResult(_NAME, "PASS", detail=f"{law.name}: a source (it supplies work), exempt from "
                                                        f"passivity; nothing else is checked here")
    draws = PORT_LAW_DRAWS.get(law.name)
    if draws is None:
        return VerificationResult(_NAME, "FAIL", detail=f"no draws for the port law {law.name!r} yet "
                                                        f"(mime.testing.PORT_LAW_DRAWS)")
    failures: list[str] = []
    judged: list[bool] = []

    @settings(max_examples=max_examples, deadline=None, database=None, phases=(Phase.generate,),
              suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
    @given(draws())
    def check(drawn: tuple[float, ...]) -> None:
        row, dt, state, given = _parts(law, drawn)
        found: list[str] = []
        for step in (dt, 0.0):
            if not bool(passive(law, row, step)):
                found.append(f"its block {np.asarray(block(law, row, step)).tolist()} at dt = {step:g} is not "
                             f"positive semidefinite at {drawn}")
        a, b = law.affine(row, state, 0.0, dt)
        if law.row is RowKind.PRESSURE:
            flux = given
            pressure = a + (b @ flux if law.openings > 1 else b * flux)
        else:
            flux, pressure = a, given
        ended = law.advanced(row, state, flux, jnp.asarray(pressure), dt)
        books = law.books(row, state, ended, flux, jnp.asarray(pressure), dt)
        received = dt * jnp.sum(pressure * flux)
        parts = (books.storage_end - books.storage_start, books.loss, books.numerical)
        residual = float(received - sum(parts))
        scale = float(abs(received) + sum(abs(float(part)) for part in parts)) + \
            float(abs(books.storage_end) + abs(books.storage_start))
        if abs(residual) > 64.0 * float(np.finfo(np.asarray(received).dtype).eps) * scale:
            found.append(f"its own books are {residual:.3g} J off at {drawn} (scale {scale:.3g} J)")
        judged.append(not found)
        failures.extend(found)

    check()
    failed = judged.count(False)
    if failed:
        return VerificationResult(_NAME, "FAIL", detail=f"{law.name}: failed on {failed} of {len(judged)} draws; "
                                                        f"first: {failures[0]}")
    return VerificationResult(_NAME, "PASS", detail=f"{law.name}: passive and its books closed on {len(judged)} draws")


__all__ = ["PORT_LAW_DRAWS", "port_law_suite"]
