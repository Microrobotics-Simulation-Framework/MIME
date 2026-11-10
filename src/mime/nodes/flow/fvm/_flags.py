"""The flow's ledger flags (``mime.core.ledger.LedgerFlag``): ratios of a metered step's terms that flag a known bug the
step identity cannot see (each term booked by its own formula, so the identity closes with the bug in it), each with
its line measured on drawn correct runs and the planted bugs that prove it, each plant's detection rate measured. Read
by ``mime.testing``'s ledger harness, never by the step: a run's program is the same with or without them.

**What each catches is what its plants showed**, not its name:

- ``stall`` (``|N_Krylov| / D_visc``, the node's own ``stall_ratio``): a solve that stops short of its solution while
  reporting arrival;
- ``convection`` (``|N_conv|`` over the magnitude of convection's own product, ``dt rho <|C(u)|, |u'|>``): a
  convective term not formed from the velocity it acts on. The divergence form is skew on a divergence-free field, so
  at one time level it does no work; at IMEX Euler's mixed levels its work is first order in the step's change,
  ``|N_conv|`` a few hundredths of its magnitude at the CFL line;
- ``leak`` (the wall's penalty loss in the solid's interior, beyond the band and its stiffening ramp, over the wall's
  whole loss): the drive reaching into a solid;
- ``open_face_projection`` and ``port_kinetic_flux``: pending until a medium has openings (F2b's PR 2 gates them)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from jax import Array

from mime.core import as_run_array, jnp
from mime.core import ledger as books
from mime.core.ledger import FlagEvidence, LedgerFlag
from mime.nodes.flow.fvm._band import band_named

#: The interior's depth, in band widths from the wall, from which the solid is fully stiff (the band's half width and
#: the stiffening ramp's whole width): what the leak flag counts as inside a solid.
INSIDE = 1.5


def _velocity(state: Mapping[str, Any]) -> tuple[Array, Array, Array]:
    return (state["velocity_x"], state["velocity_y"], state["velocity_z"])


def _share(numerator: Array, denominator: Array) -> Array:
    """``numerator / denominator``, zero where the denominator is not positive (a step that formed nothing to judge)."""
    held = denominator > jnp.finfo(denominator.dtype).tiny
    return jnp.where(held, numerator / jnp.where(held, denominator, 1.0), 0.0)


def stall_reading(node: Any, p: Mapping[str, Any], before: Mapping[str, Any], after: Mapping[str, Any]) -> Array:
    """``|N_Krylov| / D_visc`` of the step, as the node reports it (``stall_ratio``)."""
    del node, p, before
    return jnp.asarray(after["stall_ratio"])


def convection_reading(node: Any, p: Mapping[str, Any], before: Mapping[str, Any], after: Mapping[str, Any]) -> Array:
    """``|N_conv| / (dt rho <|C(u)|, |u'|>)``: the convective term's booked work over the magnitude of its product,
    ``C`` the grid kind's convection at the step's starting velocity ``u``, ``u'`` its solution. Zero where the product
    is under half the dtype's digits of its own natural size, ``(max|u| / h) <|u|, |u'|>`` (a flow whose convection
    vanishes, a straight tube's Poiseuille flow: there ``C`` is rounding, and so is the reading's every digit)."""
    grid = node.grid(p)
    started, ended = _velocity(before), _velocity(after)
    terms = cast(tuple[str, ...], node.LEDGER_TERMS)
    term = after[f"{books.RECORDS}step_terms"][terms.index("convection")]
    if not node.constants()["convection"]:
        return jnp.zeros((), dtype=term.dtype)
    size = grid.inner_magnitude(grid.convection(started), ended)
    speed = jnp.max(jnp.stack([jnp.max(jnp.abs(component)) for component in started]))
    natural = speed / grid.spacing * grid.inner_magnitude(started, ended)
    digits = jnp.sqrt(jnp.finfo(term.dtype).eps)
    product = node.delta_t * as_run_array(p["density"]) * size
    return jnp.where(size > digits * natural, _share(jnp.abs(term), product), 0.0)


def leak_reading(node: Any, p: Mapping[str, Any], before: Mapping[str, Any], after: Mapping[str, Any]) -> Array:
    """The wall's penalty loss in the solid's interior (the signed distance at least :data:`INSIDE` band widths into
    the solid), ``dt rho <Sigma u', u'>`` there, over the wall's whole loss as the ledger booked it."""
    del before
    grid = node.grid(p)
    fields = node.fields(p, grid)
    sigma = node.resistivity(p, fields)
    ended = _velocity(after)
    depth = INSIDE * band_named(node.constants()["band"]).width * grid.spacing
    inside = tuple(jnp.where(fields.signed_distance[c] >= depth, sigma[c] * ended[c], 0.0) for c in range(3))
    loss = node.delta_t * as_run_array(p["density"]) * grid.inner(inside, ended)
    held = books.deliveries(node.constants())
    wall = next(d.law for d in held if d.kind == books.HOSTED and d.field == "penalty")
    booked = -after[f"{books.RECORDS}step_work"][books.laws(held).index(wall)]
    return _share(loss, booked)


#: The openings' two, declared and not yet checks: no medium has an opening to plant at.
PENDING = (
    LedgerFlag("open_face_projection", "N_proj at the open faces over the step's scale", None, {},
               "not measured", pending="no medium has openings yet (a periodic box only), so no plant can be gated: "
                                       "F2b's PR 2 gates it with its measured rate"),
    LedgerFlag("port_kinetic_flux", "a port's work against the kinetic-energy flux through its faces", None, {},
               "not measured", pending="no medium has openings yet (a periodic box only), so no plant can be gated: "
                                       "F2b's PR 2 gates it with its measured rate"),
)

#: The run that measured the lines: 200 draws a dtype of ``tests/flow/flag_cases.py``'s envelope (seeds 0 to 199), 60
#: steps each, every flag read at every step.
LINES_MEASURED = "https://github.com/Microrobotics-Simulation-Framework/MIME/actions/runs/42769581278"
_CLEAN = (f"200 drawn correct runs a dtype (IMEX Euler, D/h 8 to 16, Re 1 to 200, CFL 0.04 to 0.4, tau/dt about 4 to "
          f"3e3; tests/flow/flag_cases.py), 60 steps each, {LINES_MEASURED}")

#: The implicit node's flags (IMEX Euler and its steady limit). Each line is about twice the worst clean reading.
FLAGS: tuple[LedgerFlag, ...] = (
    LedgerFlag("stall", "|N_Krylov| / D_visc", stall_reading, {"float64": 1e-5, "float32": 4e-3},
               f"{_CLEAN}: worst 5.1e-6 (float64) and 2.0e-3 (float32), every step judged; float32's clean runs read "
               f"up to twice the node's own stall line, 1e-3, at tau/dt near 3e3 (its tolerance against tau/dt)", ()),
    LedgerFlag("convection", "|N_conv| / (dt rho <|C(u)|, |u'|>)", convection_reading,
               {"float64": 0.2, "float32": 0.2},
               f"{_CLEAN}: worst 0.104 in both dtypes, every step judged (IMEX Euler's mixed time levels at the CFL "
               f"line)", ()),
    LedgerFlag("leak", "the wall's penalty loss inside the solid / the wall's whole loss", leak_reading,
               {"float64": 4e-3, "float32": 4e-3},
               f"{_CLEAN}: worst 1.3e-3 in both dtypes after 30 steps (a drawn start's flow in the solid decays over "
               f"them: 1.35e-2 at 10, 3.4e-3 at 20)", (), settle=30),
    *PENDING,
)

__all__ = ["FLAGS", "INSIDE", "LINES_MEASURED", "PENDING", "FlagEvidence", "convection_reading", "leak_reading",
           "stall_reading"]
