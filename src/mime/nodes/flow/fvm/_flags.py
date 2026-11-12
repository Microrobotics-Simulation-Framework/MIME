"""The flow's ledger flags (``mime.core.ledger.LedgerFlag``): ratios of a metered step's terms that flag a known bug the
step identity cannot see (each term booked by its own formula, so the identity closes with the bug in it), each with
its line measured on drawn correct runs and the planted bugs that prove it, each plant's detection rate measured. Read
by ``mime.testing``'s ledger harness, never by the step: a run's program is the same with or without them.

**What each catches is what its plants showed**, not its name:

- ``stall`` (``|N_Krylov| / D_visc``, the node's own ``stall_ratio``): a solve that stops short of its solution while
  reporting arrival. **It does not catch every small stall in float32**: a solve stopped after two iterations reads
  from 6e-4 to 4.5e-3 on seven of 50 draws, under the line, which sits twice the clean maximum (2.0e-3), so a stall
  that small is inside float32's clean readings;
- ``convection`` (``N_conv`` against its time level's own work, ``-dt rho <C(u), u' - u>``, over the magnitude of
  convection's product): convection doing work beyond its time level's. The divergence form is skew on a
  divergence-free field, so at one level it does no work and at IMEX Euler's mixed levels all its work is the
  level's (``N_conv``'s own order is the order check's); a convective term formed from another velocity, or not
  skew, does more. **It does not see an error in the operator itself that keeps it skew** (scaled, negated): the
  level's work comes from the same operator, so the reading stays rounding; gate 1's manufactured solution
  (``tests/flow/test_fvm_mms.py``, its controls a sign flip and one component scaled by 1.05) is what sees one;
- ``leak`` (the wall's penalty loss in the solid's interior, beyond the band and its stiffening ramp, over the wall's
  whole loss): the drive reaching into a solid;
- the open box's own (``OPEN_FLAGS``): ``open_face_projection`` (``N_proj`` over the step's scale: a projector that is
  not orthogonal under the books' inner product, its port faces at the wrong weight); and ``opening_kinetic_flux``, an
  operator check of convection at the planes: the kinetic energy the books hold as crossing the openings (the one line
  ``<node>.openings``, the scheme's own convective power) against what crosses each opening, the convective power split
  by plane (``OpenMACGrid.opening_kinetic_fluxes``, exact to the field's divergence): a flux booked elsewhere, or
  convection doing work inside the box. Its numerator is the planes' net (the books hold one line), its scale each
  opening's flux in magnitude and the convective product's, so the scale does not cancel where in and out do. **It
  sees no port work**: its reading is a function of the step's starting velocity alone. Its lines are about three
  times the worst clean reading (15 eps in both dtypes, flat from D/h 8 to 16), measured, not derived.

**Not built** (no flag is declared for them, pending or otherwise): the periodic box's three on the open box and on
RK3, whose lines were measured on the periodic box's IMEX Euler runs, until they are measured on those runs' own
draws; and the record's rung-2 boundary flag, port work against the kinetic-energy flux, which ``opening_kinetic_flux``
is not."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from jax import Array

from mime.core import as_run_array, jnp
from mime.core import ledger as books
from mime.core.ledger import FlagEvidence, LedgerFlag
from mime.nodes.flow.fvm._band import band_named
from mime.nodes.flow.fvm._solve import DEFAULT_TOLERANCE

#: The stall detector's line, in the solve's tolerance: ``|N_Krylov| / D_visc`` above this many tolerances is a stall. **One
#: number a dtype, read by the node's ``stall_flagged`` at the tolerance the run was built with and by the harness's stall
#: flag at the default tolerance** (:data:`STALL_LINES`), so the two cannot disagree. The reading scales with the tolerance
#: (:data:`STALL_MEASURED`), so a run that asks for a looser solve is judged against that solve, not the default one.
STALL_PER_TOLERANCE = {"float64": 1000.0, "float32": 500.0}
#: The run that measured the stall reading's clean maxima at the default tolerance and a tenfold tighter and looser one
#: (200 drawn correct runs each, seeds 0 to 199 of ``tests/flow/flag_cases.py``, 60 steps, every step read), and what it
#: showed: the worst reading over the tolerance 510 in float64 and 200 in float32 at the default, 391 and 479 at a tenfold
#: tighter tolerance (float32's below its rounding floor at the worst draws), 329 and 245 at a tenfold looser one; a
#: draw's reading moves with the tolerance by 0.106 and 9.6 times (float64) and 0.108 and 9.6 times (float32), medians.
STALL_MEASURED = "https://github.com/Microrobotics-Simulation-Framework/MIME/actions/runs/42858484810"
#: The run that repeated the clean draws against the node's own flag and measured the stall plants' rates (40 planted and
#: 40 clean runs a chunk, five chunks of seeds from 1000 for the two tolerance plants, one for the two-iteration plant).
STALL_RATES_MEASURED = "https://github.com/Microrobotics-Simulation-Framework/MIME/actions/runs/42860952104"
#: The line at the default tolerance, a dtype: the harness's.
STALL_LINES = {name: STALL_PER_TOLERANCE[name] * DEFAULT_TOLERANCE[name] for name in STALL_PER_TOLERANCE}

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
    """``|N_conv - N_level| / (dt rho <|C(u)|, |u'|> + |N_conv|)``: the convective term's booked work against its time
    level's own, ``N_level = -dt rho <C(u), u' - u>``, ``C`` the grid kind's convection at the step's starting velocity
    ``u`` and ``u'`` its solution. The divergence form is skew on a divergence-free field, ``<C(u), u> = 0``, so at
    IMEX Euler's mixed levels its whole work is that level's and the difference is rounding; a convective term formed
    from another velocity, or not skew, does work beyond it. Over the magnitude of the recomputed product and of what
    was booked; zero where both are under half the dtype's digits of the product's natural size, ``(max|u| / h) <|u|,
    |u'|>`` (a flow whose convection vanishes, a straight tube's Poiseuille flow: there every digit is rounding)."""
    grid = node.grid(p)
    started, ended = _velocity(before), _velocity(after)
    terms = cast(tuple[str, ...], node.LEDGER_TERMS)
    term = after[f"{books.RECORDS}step_terms"][terms.index("convection")]
    if not node.constants()["convection"]:
        return jnp.zeros((), dtype=term.dtype)
    convection = grid.convection(started)
    change = (ended[0] - started[0], ended[1] - started[1], ended[2] - started[2])
    weight = node.delta_t * as_run_array(p["density"])
    level = -weight * grid.inner(convection, change)
    size = weight * grid.inner_magnitude(convection, ended) + jnp.abs(term)
    speed = jnp.max(jnp.stack([jnp.max(jnp.abs(component)) for component in started]))
    natural = weight * speed / grid.spacing * grid.inner_magnitude(started, ended)
    digits = jnp.sqrt(jnp.finfo(term.dtype).eps)
    # A step past the CFL line holds the state and its records (the run is refused): nothing was stepped to judge.
    stepped = (size > digits * natural) & (jnp.asarray(after["stopped_at"]) < 0)
    return jnp.where(stepped, _share(jnp.abs(term - level), size), 0.0)


def leak_reading(node: Any, p: Mapping[str, Any], before: Mapping[str, Any], after: Mapping[str, Any]) -> Array:
    """The wall's penalty loss in the solid's interior (the signed distance at least :data:`INSIDE` band widths into
    the solid), ``dt rho <Sigma u', u'>`` there, over the wall's whole loss as the ledger booked it; the wall's own
    resistivity, without a porous region's (booked under its own law)."""
    del before
    grid = node.grid(p)
    fields = node.fields(p, grid)
    nu, walls = node.kinematic(p), fields.walls
    sigma = (nu * walls[0], nu * walls[1], nu * walls[2])
    ended = _velocity(after)
    depth = INSIDE * band_named(node.constants()["band"]).width * grid.spacing
    inside = tuple(jnp.where(fields.signed_distance[c] >= depth, sigma[c] * ended[c], 0.0) for c in range(3))
    loss = node.delta_t * as_run_array(p["density"]) * grid.inner(inside, ended)
    held = books.deliveries(node.constants())
    wall = next(d.law for d in held if d.kind == books.HOSTED and d.field == "penalty")
    booked = -after[f"{books.RECORDS}step_work"][books.laws(held).index(wall)]
    return _share(loss, booked)


def _step_scale(before: Mapping[str, Any], after: Mapping[str, Any]) -> Array:
    """The step's scale, from the running sum's two ends."""
    return jnp.asarray(after[f"{books.RECORDS}scale"]) - jnp.asarray(before[f"{books.RECORDS}scale"])


def projection_reading(node: Any, p: Mapping[str, Any], before: Mapping[str, Any], after: Mapping[str, Any]) -> Array:
    """``|N_proj|`` over the step's scale: the projection's own term at the open faces, rounding where the
    projector is orthogonal under the inner product the books are formed in (the half-weight faces)."""
    del p
    terms = cast(tuple[str, ...], node.LEDGER_TERMS)
    term = jnp.asarray(after[f"{books.RECORDS}step_terms"])[terms.index("projection")]
    stepped = jnp.asarray(after["stopped_at"]) < 0
    return jnp.where(stepped, _share(jnp.abs(term), _step_scale(before, after)), 0.0)


def kinetic_flux_reading(node: Any, p: Mapping[str, Any], before: Mapping[str, Any],
                         after: Mapping[str, Any]) -> Array:
    """``|W_KE + dt rho sum_k K_k| / (dt rho (sum_k |K_k| + <|C(u)|, |u|>))``: the kinetic energy the books hold as
    crossing the openings, ``W_KE`` (the line ``<node>.openings``), against each opening's own, ``K_k`` out through it
    (the convective power split by plane at the step's start, ``OpenMACGrid.opening_kinetic_fluxes``), over each
    opening's flux in magnitude and the convective product's (the line's own rounding scale). On a discretely
    divergence-free field the two agree to rounding; a flux booked in another term, or convection doing work inside the
    box, reads above it. Zero on a step held past the CFL line and where convection is off."""
    held = books.deliveries(node.constants())
    line = next(d.law for d in held if d.kind == books.HOSTED and d.field == "kinetic_flux")
    booked = jnp.asarray(after[f"{books.RECORDS}step_work"])[books.laws(held).index(line)]
    if not node.constants()["convection"]:
        return jnp.zeros((), dtype=booked.dtype)
    grid = node.open_grid(p)
    started = _velocity(before)
    flux, size = grid.opening_kinetic_fluxes(started)
    weight = node.delta_t * as_run_array(p["density"])
    product = grid.inner_magnitude(grid.convection(started), started)
    stepped = jnp.asarray(after["stopped_at"]) < 0
    return jnp.where(stepped, _share(jnp.abs(booked + weight * jnp.sum(flux)), weight * (jnp.sum(size) + product)),
                     0.0)


#: The run that measured the lines: 200 draws a dtype of ``tests/flow/flag_cases.py``'s envelope (seeds 0 to 199), 60
#: steps each, every flag read at every step.
LINES_MEASURED = "https://github.com/Microrobotics-Simulation-Framework/MIME/actions/runs/42769581278"
_CLEAN = (f"200 drawn correct runs a dtype (IMEX Euler, D/h 8 to 16, Re 1 to 200, CFL 0.04 to 0.4, tau/dt about 4 to "
          f"3e3; tests/flow/flag_cases.py), 60 steps each, {LINES_MEASURED}")

#: The run that measured the convection reading's line (the reading as it now is: the first, ``|N_conv|`` over its
#: product, caught a stale level on 186 of 200 runs, Actions run 42771046829).
CONVECTION_LINE_MEASURED = "https://github.com/Microrobotics-Simulation-Framework/MIME/actions/runs/42775731499"
#: The run that measured the convection flag's rates, each plant on fresh draws a dtype (seeds from 1000).
CONVECTION_RATES_MEASURED = "https://github.com/Microrobotics-Simulation-Framework/MIME/actions/runs/42776571840"
_STALE = "a convective term formed from the run's starting velocity (a stale level)"
_PERMUTED = "a convective term formed from the velocity's components permuted (the wrong field)"
_ADVECTIVE = "the convective term in the advective form, (u . grad) u (not skew)"
_PAST = "four times the step, the CFL line lifted (its convection formed as declared: not this flag's)"
#: The run that measured the convection flag's blindness to an operator error, fresh draws a dtype (seeds from 1300).
OPERATOR_MEASURED = "https://github.com/Microrobotics-Simulation-Framework/MIME/actions/runs/42794715500"
_SCALED = ("the convective operator scaled by 1.5, still skew (an operator error: the manufactured-solution gate's, "
           "tests/flow/test_fvm_mms.py)")
_NEGATED = "the convective operator negated, still skew (an operator error: the manufactured-solution gate's)"
#: The run that measured the stall's and the leak's rates: each plant on 200 (or 50) fresh draws a dtype, seeds from
#: 1000, and the same draws clean.
RATES_MEASURED = "https://github.com/Microrobotics-Simulation-Framework/MIME/actions/runs/42771046829"
_TOLERANCE = "a solve whose tolerance is 1e4 times its default, reporting arrival"
_TWO = "a solve stopped after two iterations"
_STOPS_SHORT = "a solve whose CG stops at 1e4 times the tolerance the node declares (its default), reporting arrival"
_MASK = "the fluid's mask lost beyond the band (the drive acting in the solid)"
_WEAK = "the solid's interior 1e3 times less stiff (its bypass flux rises; its loss inside falls)"


def _evidence(plant: str, caught: dict[str, tuple[int, int, int, int]], link: str) -> tuple[FlagEvidence, ...]:
    return tuple(FlagEvidence(plant, dtype, *counts, link) for dtype, counts in caught.items())


#: The implicit node's flags (IMEX Euler and its steady limit). Each line is 2 to 4 times the worst clean reading (the
#: stall 2 times, the leak 3.1, convection 4.1 and 3.8), on draws whose worst sits at the envelope's edge (float32's
#: stall at tau/dt near 4e3): beyond the envelope a line is unvalidated.
FLAGS: tuple[LedgerFlag, ...] = (
    LedgerFlag("stall", "|N_Krylov| / D_visc", stall_reading, STALL_LINES,
               f"200 drawn correct runs a dtype and tolerance (a tenfold tighter and looser one and, in float32, the "
               f"nominal one set explicitly; IMEX Euler, D/h 8 to 16, Re 1 to 200, tau/dt up to 4.1e3, "
               f"tests/flow/flag_cases.py), 60 steps each: the reading over the run's tolerance at most 510 (float64) "
               f"and 200 (float32) at the default, the line {STALL_PER_TOLERANCE['float64']:g} and "
               f"{STALL_PER_TOLERANCE['float32']:g}, every step judged "
               f"({STALL_MEASURED}); the node's own flag quiet on all of them ({STALL_RATES_MEASURED})",
               _evidence(_TOLERANCE, {"float64": (200, 200, 0, 200), "float32": (200, 200, 0, 200)},
                         STALL_RATES_MEASURED)
               + _evidence(_STOPS_SHORT, {"float64": (200, 200, 0, 200), "float32": (200, 200, 0, 200)},
                           STALL_RATES_MEASURED)
               + _evidence(_TWO, {"float64": (50, 50, 0, 50), "float32": (43, 50, 0, 50)}, STALL_RATES_MEASURED)),
    LedgerFlag("convection", "|N_conv + dt rho <C(u), u' - u>| / (dt rho <|C(u)|, |u'|> + |N_conv|)",
               convection_reading, {"float64": 2e-9, "float32": 5e-5},
               f"200 drawn correct runs a dtype, as the others' ({CONVECTION_LINE_MEASURED}): worst 4.9e-10 (float64) "
               f"and 1.3e-5 (float32), every step judged: rounding, as a skew convection's whole work is its time "
               f"level's",
               _evidence(_STALE, {"float64": (200, 200, 0, 200), "float32": (200, 200, 0, 200)},
                         CONVECTION_RATES_MEASURED)
               + _evidence(_PERMUTED, {"float64": (50, 50, 0, 50), "float32": (50, 50, 0, 50)},
                           CONVECTION_RATES_MEASURED)
               + _evidence(_ADVECTIVE, {"float64": (50, 50, 0, 50), "float32": (50, 50, 0, 50)},
                           CONVECTION_RATES_MEASURED)
               + _evidence(_PAST, {"float64": (0, 50, 0, 50), "float32": (0, 50, 0, 50)}, CONVECTION_RATES_MEASURED)
               + _evidence(_SCALED, {"float64": (0, 50, 0, 50), "float32": (0, 50, 0, 50)}, OPERATOR_MEASURED)
               + _evidence(_NEGATED, {"float64": (0, 50, 0, 50), "float32": (0, 50, 0, 50)}, OPERATOR_MEASURED)),
    LedgerFlag("leak", "the wall's penalty loss inside the solid / the wall's whole loss", leak_reading,
               {"float64": 4e-3, "float32": 4e-3},
               f"{_CLEAN}: worst 1.3e-3 in both dtypes after 30 steps (a drawn start's flow in the solid decays over "
               f"them: 1.35e-2 at 10, 3.4e-3 at 20)",
               _evidence(_MASK, {"float64": (200, 200, 0, 200), "float32": (200, 200, 0, 200)}, RATES_MEASURED)
               + _evidence(_WEAK, {"float64": (0, 50, 0, 50), "float32": (0, 50, 0, 50)}, RATES_MEASURED),
               settle=30),
)

#: The run that measured the open box's lines and rates: 200 drawn open runs a dtype (``tests/flow/open_cases.py``,
#: seeds 0 to 199, 60 steps each) and each plant on 40 fresh draws (seeds from 3000), the same draws clean.
OPEN_MEASURED = "https://github.com/Microrobotics-Simulation-Framework/MIME/actions/runs/42877559684"
_OPEN_CLEAN = (f"drawn open runs (an inflow and a Windkessel with a downstream resistance of 1 to 1e4 times the "
               f"vessel's, IMEX Euler, D/h 8 to 16, Re 1 to 100 through a narrowing of up to 0.4, a run held past the "
               f"CFL line read on its steps before the hold; tests/flow/open_cases.py), 60 steps each, "
               f"{OPEN_MEASURED}")
_UNIFORM = "the planes' faces at full weight in the inner product (the projector not orthogonal under it)"

#: The run that measured the kinetic flag's lines (200 open draws a dtype, seeds 0 to 199, 60 steps each, every step
#: read) and the run that measured its plants' rates (40 planted and 40 clean runs each, seeds from 3000).
KINETIC_LINES_MEASURED = "https://github.com/Microrobotics-Simulation-Framework/MIME/actions/runs/42958800740"
KINETIC_RATES_MEASURED = "https://github.com/Microrobotics-Simulation-Framework/MIME/actions/runs/42962620848"
_IN_N_CONV = ("the kinetic energy crossing the openings left inside N_conv (its line booked as zero, N_conv as "
              "-dt rho <C(u), u'>: the step identity still closing)")
_END_HALF = "the end plane's convective half cell at a full cell (the open planes' operator gate's plant)"
_START_HALF = "the start plane's convective half cell at a full cell (the open planes' operator gate's plant)"

#: The open box's flags (``OpenFlowNode``): its own (the module says which are not built).
OPEN_FLAGS: tuple[LedgerFlag, ...] = (
    LedgerFlag("open_face_projection", "|N_proj| over the step's scale", projection_reading,
               {"float64": 2e-17, "float32": 1e-8},
               f"200 {_OPEN_CLEAN}: worst 2.4e-18 (float64) and 1.2e-9 (float32), every step judged",
               _evidence(_UNIFORM, {"float64": (40, 40, 0, 40), "float32": (40, 40, 0, 40)}, OPEN_MEASURED)),
    LedgerFlag("opening_kinetic_flux", "|the kinetic energy booked as crossing the openings + each opening's own, summed| "
                                    "over each opening's in magnitude and the convective product's",
               kinetic_flux_reading, {"float64": 1e-14, "float32": 5e-6},
               f"200 {_OPEN_CLEAN.replace(OPEN_MEASURED, KINETIC_LINES_MEASURED)}: worst 3.4e-15 (float64) and 1.8e-6 "
               f"(float32), every step judged: rounding, the convective power's split by plane being exact to the "
               f"field's divergence",
               _evidence(_IN_N_CONV, {"float64": (40, 40, 0, 40), "float32": (40, 40, 0, 40)}, KINETIC_RATES_MEASURED)
               + _evidence(_END_HALF, {"float64": (40, 40, 0, 40), "float32": (40, 40, 0, 40)}, KINETIC_RATES_MEASURED)
               + _evidence(_START_HALF, {"float64": (40, 40, 0, 40), "float32": (40, 40, 0, 40)},
                           KINETIC_RATES_MEASURED)),
)

__all__ = ["CONVECTION_LINE_MEASURED", "CONVECTION_RATES_MEASURED", "FLAGS", "INSIDE", "LINES_MEASURED", "OPEN_FLAGS",
           "KINETIC_LINES_MEASURED", "KINETIC_RATES_MEASURED", "OPEN_MEASURED", "OPERATOR_MEASURED", "RATES_MEASURED",
           "FlagEvidence", "convection_reading",
           "kinetic_flux_reading", "leak_reading", "projection_reading", "stall_reading"]
