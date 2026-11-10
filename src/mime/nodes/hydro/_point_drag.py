"""A point body's drag in a medium whose flow is solved on a grid, coupled two-way through the exchange law (F3; the
record's B8, body half points 1 to 4, and C5): ``Drag.Stokes``.

**The reading.** Each step the node reads the flow at the body's points through the exchange law's gather (two
geometry edges a face set: the kernel's average ``G u`` and its Laplacian's ``G L u``), at the points it holds, where
it delivered last step's force. Two readings are made of it:

- **the drag's** (the constant correction): ``u_d = G u + c6 G L u - M_inf f``, ``c6 = a^2/6 - sigma^2/2`` (Faxén's
  first law beyond the Gaussian's own second moment), ``M_inf`` the paired kernel's steady self-response in a free box
  by the grid's own operators, ``f`` the force the fluid last took. The grid's undeveloped self-field acts in it as a
  history (Basset-like) term, which the coupled one-step resistance includes;
- **the flow without the body's** (the time-consistent correction, the rate terms'): ``u_c = G u + c10 G L u - S``,
  ``c10 = a^2/10 - sigma^2/2`` (the volume average), ``S`` the kernel's own field from the body's past forces by the free
  box's impulse response, held as fitted decay modes (``mime.nodes.flow.fvm._exchange.self_history``). The constant
  correction's reading carries the body's own developing field, whose rate fed back into the body's inertia diverges
  (the floor sweep); ``u_c`` holds none of it.

**The law** (Maxey and Riley, Faxén; the record's C5 point 2), implicit in the body's velocity ``V'`` through its
resistance: ``(M + m_a)(V' - V)/dt = R0 (u_d - V') + (m_f + m_a) Du_c/Dt + the body's other loads``, ``R0 = 6 pi mu a``,
``m_f`` the displaced fluid's mass, ``m_a = m_f/2`` the added mass on the acceleration relative to the fluid's,
``Du_c/Dt`` the rate of ``u_c`` along the body's path (its difference from the fluid's material derivative is ``(u -
V) . grad u``, which the particle Reynolds number's line bounds). The fluid-acceleration force carries the drive's
share of a body that displaces no fluid in the solver; neither it nor the added mass is delivered to the fluid, which
already fills the body's volume and takes the same stresses and drive. The rotation is resisted by ``8 pi mu a^3``
with no fluid rotation read (the couple's exchange waits for the maintainer's word).

**The exchange.** Lagged: the fluid's step takes the drag the body felt in the step before, ``f = R0 (V - u_d)``, at
the body's position at this step's start, by the scatter's two edges a face set (``S f`` and ``L S (c6 f)``: the
transpose of the reading's, so the pair is workless at one field). ``one_way`` delivers none and corrects for none.

**Its checks, each step** (each a flag; the build refuses what it can see at its own values): the coupled one-step
margin ``(M0 + M_c - M_inf)/M0`` against the floor (``M_c`` the kernel's one-step self-response at the step's
``r = nu dt/h^2``); the self-history fit's residual; the body's gap to the vessel's wall in radii; the particle
Reynolds number."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, override

from jax import Array
from jax import lax as _lax
from jaxtyping import Float
from maddening.core.compliance.metadata import NodeMeta, Reference
from maddening.core.node import BoundaryInputSpec
from maddening.core.params import ParamSpec
from typing_extensions import Annotated, ReadOnly, TypedDict

from mime.core import MimeNode, NodeFields, Symbol, SymbolKind, Units, as_run_array, jnp, node_fields, node_input
from mime.nodes.flow.fvm._exchange import faxen, kernel_spectrum, self_history, steady_response, step_response

#: The coupled one-step margin a definite point body is held above (the floor sweep's knee: the one-step resistance
#: within -0.42..+0.61 of the exact sphere's above it, growing as 1/margin below; F3's first step, proposed to the
#: maintainer).
FLOOR = 0.2
#: The self-history fit's residual (relative to the one-step response) above which a step is flagged: the fit read
#: 1e-5 in float64 and 3.8e-4 in float32 over widths of 1 and 1.5 cells and r of 0.01 to 30 [measured, F3].
HISTORY_LINE = 1e-3
#: The body's gap to the wall, in its radii, within which a step is flagged (C5 point 6: one radius from the wall the
#: point model is -5.0 % at a = h and -8.9 % at a = 2h; under 1 % at half a radius beyond it).
WALL_LINE = 1.0
#: The particle Reynolds number, by the body's diameter and its speed relative to the reading, above which a step is
#: flagged: declared, not fitted (the law is Stokes's).
REYNOLDS_LINE = 1.0
#: The fields the scatter reads (the force a component) and the reading's inputs (each mode a component).
FORCES = ("force_x", "force_y", "force_z")
FAXEN_FORCES = ("faxen_x", "faxen_y", "faxen_z")
AVERAGES = ("gathered_x", "gathered_y", "gathered_z")
LAPLACIANS = ("curvature_x", "curvature_y", "curvature_z")

EQUATIONS = (r"(m + m_a)\frac{\mathbf{V}' - \mathbf{V}}{\Delta t} = 6 \pi \mu a \left( \mathbf{u}_d - \mathbf{V}' "
             r"\right) + (m_f + m_a) \frac{D \mathbf{u}_c}{D t}, \quad \mathbf{u}_d = G\left(1 + c_6 L\right) "
             r"\mathbf{u} - M_\infty \mathbf{f}")
_STEP = "mime.nodes.hydro._point_drag.PointDragNode.stepped"


class PointDragState(TypedDict, closed=True):
    points: ReadOnly[Annotated[Float[Array, "1 3"], Units("m")]]  # noqa: F722 - a jaxtyping shape
    force_x: ReadOnly[Annotated[Float[Array, "1"], Units("N")]]
    force_y: ReadOnly[Annotated[Float[Array, "1"], Units("N")]]
    force_z: ReadOnly[Annotated[Float[Array, "1"], Units("N")]]
    faxen_x: ReadOnly[Annotated[Float[Array, "1"], Units("N m^2")]]
    faxen_y: ReadOnly[Annotated[Float[Array, "1"], Units("N m^2")]]
    faxen_z: ReadOnly[Annotated[Float[Array, "1"], Units("N m^2")]]
    drag_reading: ReadOnly[Annotated[Float[Array, "3"], Units("m/s")]]
    clean_reading: ReadOnly[Annotated[Float[Array, "3"], Units("m/s")]]
    history: ReadOnly[Annotated[Float[Array, "modes 3"], Units("m/s")]]  # noqa: F722 - a jaxtyping shape
    resistance: ReadOnly[Float[Array, "6 6"]]  # noqa: F722 - a jaxtyping shape
    background_load: ReadOnly[Float[Array, "6"]]
    steps: ReadOnly[Annotated[Float[Array, ""], Units("1")]]
    margin: ReadOnly[Annotated[Float[Array, ""], Units("1")]]
    below_floor: ReadOnly[Annotated[Float[Array, ""], Units("1")]]
    history_residual: ReadOnly[Annotated[Float[Array, ""], Units("1")]]
    history_flagged: ReadOnly[Annotated[Float[Array, ""], Units("1")]]
    wall_gap: ReadOnly[Annotated[Float[Array, ""], Units("1")]]
    near_wall: ReadOnly[Annotated[Float[Array, ""], Units("1")]]
    particle_reynolds: ReadOnly[Annotated[Float[Array, ""], Units("1")]]
    reynolds_flagged: ReadOnly[Annotated[Float[Array, ""], Units("1")]]


class PointDragNode(MimeNode):
    """A point sphere's drag, fluid-acceleration force and added mass in a flow on a grid, coupled two-way through the
    exchange law and lagged (the module states the law, the readings and the checks)."""

    meta = NodeMeta(
        algorithm_id="MIME-NODE-130",
        description="A point sphere in a flow solved on a grid: Stokes drag with Faxén's corrections on the flow read "
                    "through the exchange kernel less the kernel's own field, the fluid-acceleration force and the "
                    "added mass on the flow without the body, its drag's reaction scattered back onto the grid.",
        governing_equations=EQUATIONS,
        assumptions=("A sphere small against the flow's scales, its particle Reynolds number small: Stokes drag, "
                     "Faxén's corrections and Maxey and Riley's terms; no Basset history force (a validity line)",
                     "The flow's solver fills the body's volume: the fluid-acceleration force and the added mass are "
                     "not delivered back to it",
                     "Lagged: the fluid's step takes the drag the body felt in the step before, at its position at "
                     "this step's start"),
        limitations=("The fluid's rotation is not read and no couple is exchanged (the body's rotation is resisted "
                     "by 8 pi mu a^3 only)",
                     "The drag's reading keeps the grid's undeveloped self-field as a history term: the coupled "
                     "one-step resistance is the floor sweep's, within -0.42..+0.61 of the exact sphere's above the "
                     "floor"),
        references=(Reference("MaxeyRiley1983", "the equation of motion of a small sphere in a nonuniform flow"),
                    Reference("Faxen1922", "the corrections for the flow's curvature"),
                    Reference("MaxeyPatel2001", "a force distributed by a Gaussian kernel on a grid")),
    )
    symbols = (Symbol(r"\mathbf{R}", SymbolKind.STATE, "resistance", "N s/m; N s; N m s", _STEP),
               Symbol(r"\mathbf{f}", SymbolKind.STATE, "force_x", "N", _STEP))

    def __init__(self, name: str, timestep: float, *, radius: float, viscosity: float, density: float,
                 spacing: float, width_cells: float, origin: Sequence[float], start: Sequence[float],
                 faxen: bool = True, maxey_riley: bool = True, one_way: bool = False,
                 vessel_centre: Sequence[float] = (), vessel_axis: Sequence[float] = (),
                 vessel_radius: float = 0.0) -> None:
        super().__init__(name, timestep, radius=float(radius), viscosity=float(viscosity), density=float(density),
                         spacing=float(spacing), width_cells=float(width_cells),
                         origin=tuple(float(c) for c in origin), start=tuple(float(c) for c in start),
                         faxen=bool(faxen), maxey_riley=bool(maxey_riley), one_way=bool(one_way),
                         **({"vessel_centre": tuple(float(c) for c in vessel_centre),
                             "vessel_axis": tuple(float(c) for c in vessel_axis),
                             "vessel_radius": float(vessel_radius)} if vessel_radius > 0.0 else {}))

    @override
    def param_specs(self) -> dict[str, ParamSpec]:
        structural = "a structural setting: the exchange edges' kernel was built with it"
        return {**super().param_specs(),
                "radius": ParamSpec(units="m", bounds=(0.0, None), description="the sphere's radius"),
                "viscosity": ParamSpec(units="Pa s", bounds=(0.0, None)),
                "density": ParamSpec(units="kg/m^3", bounds=(0.0, None), description="the fluid's density"),
                "spacing": ParamSpec(trainable=False, units="m", description=structural),
                "width_cells": ParamSpec(trainable=False, units="1", description=structural)}

    @override
    def initial_state(self) -> NodeFields:
        c = self.constants()
        zero = as_run_array(jnp.zeros(()))
        one = as_run_array(jnp.zeros(1))
        start = as_run_array(jnp.asarray(c["start"])) + as_run_array(jnp.asarray(c["origin"]))
        modes = self.history_modes(c)
        return node_fields(PointDragState(
            points=start[None, :], force_x=one, force_y=one, force_z=one, faxen_x=one, faxen_y=one, faxen_z=one,
            drag_reading=as_run_array(jnp.zeros(3)), clean_reading=as_run_array(jnp.zeros(3)),
            history=as_run_array(jnp.zeros((modes, 3))), resistance=as_run_array(jnp.zeros((6, 6))),
            background_load=as_run_array(jnp.zeros(6)), steps=zero, margin=zero, below_floor=zero,
            history_residual=zero, history_flagged=zero, wall_gap=zero, near_wall=zero, particle_reynolds=zero,
            reynolds_flagged=zero))

    @staticmethod
    def history_modes(c: Mapping[str, Any]) -> int:
        from mime.nodes.flow.fvm._exchange import HISTORY_MODES

        del c
        return HISTORY_MODES

    @override
    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        inputs = {"pose": node_input("m; 1", (7,)), "twist": node_input("m/s; rad/s", (6,))}
        for field in AVERAGES:
            inputs[field] = node_input("m/s", (1,), description="the flow read by the exchange kernel's average")
        for field in LAPLACIANS:
            inputs[field] = node_input("1/(m s)", (1,), description="the flow's Laplacian read by the kernel")
        return inputs

    # -- the step ------------------------------------------------------------------------------------------------------
    @override
    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        return self.stepped(p, state, boundary_inputs, dt)

    def coefficients(self, p: Mapping[str, Any]) -> tuple[Array, Array, Array]:
        """Faxén's coefficients beyond the kernel's own second moment (m^2): the drag reading's (``a^2/6``), the volume
        average's (``a^2/10``) and the delivery's (the pair's, the drag reading's; zero without Faxén)."""
        c = self.constants()
        a = as_run_array(p["radius"])
        width = float(c["width_cells"]) * float(c["spacing"])
        if not c["faxen"]:
            zero = jnp.zeros((), dtype=a.dtype)
            return zero, zero, zero
        drag = faxen(a, width, 1.0 / 6.0)
        return drag, faxen(a, width, 1.0 / 10.0), drag

    def responses(self, p: Mapping[str, Any], dt: float) -> dict[str, Any]:
        """The kernel's own field by the grid's operators in a free box, made in the step from the leaves (traced in
        the radius, the viscosity and the density): its steady self-response the drag reading subtracts, its one-step
        response, the coupled one-step margin, and the self-history's fitted modes."""
        c = self.constants()
        h = float(c["spacing"])
        mu, rho, a = as_run_array(p["viscosity"]), as_run_array(p["density"]), as_run_array(p["radius"])
        r = mu / rho * dt / h ** 2
        drag, average, delivery = self.coefficients(p)
        spectrum = kernel_spectrum(jnp.asarray(float(c["width_cells"]), dtype=mu.dtype))
        scale = 1.0 / (mu * h)
        steady = steady_response(spectrum, drag / h ** 2, delivery / h ** 2) * scale
        one_step = step_response(spectrum, r, jnp.zeros(1, dtype=mu.dtype), drag / h ** 2, delivery / h ** 2)[0] * scale
        mobility = 1.0 / (6.0 * jnp.pi * mu * a)
        history = self_history(spectrum, r, average / h ** 2, delivery / h ** 2)
        return {"steady": steady, "one_step": one_step, "margin": (mobility + one_step - steady) / mobility,
                "amplitude": history.amplitude * scale, "decay": history.decay, "residual": history.residual}

    def stepped(self, p: Mapping[str, Any], state: Mapping[str, Array], boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        c = self.constants()
        mu, rho, a = as_run_array(p["viscosity"]), as_run_array(p["density"]), as_run_array(p["radius"])
        dtype = mu.dtype
        pose = jnp.asarray(boundary_inputs.get("pose", jnp.zeros(7, dtype=dtype)), dtype=dtype)
        twist = jnp.asarray(boundary_inputs.get("twist", jnp.zeros(6, dtype=dtype)), dtype=dtype)
        average = jnp.concatenate([jnp.asarray(boundary_inputs[f], dtype=dtype) if f in boundary_inputs else
                                   jnp.zeros(1, dtype=dtype) for f in AVERAGES])
        laplacian = jnp.concatenate([jnp.asarray(boundary_inputs[f], dtype=dtype) if f in boundary_inputs else
                                     jnp.zeros(1, dtype=dtype) for f in LAPLACIANS])
        drag_c, average_c, delivery_c = self.coefficients(p)
        made = self.responses(p, dt)
        two_way = not c["one_way"]
        last = jnp.concatenate([jnp.asarray(state[f], dtype=dtype) for f in FORCES])
        history = jnp.asarray(state["history"], dtype=dtype)
        # The drag's reading (the constant correction) and the flow without the body's (the self-history's).
        drag_reading = average + drag_c * laplacian - (made["steady"] * last if two_way else 0.0)
        clean = average + average_c * laplacian - (jnp.sum(history, axis=0) if two_way else 0.0)
        started = jnp.asarray(state["steps"]) > 0.0
        rate = jnp.where(started, (clean - jnp.asarray(state["clean_reading"], dtype=dtype)) / dt, 0.0)
        velocity = twist[:3]
        r0 = 6.0 * jnp.pi * mu * a
        fluid_mass = 4.0 / 3.0 * jnp.pi * a ** 3 * rho
        added = 0.5 * fluid_mass if c["maxey_riley"] else jnp.zeros((), dtype=dtype)
        share = fluid_mass if c["maxey_riley"] else jnp.zeros((), dtype=dtype)
        rotation = 8.0 * jnp.pi * mu * a ** 3
        eye = jnp.eye(3, dtype=dtype)
        resistance = jnp.zeros((6, 6), dtype=dtype).at[:3, :3].set((r0 + added / dt) * eye).at[3:, 3:].set(
            rotation * eye)
        load = r0 * drag_reading + added / dt * velocity + (share + added) * rate
        background = jnp.concatenate([load, jnp.zeros(3, dtype=dtype)])
        # The drag the body felt in the step before, which the fluid's step takes now, at the body's present point.
        felt = r0 * (velocity - jnp.asarray(state["drag_reading"], dtype=dtype))
        force = jnp.where(started, felt, 0.0) if two_way else jnp.zeros(3, dtype=dtype)
        decay = made["decay"].astype(dtype)
        history = decay[:, None] * (history + made["amplitude"].astype(dtype)[:, None] * force[None, :])
        points = (pose[:3] + as_run_array(jnp.asarray(c["origin"])))[None, :]
        relative = jnp.sqrt(jnp.sum((velocity - drag_reading) ** 2))
        reynolds = 2.0 * a * relative * rho / mu
        gap = self.wall_gap(p, pose[:3], a)
        margin = made["margin"].astype(dtype)
        ended = PointDragState(
            points=points.astype(dtype), force_x=force[0:1], force_y=force[1:2], force_z=force[2:3],
            faxen_x=delivery_c * force[0:1], faxen_y=delivery_c * force[1:2], faxen_z=delivery_c * force[2:3],
            drag_reading=drag_reading, clean_reading=clean, history=history, resistance=resistance,
            background_load=background, steps=jnp.asarray(state["steps"]) + 1.0, margin=margin,
            below_floor=(margin < FLOOR).astype(dtype), history_residual=made["residual"].astype(dtype),
            history_flagged=(made["residual"] > HISTORY_LINE).astype(dtype), wall_gap=gap,
            near_wall=(gap < WALL_LINE).astype(dtype), particle_reynolds=_lax.stop_gradient(reynolds),
            reynolds_flagged=(reynolds > REYNOLDS_LINE).astype(dtype))
        return node_fields(ended)

    def wall_gap(self, p: Mapping[str, Any], position: Array, a: Array) -> Array:
        """The body's gap to the vessel's wall in its radii (a report, held out of derivatives); a large number with
        no vessel."""
        c = self.constants()
        if "vessel_radius" not in c:
            return jnp.asarray(1e30, dtype=position.dtype)
        centre = jnp.asarray(c["vessel_centre"], dtype=position.dtype)
        axis = jnp.asarray(c["vessel_axis"], dtype=position.dtype)
        axis = axis / jnp.sqrt(jnp.sum(axis * axis))
        offset = position - centre
        across = offset - jnp.dot(offset, axis) * axis
        gap = (float(c["vessel_radius"]) - jnp.sqrt(jnp.sum(across * across)) - a) / a
        return _lax.stop_gradient(gap)


def free_space_margin(radius: float, viscosity: float, density: float, spacing: float, width_cells: float, dt: float,
                      faxen_on: bool = True) -> float:
    """The coupled one-step margin a point body of ``radius`` reads in a free box (the build's check, at its own
    values), with the Faxén pair or without."""
    width = width_cells * spacing
    drag = (radius ** 2 / 6.0 - 0.5 * width ** 2) / spacing ** 2 if faxen_on else 0.0
    spectrum = kernel_spectrum(as_run_array(width_cells))
    r = viscosity / density * dt / spacing ** 2
    scale = 1.0 / (viscosity * spacing)
    steady = float(steady_response(spectrum, drag, drag)) * scale
    one_step = float(step_response(spectrum, r, as_run_array(jnp.zeros(1)), drag, drag)[0]) * scale
    mobility = 1.0 / (6.0 * math.pi * viscosity * radius)
    return (mobility + one_step - steady) / mobility


__all__ = ["AVERAGES", "FAXEN_FORCES", "FLOOR", "FORCES", "HISTORY_LINE", "LAPLACIANS", "REYNOLDS_LINE", "WALL_LINE",
           "PointDragNode", "PointDragState", "free_space_margin"]
