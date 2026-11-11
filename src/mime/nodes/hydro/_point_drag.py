"""A point body's drag in a medium whose flow is solved on a grid, coupled two-way through the exchange law (F3; the
record's B8, body half points 1 to 4, and C5): ``Drag.Stokes``.

**The reading.** Each step the node reads the flow at the body's point through the exchange law's gather (three
geometry edges a face set: the kernel's average ``G u``, its Laplacian's ``G L u`` and its antisymmetric dipole's, the
fluid's rotation ``(1/2) G curl u``), at the point it holds, where it delivered last step's force and couple. Three
readings are made of it:

- **the drag's** (the constant correction): ``u_d = G u + c6 G L u - M_inf f``, ``c6 = a^2/6 - sigma^2/2`` (Faxén's
  first law beyond the Gaussian's own second moment), ``M_inf`` the paired kernel's steady self-response in a free box
  by the grid's own operators, ``f`` the force the fluid last took. The grid's undeveloped self-field acts in it as a
  history (Basset-like) term, which the coupled one-step resistance includes;
- **the couple's** (likewise): ``Omega_d = (1/2) G curl u - K_inf T``, ``K_inf`` the dipole's steady self-rotation,
  ``T`` the couple the fluid last took (Faxén's second law: no curvature term);
- **the flow without the body's** (the rate terms'): ``u_c = G u + c10 G L u - S``, ``c10 = a^2/10 - sigma^2/2`` (the
  volume average), ``S`` the kernel's own field from the body's past forces in a free box. By default **exact** (route
  2): the body's own free-box flow held by its transform and advanced by the grid's operator each step
  (``mime.nodes.flow.fvm._exchange.own_advance``), with Hasimoto's image term; ``self_history="modes"`` (route 1) holds
  it as fitted decay modes instead (``self_history``, labelled fitted, its residual and span reported and flagged).
  Under the field's steady limit the own field is the steady response to last step's force, exactly, for both. The
  constant correction's reading carries the body's own developing field, whose rate fed back into the body's inertia
  diverges (the floor sweep); ``u_c`` holds none of it.

**The law** (Maxey and Riley, Faxén; the record's C5 point 2), implicit in the body's twist ``(V', W')`` through its
resistance: ``(M + m_a)(V' - V)/dt = R0 (u_d - V') + (m_f + m_a) Du_c/Dt + the body's other loads``, the couple ``R_r
(Omega_d - W')``, ``R0 = 6 pi mu a``, ``R_r = 8 pi mu a^3``, ``m_f`` the displaced fluid's mass, ``m_a = m_f/2`` the
added mass on the acceleration relative to the fluid's, ``Du_c/Dt`` the rate of ``u_c`` along the body's path (its
difference from the fluid's material derivative is ``(u - V) . grad u``, which the particle Reynolds number's line
bounds). The fluid-acceleration force carries the drive's share of a body that displaces no fluid in the solver;
neither it nor the added mass is delivered to the fluid, which already fills the body's volume and takes the same
stresses and drive.

**The exchange.** Lagged: the fluid's step takes the drag and the couple the body felt in the step before, ``f = R0 (V -
u_d)`` and ``T = R_r (W - Omega_d)``, at the body's point at this step's start, by the scatter's edges (``S f``, ``L S
(c6 f)`` and the dipole's ``(1/2) curl (T w)``: the transposes of the readings', so the pairs are workless at one
field). ``one_way`` delivers none and corrects for none; the field then takes nothing from the law, so the graph steps
it first and the readings are the field the step has just made (two-way, the field at the step's start).

**Its checks, each step** (each a flag; the build refuses what it can see at its own values): the coupled one-step
margin of the force and of the couple against the floor; the law's power at one velocity (its gift, judged here for
the energy ledger, which judges the drag's rows in this sum: :data:`GIFT_STEPS`); the self-history fit's residual and
span (route 1); the body's gap to the vessel's wall in radii; the particle Reynolds number; and a growing alternation
of the rate terms' load (the lagged loop through the body's own field, unstable)."""

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
from typing_extensions import Annotated, NotRequired, ReadOnly, TypedDict

from mime.core import MimeNode, NodeFields, Symbol, SymbolKind, Units, as_run_array, jnp, node_fields, node_input, \
    run_constants_given
from mime.core.ledger import IDENTITY_ULPS
from mime.nodes.flow.fvm._exchange import HISTORY_MODES, JUDGED_PLACES, KernelSpectrum, box_side, faxen, \
    held_spectrum, image_mode, own_advance, own_box, own_kernels, own_read, self_history, steady_response, step_response

#: The coupled one-step margin a definite point body is held above, the force's and the couple's alike (the floor
#: sweep's knee: the one-step resistance within -0.42..+0.61 of the exact sphere's above it, growing as 1/margin below;
#: the maintainer's number).
FLOOR = 0.2
#: The self-history fit's residual (relative to the one-step response) above which a step is flagged (route 1): the
#: fit read 1e-5 in float64 and 3.8e-4 in float32 over widths of 1 and 1.5 cells and r of 0.01 to 30 [measured, F3],
#: and the response's spread over places in the cell and face sets, which the residual carries, up to 3e-4 at a kernel
#: one cell wide.
HISTORY_LINE = 1e-3
#: The body's gap to the wall, in its radii, within which a step is flagged (C5 point 6: one radius from the wall the
#: point model is -5.0 % at a = h and -8.9 % at a = 2h; under 1 % at half a radius beyond it).
WALL_LINE = 1.0
#: The particle Reynolds number, by the body's diameter and its speed relative to the reading, above which a step is
#: flagged: declared, not fitted, the rung's ReynoldsRange (STOKES_REYNOLDS: the law is Stokes's).
REYNOLDS_LINE = 0.1
#: The Basset number a sqrt(omega/nu) of the drive's frequency above which the build reports the history force as
#: outside the model (it scales against Stokes drag as that number: C5 point 2); declared, not fitted.
BASSET_LINE = 0.3
#: The steps in a row in which the rate terms' load reverses its sign and grows before ``rate_feedback`` is raised: a
#: lagged loop through the body's own field whose gain is above one rings so (the refuting review's light body under
#: the steady limit, 14 % a step); a stable loop's alternation decays. Declared.
RINGING_STEPS = 3
#: The self-history's routes: the body's own free-box flow advanced exactly (the default), or fitted modes.
ROUTES = ("exact", "modes")
#: The fields the scatters read and the reading's inputs (each mode a face set's).
FORCES = ("force_x", "force_y", "force_z")
FAXEN_FORCES = ("faxen_x", "faxen_y", "faxen_z")
AVERAGES = ("gathered_x", "gathered_y", "gathered_z")
LAPLACIANS = ("curvature_x", "curvature_y", "curvature_z")
ROTATIONS = ("rotation_x", "rotation_y", "rotation_z")
COUPLE = "couple"
#: The node's state field counting the steps whose law gave energy, which the run's energy account reads for the
#: drag's rows (``mime.results._energy``, ``CHECK_CLUSTER``).
GIFT_STEPS = "cluster_gift_steps"
#: The run-constants the node makes from its leaves once a chunk (:meth:`PointDragNode.run_constants`), in this order
#: in ``self_responses``: the kernel's steady self-response the drag reading subtracts and its one-step response
#: (m/(N s)), the force's coupled margin, the own field's steady response under the steady limit (m/(N s)), the
#: couple's steady self-rotation and one-step response (rad/(N m s)), the couple's margin, and Hasimoto's image term
#: as a mode (its amplitude, m/(N s), and its decay a step).
RESPONSES = ("steady", "one_step", "margin", "steady_clean", "couple_steady", "couple_one_step", "couple_margin",
             "image", "image_decay")

EQUATIONS = r"\mathbf{f} = 6 \pi \mu a \left( \mathbf{V} - \mathbf{u} \right)"
_STEP = "mime.nodes.hydro._point_drag.PointDragNode.stepped"


class PointDragState(TypedDict, closed=True):
    points: ReadOnly[Annotated[Float[Array, "1 3"], Units("m")]]  # noqa: F722 - a jaxtyping shape
    force_x: ReadOnly[Annotated[Float[Array, "1"], Units("N")]]
    force_y: ReadOnly[Annotated[Float[Array, "1"], Units("N")]]
    force_z: ReadOnly[Annotated[Float[Array, "1"], Units("N")]]
    faxen_x: ReadOnly[Annotated[Float[Array, "1"], Units("N m^2")]]
    faxen_y: ReadOnly[Annotated[Float[Array, "1"], Units("N m^2")]]
    faxen_z: ReadOnly[Annotated[Float[Array, "1"], Units("N m^2")]]
    couple: ReadOnly[Annotated[Float[Array, "1 3"], Units("N m")]]  # noqa: F722 - a jaxtyping shape
    drag_reading: ReadOnly[Annotated[Float[Array, "3"], Units("m/s")]]
    raw_reading: ReadOnly[Annotated[Float[Array, "3"], Units("m/s")]]
    rotation_reading: ReadOnly[Annotated[Float[Array, "3"], Units("rad/s")]]
    raw_rotation: ReadOnly[Annotated[Float[Array, "3"], Units("rad/s")]]
    clean_reading: ReadOnly[Annotated[Float[Array, "3"], Units("m/s")]]
    #: Route 1's modes (``(modes, 3)``); a row of zeros on route 2.
    history: ReadOnly[Annotated[Float[Array, "modes 3"], Units("m/s")]]  # noqa: F722 - a jaxtyping shape
    #: Route 2's own field on its free box, each face set's transform's real and imaginary parts (``(2, 3, n, n, n//2 +
    #: 1)``), and Hasimoto's image term beside it; one entry each on route 1, one-way or under the steady limit.
    own_field: ReadOnly[Annotated[Float[Array, "2 3 n n m"], Units("m/s")]]  # noqa: F722 - a jaxtyping shape
    own_image: ReadOnly[Annotated[Float[Array, "3"], Units("m/s")]]
    resistance: ReadOnly[Float[Array, "6 6"]]  # noqa: F722 - a jaxtyping shape
    background_load: ReadOnly[Float[Array, "6"]]
    added_resistance: ReadOnly[Float[Array, "6 6"]]  # noqa: F722 - a jaxtyping shape
    added_load: ReadOnly[Float[Array, "6"]]
    steps: ReadOnly[Annotated[Float[Array, ""], Units("1")]]
    #: The law's own dissipation over the run, ``sum dt (f . (V - G_F u) + T . (W - Omega))`` with each force and couple
    #: against the slip from the raw reading it was felt at (the step before's, summed a step late): the drag's
    #: dissipation less the fluid's own viscous dissipation of the kernel's field (``M_inf f . f``), which the fluid's
    #: row holds. The law's paired work on the body and on the fluid plus this is the lagged exchange's unpaired work,
    #: first order in dt (summed over a fixed time).
    dissipated: ReadOnly[Annotated[Float[Array, ""], Units("J")]]
    #: The law's power at one velocity, ``F . V + f . u_d`` (the load it delivered to the body at the body's solved
    #: twist, and its reaction against the reading it was solved with, couple likewise): ``-R0 |V - u_d|^2 - R_r |W -
    #: Omega_d|^2`` for the law, never above its rounding floor.
    power: ReadOnly[Annotated[Float[Array, ""], Units("W")]]
    cluster_gift_steps: ReadOnly[Annotated[Float[Array, ""], Units("1")]]
    margin: ReadOnly[Annotated[Float[Array, ""], Units("1")]]
    couple_margin: ReadOnly[Annotated[Float[Array, ""], Units("1")]]
    below_floor: ReadOnly[Annotated[Float[Array, ""], Units("1")]]
    history_residual: ReadOnly[Annotated[Float[Array, ""], Units("1")]]
    history_flagged: ReadOnly[Annotated[Float[Array, ""], Units("1")]]
    wall_gap: ReadOnly[Annotated[Float[Array, ""], Units("1")]]
    near_wall: ReadOnly[Annotated[Float[Array, ""], Units("1")]]
    particle_reynolds: ReadOnly[Annotated[Float[Array, ""], Units("1")]]
    reynolds_flagged: ReadOnly[Annotated[Float[Array, ""], Units("1")]]
    rate_load: ReadOnly[Annotated[Float[Array, "3"], Units("N")]]
    ringing: ReadOnly[Annotated[Float[Array, ""], Units("1")]]
    rate_feedback: ReadOnly[Annotated[Float[Array, ""], Units("1")]]


class PointDragInputs(TypedDict, closed=True):
    pose: NotRequired[ReadOnly[Float[Array, "7"]]]
    twist: NotRequired[ReadOnly[Float[Array, "6"]]]
    gathered_x: NotRequired[ReadOnly[Float[Array, "1"]]]
    gathered_y: NotRequired[ReadOnly[Float[Array, "1"]]]
    gathered_z: NotRequired[ReadOnly[Float[Array, "1"]]]
    curvature_x: NotRequired[ReadOnly[Float[Array, "1"]]]
    curvature_y: NotRequired[ReadOnly[Float[Array, "1"]]]
    curvature_z: NotRequired[ReadOnly[Float[Array, "1"]]]
    rotation_x: NotRequired[ReadOnly[Float[Array, "1 3"]]]  # noqa: F722 - a jaxtyping shape
    rotation_y: NotRequired[ReadOnly[Float[Array, "1 3"]]]  # noqa: F722 - a jaxtyping shape
    rotation_z: NotRequired[ReadOnly[Float[Array, "1 3"]]]  # noqa: F722 - a jaxtyping shape


class PointDragNode(MimeNode):
    """A point sphere's drag, couple, fluid-acceleration force and added mass in a flow on a grid, coupled two-way
    through the exchange law and lagged (the module states the law, the readings and the checks)."""

    meta = NodeMeta(
        algorithm_id="MIME-NODE-130",
        description="A point sphere in a flow solved on a grid: Stokes drag and couple with Faxén's corrections on the "
                    "flow read through the exchange kernel less the kernel's own field, the fluid-acceleration force "
                    "and the added mass on the flow without the body, its drag's and couple's reactions scattered back "
                    "onto the grid.",
        governing_equations=EQUATIONS,
        assumptions=("A sphere small against the flow's scales, its particle Reynolds number small: Stokes drag and "
                     "couple, Faxén's corrections and Maxey and Riley's terms; no Basset history force (a validity "
                     "line)",
                     "The flow's solver fills the body's volume: the fluid-acceleration force and the added mass are "
                     "not delivered back to it",
                     "Lagged: the fluid's step takes the drag and the couple the body felt in the step before, at its "
                     "point at this step's start"),
        limitations=("No torque from the fluid's rotational acceleration (the displaced fluid's angular inertia, of "
                     "order (a^2/nu)/T against the viscous couple)",
                     "The drag's reading keeps the grid's undeveloped self-field as a history term: the lagged first "
                     "step resists R0 (1 + lambda^2/9) against the exact sphere's R0 (1 + lambda + lambda^2/9), lambda "
                     "= a/sqrt(nu dt)"),
        references=(Reference("MaxeyRiley1983", "the equation of motion of a small sphere in a nonuniform flow"),
                    Reference("Faxen1922", "the corrections for the flow's curvature"),
                    Reference("MaxeyPatel2001", "a force distributed by a Gaussian kernel on a grid"),
                    Reference("LomholtMaxey2003", "a couple distributed by a Gaussian kernel's dipole")),
    )
    symbols = (Symbol(r"\mathbf{f}", SymbolKind.STATE, "force_x", "N", _STEP),
               Symbol(r"\mu", SymbolKind.PARAMETER, "viscosity", "Pa s", _STEP),
               Symbol("a", SymbolKind.PARAMETER, "radius", "m", _STEP),
               Symbol(r"\mathbf{V}", SymbolKind.INPUT, "twist", "m/s; rad/s", _STEP),
               Symbol(r"\mathbf{u}", SymbolKind.STATE, "drag_reading", "m/s", _STEP))
    #: The class each part of the law's deliveries is booked in (``<id>.<part>``, a row of the energy ledger beside
    #: the drag's): the fluid-acceleration force and the added mass exchange kinetic energy with the fluid the solver
    #: does not hold for them, neither dissipative nor a potential, so their work is recorded under their own name and
    #: not judged (until I2's delivered inertia takes the added mass).
    DELIVERY_CLASSES = {"maxey_riley": "unmetered"}

    def __init__(self, name: str, timestep: float, *, radius: float, viscosity: float, density: float,
                 spacing: float, width_cells: float, origin: Sequence[float], initial_points: Sequence[float],
                 faxen: bool = True, maxey_riley: bool = True, added_mass: bool = True, one_way: bool = False,
                 scheme: str = "imex_euler", self_history: str = "exact", flow_cells: int = 0,
                 vessel_centre: Sequence[float] = (), vessel_axis: Sequence[float] = (),
                 vessel_radius: float = 0.0, drive_frequency: float = 0.0) -> None:
        if scheme not in ("imex_euler", "steady") or self_history not in ROUTES:
            raise ValueError(f"a point body's field steps by 'imex_euler' or 'steady', and its self-history is one of "
                             f"{ROUTES}: not {scheme!r}, {self_history!r}")
        modes = self_history == "modes"
        super().__init__(name, timestep, radius=float(radius), viscosity=float(viscosity), density=float(density),
                         spacing=float(spacing), width_cells=float(width_cells),
                         origin=tuple(float(c) for c in origin),
                         initial_points=tuple(float(c) for c in initial_points),
                         faxen=bool(faxen), maxey_riley=bool(maxey_riley), added_mass=bool(added_mass),
                         one_way=bool(one_way), scheme=scheme, self_history=self_history,
                         **({"flow_cells": int(flow_cells)} if flow_cells else {}),
                         self_responses=as_run_array(jnp.zeros(len(RESPONSES))),
                         **({"history_amplitude": as_run_array(jnp.zeros(HISTORY_MODES + 1)),
                             "history_decay": as_run_array(jnp.zeros(HISTORY_MODES + 1)),
                             "history_fit": as_run_array(jnp.zeros(2))} if modes else {}),
                         **({"vessel_centre": tuple(float(c) for c in vessel_centre),
                             "vessel_axis": tuple(float(c) for c in vessel_axis),
                             "vessel_radius": float(vessel_radius)} if vessel_radius > 0.0 else {}),
                         **({"drive_frequency": float(drive_frequency)} if drive_frequency > 0.0 else {}))
        self.spectra = self.made_spectra()

    def made_spectra(self) -> dict[str, KernelSpectrum]:
        """The kernel's free-box spectra the responses sum over, aggregated by eigenvalue (``aggregated``) where the
        node is built: the force's, the couple's and, on route 1, the places in the cell its fit is judged at. Functions
        of the kernel's width and the box alone, structural; every leaf the responses read stays traced."""
        width, side = float(self.constants()["width_cells"]), self.side
        dtype = str(jnp.zeros(()).dtype)
        held = {"force": held_spectrum("force", width, side, dtype),
                "couple": held_spectrum("couple", width, side, dtype)}
        if self.route == "modes":
            held.update({f"place_{face}_{i}": held_spectrum("force", width, side, dtype, face, place)
                         for face in range(3) for i, place in enumerate(JUDGED_PLACES)})
        return held

    @override
    def param_specs(self) -> dict[str, ParamSpec]:
        structural = "a structural setting: the exchange edges' kernel was built with it"
        made = "a run-constant the node makes from its leaves once a chunk (run_constants); the value held is a stand-in"
        specs = {**super().param_specs(),
                 "radius": ParamSpec(units="m", bounds=(0.0, None), description="the sphere's radius"),
                 "viscosity": ParamSpec(units="Pa s", bounds=(0.0, None)),
                 "density": ParamSpec(units="kg/m^3", bounds=(0.0, None), description="the fluid's density"),
                 "spacing": ParamSpec(trainable=False, units="m", description=structural),
                 "width_cells": ParamSpec(trainable=False, units="1", description=structural),
                 "initial_points": ParamSpec(trainable=False, units="m",
                                             description="where the body starts, the exchange edges' first points: an "
                                                         "initial condition, which reaches the run through the state"),
                 "self_responses": ParamSpec(trainable=False, units="1", description=made)}
        if self.route == "modes":
            specs.update({key: ParamSpec(trainable=False, units="1", description=made)
                          for key in ("history_amplitude", "history_decay", "history_fit")})
        return specs

    @property
    def route(self) -> str:
        """The self-history's route: ``exact`` (route 2, the default) or ``modes`` (route 1)."""
        return str(self.constants()["self_history"])

    @property
    def holds_own_field(self) -> bool:
        """Whether the node advances its own free-box field (route 2, two-way, stepped in time)."""
        c = self.constants()
        return self.route == "exact" and not c["one_way"] and c["scheme"] == "imex_euler"

    @property
    def side(self) -> int:
        """Route 2's free box's side, and the spectral sums', in cells (``box_side``)."""
        return box_side(float(self.constants()["width_cells"]))

    @override
    def initial_state(self) -> NodeFields:
        c = self.constants()
        zero = as_run_array(jnp.zeros(()))
        one = as_run_array(jnp.zeros(1))
        three = as_run_array(jnp.zeros(3))
        start = as_run_array(jnp.asarray(c["initial_points"])) + as_run_array(jnp.asarray(c["origin"]))
        side = self.side
        field = (2, 3, side, side, side // 2 + 1) if self.holds_own_field else (2, 3, 1, 1, 1)
        modes = HISTORY_MODES + 1 if self.route == "modes" else 1
        return node_fields(PointDragState(
            points=start[None, :], force_x=one, force_y=one, force_z=one, faxen_x=one, faxen_y=one, faxen_z=one,
            couple=as_run_array(jnp.zeros((1, 3))), drag_reading=three, raw_reading=three, rotation_reading=three,
            raw_rotation=three, clean_reading=three, history=as_run_array(jnp.zeros((modes, 3))),
            own_field=as_run_array(jnp.zeros(field)), own_image=three, resistance=as_run_array(jnp.zeros((6, 6))),
            background_load=as_run_array(jnp.zeros(6)), added_resistance=as_run_array(jnp.zeros((6, 6))),
            added_load=as_run_array(jnp.zeros(6)), steps=zero, dissipated=zero, power=zero, cluster_gift_steps=zero,
            margin=zero, couple_margin=zero, below_floor=zero, history_residual=zero, history_flagged=zero,
            wall_gap=zero, near_wall=zero, particle_reynolds=zero, reynolds_flagged=zero, rate_load=three,
            ringing=zero, rate_feedback=zero))

    @override
    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        inputs = {"pose": node_input("m; 1", (7,)), "twist": node_input("m/s; rad/s", (6,))}
        for field in AVERAGES:
            inputs[field] = node_input("m/s", (1,), description="the flow read by the exchange kernel's average")
        for field in LAPLACIANS:
            inputs[field] = node_input("1/(m s)", (1,), description="the flow's Laplacian read by the kernel")
        for field in ROTATIONS:
            inputs[field] = node_input("rad/s", (1, 3), description="one face set's share of the fluid's rotation, "
                                                                    "read by the kernel's antisymmetric dipole")
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
        """The kernel's own field by the grid's operators in a free box, from the leaves (traced in the radius, the
        viscosity and the density): the force's and the couple's steady self-responses the readings subtract, their
        one-step responses (the steady ones under the field's steady limit) and coupled margins, the own field's
        steady response, Hasimoto's image term as a mode, and on route 1 the self-history's fitted modes."""
        c = self.constants()
        h = float(c["spacing"])
        mu, rho, a = as_run_array(p["viscosity"]), as_run_array(p["density"]), as_run_array(p["radius"])
        steady_limit = c["scheme"] == "steady"
        r = mu / rho * dt / h ** 2
        drag, average, delivery = self.coefficients(p)
        side = self.side
        spectrum, dipole = self.spectra["force"], self.spectra["couple"]
        scale, turn = 1.0 / (mu * h), 1.0 / (mu * h ** 3)
        steady = steady_response(spectrum, drag / h ** 2, delivery / h ** 2) * scale
        one_step = steady if steady_limit else \
            step_response(spectrum, r, jnp.zeros(1, dtype=mu.dtype), drag / h ** 2, delivery / h ** 2)[0] * scale
        couple_steady = steady_response(dipole, images=False) * turn
        couple_one_step = couple_steady if steady_limit else step_response(dipole, r, jnp.zeros(1, dtype=mu.dtype))[0] \
            * turn
        mobility, spin = 1.0 / (6.0 * jnp.pi * mu * a), 1.0 / (8.0 * jnp.pi * mu * a ** 3)
        image, image_decay = image_mode(r, side)
        made: dict[str, Any] = {
            "steady": steady, "one_step": one_step, "margin": (mobility + one_step - steady) / mobility,
            "steady_clean": steady_response(spectrum, average / h ** 2, delivery / h ** 2) * scale,
            "couple_steady": couple_steady, "couple_one_step": couple_one_step,
            "couple_margin": (spin + couple_one_step - couple_steady) / spin,
            "image": jnp.asarray(image, dtype=mu.dtype) * scale, "image_decay": jnp.asarray(image_decay, dtype=mu.dtype)}
        if self.route == "modes":
            places = [held for key, held in self.spectra.items() if key.startswith("place_")]
            history = self_history(spectrum, r, average / h ** 2, delivery / h ** 2, places)
            made.update({"amplitude": history.amplitude * scale, "decay": history.decay,
                         "residual": history.residual, "span": history.span})
        return made

    @override
    def run_constants(self, params: Mapping[str, Any]) -> dict[str, Array]:
        """The kernel's own responses (and route 1's modes) from the run's leaves, made once a chunk in place of their
        stand-ins: functions of the radius, the viscosity, the density and the step alone, constant over a run."""
        made = self.responses(params, self.delta_t)
        held = jnp.asarray(params["self_responses"])
        found = {"self_responses": jnp.stack([jnp.asarray(made[key]) for key in RESPONSES]).astype(held.dtype)}
        if "history_amplitude" in params:
            dtype = jnp.asarray(params["history_amplitude"]).dtype
            found.update({"history_amplitude": made["amplitude"].astype(dtype),
                          "history_decay": made["decay"].astype(dtype),
                          "history_fit": jnp.stack([made["residual"], made["span"]]).astype(dtype)})
        return found

    def made(self, p: Mapping[str, Any], dt: float) -> dict[str, Any]:
        """The step's responses: as its chunk made them (:meth:`run_constants`), else made here."""
        if not run_constants_given(p):
            return self.responses(p, dt)
        held = jnp.asarray(p["self_responses"])
        made: dict[str, Any] = {key: held[i] for i, key in enumerate(RESPONSES)}
        if "history_amplitude" in p:
            fit = jnp.asarray(p["history_fit"])
            made.update({"amplitude": jnp.asarray(p["history_amplitude"]), "decay": jnp.asarray(p["history_decay"]),
                         "residual": fit[0], "span": fit[1]})
        return made

    def stepped(self, p: Mapping[str, Any], state: Mapping[str, Array], boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        c = self.constants()
        mu, rho, a = as_run_array(p["viscosity"]), as_run_array(p["density"]), as_run_array(p["radius"])
        dtype = mu.dtype
        pose = jnp.asarray(boundary_inputs.get("pose", jnp.zeros(7, dtype=dtype)), dtype=dtype)
        twist = jnp.asarray(boundary_inputs.get("twist", jnp.zeros(6, dtype=dtype)), dtype=dtype)

        def read(fields: Sequence[str], shape: tuple[int, ...]) -> list[Array]:
            return [jnp.asarray(boundary_inputs[f], dtype=dtype) if f in boundary_inputs else
                    jnp.zeros(shape, dtype=dtype) for f in fields]

        average = jnp.concatenate(read(AVERAGES, (1,)))
        laplacian = jnp.concatenate(read(LAPLACIANS, (1,)))
        rotation = sum(read(ROTATIONS, (1, 3)), jnp.zeros((1, 3), dtype=dtype))[0]
        drag_c, average_c, delivery_c = self.coefficients(p)
        made = self.made(p, dt)
        two_way = not c["one_way"]
        last = jnp.concatenate([jnp.asarray(state[f], dtype=dtype) for f in FORCES])
        last_couple = jnp.asarray(state[COUPLE], dtype=dtype)[0]
        points = (pose[:3] + as_run_array(p["origin"]))[None, :]
        # The drag's and the couple's readings (the constant correction), and the flow without the body's.
        raw = average + drag_c * laplacian
        drag_reading = raw - (made["steady"] * last if two_way else 0.0)
        rotation_reading = rotation - (made["couple_steady"] * last_couple if two_way else 0.0)
        own_field, own_image = jnp.asarray(state["own_field"], dtype=dtype), jnp.asarray(state["own_image"], dtype=dtype)
        history = jnp.asarray(state["history"], dtype=dtype)
        kernels: list[Array] = []
        box = None
        if not two_way:
            own = jnp.zeros(3, dtype=dtype)
        elif c["scheme"] == "steady":
            own = made["steady_clean"] * last
        elif self.route == "modes":
            own = jnp.sum(history, axis=0)
        else:
            box = own_box(self.side, float(c["spacing"]), dtype)
            kernels = own_kernels(box, points[0], float(c["width_cells"]))
            own = own_read(box, own_field, kernels, average_c) + own_image
        clean = average + average_c * laplacian - own
        started = jnp.asarray(state["steps"]) > 0.0
        rate = jnp.where(started, (clean - jnp.asarray(state["clean_reading"], dtype=dtype)) / dt, 0.0)
        velocity, spin = twist[:3], twist[3:]
        r0 = 6.0 * jnp.pi * mu * a
        rotational = 8.0 * jnp.pi * mu * a ** 3
        fluid_mass = 4.0 / 3.0 * jnp.pi * a ** 3 * rho
        added = 0.5 * fluid_mass if c["maxey_riley"] and c["added_mass"] else jnp.zeros((), dtype=dtype)
        share = fluid_mass if c["maxey_riley"] else jnp.zeros((), dtype=dtype)
        eye = jnp.eye(3, dtype=dtype)
        resistance = jnp.zeros((6, 6), dtype=dtype).at[:3, :3].set(r0 * eye).at[3:, 3:].set(rotational * eye)
        background = jnp.concatenate([r0 * drag_reading, rotational * rotation_reading])
        added_resistance = jnp.zeros((6, 6), dtype=dtype).at[:3, :3].set(added / dt * eye)
        rate_load = (share + added) * rate
        added_load = jnp.concatenate([added / dt * velocity + rate_load, jnp.zeros(3, dtype=dtype)])
        # The drag and the couple the body felt in the step before, which the fluid's step takes now, at the body's
        # present point.
        before = jnp.concatenate([jnp.asarray(state["drag_reading"], dtype=dtype),
                                  jnp.asarray(state["rotation_reading"], dtype=dtype)])
        felt = jnp.concatenate([r0 * (velocity - before[:3]), rotational * (spin - before[3:])])
        delivered = jnp.where(started, felt, 0.0) if two_way else jnp.zeros(6, dtype=dtype)
        force, couple = delivered[:3], delivered[3:]
        # The law's power at one velocity: the load it gave the body (at the body's solved twist) and its reaction
        # against the readings it was solved with; never positive beyond the products' rounding for a drag.
        held_load = jnp.asarray(state["background_load"], dtype=dtype)
        load = held_load - jnp.asarray(state["resistance"], dtype=dtype) @ twist
        terms = jnp.concatenate([load * twist, felt * before])
        power = jnp.sum(terms)
        floor = IDENTITY_ULPS * jnp.finfo(dtype).eps * (jnp.sum(jnp.abs(terms)) + jnp.abs(held_load) @ jnp.abs(twist))
        gave = started & (power > floor)
        # The law's own dissipation: each force and couple against the slip from the raw reading it was felt at.
        raw_before = jnp.concatenate([jnp.asarray(state["raw_reading"], dtype=dtype),
                                      jnp.asarray(state["raw_rotation"], dtype=dtype)])
        dissipated = jnp.asarray(state["dissipated"]) + jnp.where(
            started, dt * jnp.sum(felt * (twist - raw_before)), 0.0)
        # The own field a step on, with the force the fluid takes now.
        if box is not None:
            decay = 1.0 / (1.0 + mu / rho * dt * box.eigenvalue)
            gain = jnp.where(box.eigenvalue == 0.0, 0.0, dt / rho * decay)
            own_field = own_advance(box, own_field, kernels, force, delivery_c, decay, gain)
            own_image = made["image_decay"] * own_image + made["image"] * force
        if self.route == "modes" and two_way:
            history = made["decay"][:, None] * (history + made["amplitude"][:, None] * force[None, :])
        # The rate terms' load reversing and growing step on step: a lagged loop through the body's own field ringing.
        previous = jnp.asarray(state["rate_load"], dtype=dtype)
        size, before_size = jnp.sum(rate_load * rate_load), jnp.sum(previous * previous)
        # Counted only above the rate load's own rounding: (m_f + m_a) times the readings' rounding over the step.
        noise = IDENTITY_ULPS * jnp.finfo(dtype).eps * (share + added) * (
            jnp.sum(jnp.abs(clean)) + jnp.sum(jnp.abs(jnp.asarray(state["clean_reading"], dtype=dtype)))) / dt
        reversing = started & (jnp.sum(rate_load * previous) < 0.0) & (size > before_size) & (before_size > noise ** 2)
        ringing = jnp.where(reversing, jnp.asarray(state["ringing"]) + 1.0, 0.0)
        relative = jnp.sqrt(jnp.sum((velocity - drag_reading) ** 2))
        reynolds = 2.0 * a * relative * rho / mu
        gap = self.wall_gap(p, pose[:3], a)
        margin = made["margin"].astype(dtype)
        couple_margin = made["couple_margin"].astype(dtype)
        residual = made["residual"].astype(dtype) if "residual" in made else jnp.zeros((), dtype=dtype)
        past_span = (jnp.asarray(state["steps"]) + 1.0 > made["span"]) if "span" in made else jnp.asarray(False)
        ended = PointDragState(
            points=points.astype(dtype), force_x=force[0:1], force_y=force[1:2], force_z=force[2:3],
            faxen_x=delivery_c * force[0:1], faxen_y=delivery_c * force[1:2], faxen_z=delivery_c * force[2:3],
            couple=couple[None, :], drag_reading=drag_reading, raw_reading=raw, rotation_reading=rotation_reading,
            raw_rotation=rotation, clean_reading=clean, history=history, own_field=own_field, own_image=own_image,
            resistance=resistance, background_load=background, added_resistance=added_resistance,
            added_load=added_load, steps=jnp.asarray(state["steps"]) + 1.0, dissipated=dissipated,
            power=jnp.where(started, power, 0.0), cluster_gift_steps=jnp.asarray(state[GIFT_STEPS]) + gave.astype(dtype),
            margin=margin, couple_margin=couple_margin,
            below_floor=(jnp.minimum(margin, couple_margin) < FLOOR).astype(dtype), history_residual=residual,
            history_flagged=((residual > HISTORY_LINE) | past_span).astype(dtype), wall_gap=gap,
            near_wall=(gap < WALL_LINE).astype(dtype), particle_reynolds=_lax.stop_gradient(reynolds),
            reynolds_flagged=(reynolds > REYNOLDS_LINE).astype(dtype), rate_load=rate_load, ringing=ringing,
            rate_feedback=(ringing >= RINGING_STEPS).astype(dtype))
        return node_fields(ended)

    def own_field_bytes(self) -> int:
        """The bytes route 2's own field holds for this body (its transform's real and imaginary parts, three face
        sets) in the run's dtype."""
        side = self.side
        return 2 * 3 * side * side * (side // 2 + 1) * int(jnp.dtype(as_run_array(jnp.zeros(())).dtype).itemsize)

    def built_lines(self, start: Mapping[str, Any] | None, closest_approach: float | None) -> list[str]:
        """What the build reports of the point body at its own values: the kernel's width and why, the coupled one-step
        margins against the floor, the self-history's route (route 1 labelled fitted, with its residual and span), the
        body's size against the cell, and the Basset number of the drive's frequency where one is given."""
        del start, closest_approach
        c = self.constants()
        made = self.responses(c, self.delta_t)
        a, h, width = float(c["radius"]), float(c["spacing"]), float(c["width_cells"])
        least = max(1.0, a / (1.7 * h))
        r = float(c["viscosity"]) / float(c["density"]) * self.delta_t / h ** 2
        steady_line = " (the field's steady limit: its one-step response is its steady one)" \
            if c["scheme"] == "steady" else ""
        lines = [f"the exchange kernel's width: {width:g} cells ({width * h:.4g} m), the narrowest of the ladder at or "
                 f"above max(1, a/(1.7 h)) = {least:.3g} cells whose coupled one-step margins clear the floor {FLOOR:g}"
                 f" at the run's values" if width >= least - 1e-12 else
                 f"the exchange kernel's width: {width:g} cells, written",
                 f"the coupled one-step margins (M0 + M_c - M_inf)/M0 = {float(made['margin']):.4g} (the force) and "
                 f"{float(made['couple_margin']):.4g} (the couple) in free space at r = nu dt/h^2 = {r:.4g}"
                 f"{steady_line} (floor {FLOOR:g}; the wall lowers the force's, by up to 0.15 at half a radius at r = 3 "
                 f"in the floor sweep)"]
        if c["one_way"]:
            lines.append("one-way: the rate terms read the flow as it is, the body's own field being none")
        elif c["scheme"] == "steady":
            lines.append("the self-history the rate terms remove: under the field's steady limit, the steady response "
                         "to last step's force, exactly (either route)")
        elif self.route == "modes":
            lines.append(f"the self-history the rate terms remove is fitted (self_history='modes'): "
                         f"{HISTORY_MODES} decay modes, fixed log-spaced rates, amplitudes by least squares over the "
                         f"free box's response ({self.side} cells a side) up to step {float(made['span']):.4g}, and "
                         f"Hasimoto's image term; its residual {float(made['residual']):.2g} of the one-step response "
                         f"there (flagged above {HISTORY_LINE:g}, and past step {float(made['span']):.4g})")
        else:
            held = self.own_field_bytes()
            flow = 3 * int(c.get("flow_cells", 0)) * held // (2 * 3 * self.side ** 2 * (self.side // 2 + 1))
            costly = flow and held > flow
            lines.append(f"the self-history the rate terms remove is exact (route 2): the body's own flow on a free box "
                         f"of {self.side} cells a side advanced by the grid's operator each step, with Hasimoto's image "
                         f"term, {held / 2 ** 20:.3g} MiB held for it (and recorded each step by a run with "
                         f"history=True)" + (f", more than the flow's own velocity field ({flow / 2 ** 20:.3g} MiB): "
                                             f"with many bodies, self_history='modes' (route 1, {HISTORY_MODES + 1} "
                                             f"fitted modes a body, its gate the swap test against this) is the "
                                             f"cheaper route" if costly else
                                             f"; self_history='modes' holds {HISTORY_MODES + 1} fitted modes instead "
                                             f"(route 1, its gate the swap test against this)"))
        lines.append(f"the body's radius is {a / h:.3g} cells: a point model (a resolved body, Drag.Penalty, when it is "
                     f"many cells across)")
        if c.get("drive_frequency", 0.0):
            number = a * math.sqrt(2.0 * math.pi * float(c["drive_frequency"]) * float(c["density"])
                                   / float(c["viscosity"]))
            lines.append(f"the Basset number a sqrt(omega/nu) = {number:.3g} at the medium's drive's frequency (the "
                         f"history force against Stokes drag scales as it; no memory law is held: reported against "
                         f"the line {BASSET_LINE:g}, not flagged; a body driven by a field in still fluid, or any "
                         f"transient, has no such number here)" + (" - ABOVE THE LINE" if number > BASSET_LINE else ""))
        return lines

    def wall_gap(self, p: Mapping[str, Any], position: Array, a: Array) -> Array:
        """The body's gap to the vessel's wall in its radii (a report, held out of derivatives); a large number with
        no vessel."""
        if "vessel_radius" not in self.constants():
            return jnp.asarray(1e30, dtype=position.dtype)
        centre = jnp.asarray(p["vessel_centre"], dtype=position.dtype)
        axis = jnp.asarray(p["vessel_axis"], dtype=position.dtype)
        axis = axis / jnp.sqrt(jnp.sum(axis * axis))
        offset = position - centre
        across = offset - jnp.dot(offset, axis) * axis
        gap = (jnp.asarray(p["vessel_radius"], dtype=position.dtype) - jnp.sqrt(jnp.sum(across * across)) - a) / a
        return _lax.stop_gradient(gap)


#: The kernel widths, in cells, the stokes rung chooses among: the narrowest at or above max(1, a/(1.7 h)) whose
#: coupled one-step margins clear the floor at the run's values.
KERNEL_WIDTHS: tuple[float, ...] = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)


def kernel_width(radius: float, viscosity: float, density: float, spacing: float, dt: float, steady: bool,
                 written: float | None = None, ladder: Sequence[float] = KERNEL_WIDTHS
                 ) -> tuple[float | None, float, tuple[float, ...]]:
    """The exchange kernel's width in cells a point body of ``radius`` takes: ``written``, or the narrowest on
    ``ladder`` at or above max(1, a/(1.7 h)) whose free-space coupled one-step margins (the force's and the couple's)
    clear the floor; with the least margin read there and the widths tried. ``None`` where none clears it. The drag's
    node and a flow holding the body in its step read the one width from here."""
    least = max(1.0, radius / (1.7 * spacing))
    tried_ladder = (float(written),) if written is not None else tuple(
        w for w in ladder if w >= least - 1e-12) or (ladder[-1],)
    margin = float("nan")
    tried: list[float] = []
    for width in tried_ladder:
        tried.append(width)
        margin = min(free_space_margins(radius, viscosity, density, spacing, width, dt, steady))
        if margin >= FLOOR and width >= least - 1e-12:
            return width, margin, tuple(tried)
    return None, margin, tuple(tried)


def free_space_margins(radius: float, viscosity: float, density: float, spacing: float, width_cells: float, dt: float,
                       steady_limit: bool = False, faxen_on: bool = True) -> tuple[float, float]:
    """The coupled one-step margins a point body of ``radius`` reads in a free box, the force's and the couple's (the
    build's check, at its own values), with the Faxén pair or without; both one under the field's steady limit."""
    if steady_limit:
        return 1.0, 1.0
    width = width_cells * spacing
    drag = (radius ** 2 / 6.0 - 0.5 * width ** 2) / spacing ** 2 if faxen_on else 0.0
    side = box_side(width_cells)
    dtype = str(jnp.zeros(()).dtype)
    spectrum = held_spectrum("force", width_cells, side, dtype)
    dipole = held_spectrum("couple", width_cells, side, dtype)
    r = viscosity / density * dt / spacing ** 2
    zero = as_run_array(jnp.zeros(1))
    steady = float(steady_response(spectrum, drag, drag)) / (viscosity * spacing)
    one_step = float(step_response(spectrum, r, zero, drag, drag)[0]) / (viscosity * spacing)
    turn = 1.0 / (viscosity * spacing ** 3)
    couple_steady = float(steady_response(dipole, images=False)) * turn
    couple_one_step = float(step_response(dipole, r, zero)[0]) * turn
    mobility, spin = 1.0 / (6.0 * math.pi * viscosity * radius), 1.0 / (8.0 * math.pi * viscosity * radius ** 3)
    return (mobility + one_step - steady) / mobility, (spin + couple_one_step - couple_steady) / spin


__all__ = ["AVERAGES", "BASSET_LINE", "COUPLE", "FAXEN_FORCES", "FLOOR", "FORCES", "GIFT_STEPS", "HISTORY_LINE",
           "KERNEL_WIDTHS", "LAPLACIANS", "REYNOLDS_LINE", "RESPONSES", "RINGING_STEPS", "ROTATIONS", "ROUTES", "WALL_LINE",
           "PointDragInputs", "PointDragNode", "PointDragState", "free_space_margins", "kernel_width"]
