"""The implicit flow node on a box open along its axis: the FVM's field in a straight vessel cut at its two ends, each
end an opening whose law the step holds (F0's B6/B7; the record's "F2's rows, 2"; F2b's PR 2).

**A step** is :class:`ImplicitFlowNode`'s (IMEX Euler or its steady limit on the monolithic projected CG), on the open
grid (``mime.nodes.flow.fvm._open``), with its openings coupled implicitly through the field's affine response
(decision 11): the step's solve takes the openings' forces at the pressures the last step solved for (relative to the
gauge's reference opening, decision 9), then the openings' rows (``mime.nodes.flow.fvm._ports``) are solved with the
unit fields' response, and the solution is the solve's plus each unit field times its pressure's correction. **The unit
fields are made in the program** (round 5, item 13): one a non-reference opening, the response to a unit pressure there,
held in the state and solved again at every step from the last (a warm start that converges in an iteration or two
while the operator is fixed), so that every step's are those of its own parameters and a derivative passes through
them. Each law then steps its own state with its opening's solved flux and pressure.

**Its step identity** (route 2, paired at the solution ``u'``): ``dKE = W_drive + W_ports + W_KE - D_visc - D_wall -
N_BE - N_conv - N_Krylov - N_proj``: each opening's law's work ``-dt p q`` under its own name (its outward flux ``q``
and its absolute pressure ``p``: the field's forces are the pressures relative to the reference, whose difference is
the reference's pressure times the net flux, zero to rounding); the kinetic energy convection carries through the
openings, ``W_KE = dt rho <C(u), u>``, the scheme's own convective power (B9, decision 14), as one line under
``<node>.openings``; ``N_conv = -dt rho <C(u), u' - u>``, convection at mixed time levels; ``N_Krylov`` the solve's
projected remainder against ``u'`` and ``N_proj = dt rho <G p', u'>`` its gradient part, ``p'`` the remainder's
pressure: the half-weight faces keep the projector orthogonal, so ``N_proj`` is rounding (C13), booked apart so that its
flag reads it. A law storing energy of its own (the Windkessel) keeps its own books beside the field's (its storage at
the step's start and end, its loss, its ``N_WK``: ``mime.core.ledger.law_records``), none of them the field's."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple, override

import numpy as np
from jax import Array
from jaxtyping import Float
from maddening.core.compliance.metadata import NodeMeta, Reference
from typing_extensions import Annotated, ReadOnly, TypedDict

from mime.core import NodeFields, Symbol, SymbolKind, Units, as_run_array, jnp, node_fields, run_dtype
from mime.core import ledger as books
from mime.nodes.flow.fvm._band import TubeSolid, WallFields, band_named, negative_resistivity
from mime.nodes.flow.fvm._flags import OPEN_FLAGS
from mime.nodes.flow.fvm._grid import MACGrid, Velocity
from mime.nodes.flow.fvm._node import BYPASS_LINE, STALL_LINE, STOKES_REYNOLDS, ImplicitFlowNode, StepParts, \
    check_deliveries
from mime.nodes.flow.fvm._open import OpenMACGrid
from mime.nodes.flow.fvm._ports import PARAMETERS, STATES, PortLaw, port_law_named, port_system, reference_of, \
    well_posed
from mime.nodes.flow.fvm._schemes import StageParts, StepContext
from mime.nodes.flow.fvm._solve import Stopping, preconditioner_named, projected_solve, stagnation_window

#: The roles the open node's hosted rows take beside its viscosity's and wall's: each opening's law, by its index,
#: and the kinetic energy convection carries through the openings.
KINETIC_FLUX = "kinetic_flux"
#: The open node's numerical terms: the scheme's, and the projection's own at the open faces.
OPEN_TERMS = ("backward_euler", "convection", "krylov", "projection")
OPENING = "opening_"
#: The float32 flow's distance from float64's with ports (the open-ports spike, D/h 16, measured 1.6e-6 to 3.5e-6).
FLOAT32_PORTS = "the flow within 1.6e-6 to 3.5e-6 of float64's (the open-ports spike, measured at D/h 16)"


def opening_role(index: int) -> str:
    return f"{OPENING}{index}"


class OpenFlowFields(TypedDict):
    """The open node's own fields beside the periodic node's (``FlowState``'s), with their units."""

    #: (``w``'s faces include the two planes: ``velocity_z`` is ``nx ny nz+1``.) The openings' pressures (absolute)
    #: and outward fluxes the last step solved for, and each law's own state.
    opening_pressure: ReadOnly[Annotated[Float[Array, "openings"], Units("Pa")]]
    opening_flux: ReadOnly[Annotated[Float[Array, "openings"], Units("m^3/s")]]
    opening_state: ReadOnly[Annotated[Float[Array, "openings 1"], Units("1")]]  # noqa: F722 - a jaxtyping shape
    #: The unit fields, one a non-reference opening: the velocity a unit pressure there drives over a step (m/s/Pa).
    unit_x: ReadOnly[Annotated[Float[Array, "units nx ny nz"], Units("m/(s Pa)")]]  # noqa: F722 - a jaxtyping shape
    unit_y: ReadOnly[Annotated[Float[Array, "units nx ny nz"], Units("m/(s Pa)")]]  # noqa: F722 - a jaxtyping shape
    unit_z: ReadOnly[Annotated[Float[Array, "units nx ny nzp1"], Units("m/(s Pa)")]]  # noqa: F722 - a jaxtyping shape


class OpenPorts(NamedTuple):
    """What the step's port coupling solved: the openings' pressures and the fluxes its rows hold, the solve's own
    solution before the correction, the unit fields, their response, and the forces at the solved pressures."""

    pressures: Array
    fluxes: Array
    base: Velocity
    units: tuple[Velocity, ...]
    response: Array
    force: Velocity


def _velocity(state: Mapping[str, Array]) -> Velocity:
    return (state["velocity_x"], state["velocity_y"], state["velocity_z"])


def _add(a: Velocity, b: Velocity, scale: Any = 1.0) -> Velocity:
    return (a[0] + scale * b[0], a[1] + scale * b[1], a[2] + scale * b[2])


#: The openings' conditions, the periodic node's momentum equation (``ImplicitFlowNode``'s) holding inside.
EQUATIONS = (r"\operatorname{p}_{n+1} = p_{k}, \quad p_{k} = \operatorname{a}_{k} + \operatorname{b}_{k} q_{k} \ \text{or} "
             r"\ q_{k} = \operatorname{Q}_{k}")
_STEPPED = "mime.nodes.flow.fvm._open_node.OpenFlowNode.stepped"


class OpenFlowNode(ImplicitFlowNode):
    """The flow field of a medium in a straight vessel along z cut at its two ends, each an opening whose law the step
    holds, stepped by IMEX Euler (or its steady limit) on the monolithic projected CG (the module states the step)."""

    meta = NodeMeta(
        algorithm_id="MIME-NODE-129",
        description="A medium's flow in a straight vessel open at its two ends: the periodic flow node's step on the "
                    "Cartesian MAC grid open along z, each end an opening whose law (an inflow, a Windkessel) the "
                    "step holds, coupled implicitly through the field's affine response.",
        governing_equations=EQUATIONS,
        assumptions=(
            "Inside, the periodic flow node's (ImplicitFlowNode's) momentum equation and its wall's band",
            "The grid is open along z: the faces on its two planes are evolved half cells, weighted by a half in the "
            "inner product, so the projector stays orthogonal; the pressure's field part is zero on the planes, whose "
            "pressures the openings' forces carry",
            "At an open plane the viscous flux is zero and convection's flux is the plane's own face value",
            "Each opening k's law states its pressure affine in its outward volume flux over the stage (a, b), or the "
            "flux itself (Q); the field's pressure on the opening's plane is its pressure, seen relative to one "
            "reference opening (the gauge)",
            "The unit fields are solved in the program each step from the last, while the operator is fixed"),
        limitations=(
            "A straight vessel along z only, one opening at each plane",
            "IMEX Euler or its steady limit: RK3 with openings is not built",
            "No backflow stabilisation at an outlet yet: a reversing outlet at high Reynolds number is unguarded",
        ),
        references=(Reference("HarlowWelch1965"), Reference("Westerhof2009")),
    )
    symbols = (
        Symbol(r"p", SymbolKind.STATE, "opening_pressure", "Pa", _STEPPED),
        Symbol(r"q", SymbolKind.STATE, "opening_flux", "m^3/s", _STEPPED),
    )

    LEDGER_TERMS = OPEN_TERMS
    GRID_TERMS = ("projection",)
    LEDGER_FLAGS = OPEN_FLAGS

    def __init__(self, name: str, timestep: float, *, openings: Sequence[str], opening_laws: Sequence[str],
                 opening_parameters: Any, opening_law_ids: Sequence[str] = (), initial_opening_state: Any = None,
                 **settings: Any) -> None:
        laws = [port_law_named(str(law)) for law in opening_laws]
        if len(tuple(openings)) != 2 or len(laws) != 2:
            raise ValueError(f"an open box has two openings, its z planes, each with its law: not {tuple(openings)} "
                             f"with {tuple(opening_laws)}")
        parameters = np.asarray(opening_parameters, dtype=np.float64)
        if parameters.shape != (2, PARAMETERS):
            raise ValueError(f"each opening's law takes a row of {PARAMETERS} parameters, not {parameters.shape}")
        reason = well_posed([law.row for law in laws], [tuple(row) for row in parameters.tolist()])
        if reason:
            raise ValueError(reason)
        if settings.get("periodic_images", "own") != "own" or tuple(settings.get("tube_axis", (0.0, 0.0, 1.0))) != \
                (0.0, 0.0, 1.0):
            raise ValueError("an open box's vessel lies along its z axis (tilted open vessels are not built yet)")
        ids = tuple(str(i) for i in opening_law_ids) or tuple(f"{name}.{opening}" for opening in openings)
        state = np.zeros((2, STATES)) if initial_opening_state is None else np.asarray(initial_opening_state)
        super().__init__(name, timestep, openings=tuple(str(o) for o in openings),
                         opening_laws=tuple(law.name for law in laws), opening_law_ids=ids,
                         opening_parameters=parameters, initial_opening_state=state, **settings)

    # -- the pieces of a step ------------------------------------------------------------------------------------
    @override
    def grid(self, p: Mapping[str, Any]) -> MACGrid:
        c = self.constants()
        return OpenMACGrid((c["cells_x"], c["cells_y"], c["cells_z"]), p["spacing"])

    def open_grid(self, p: Mapping[str, Any]) -> OpenMACGrid:
        c = self.constants()
        return OpenMACGrid((c["cells_x"], c["cells_y"], c["cells_z"]), p["spacing"])

    @override
    def solids(self, p: Mapping[str, Any], grid: MACGrid) -> list[TubeSolid]:
        return [TubeSolid(as_run_array(p["tube_centre"]), as_run_array(p["tube_axis"]), as_run_array(p["tube_radius"]),
                          ((0.0, 0.0, 0.0),))]

    def laws(self) -> list[PortLaw]:
        return [port_law_named(law) for law in self.constants()["opening_laws"]]

    def reference(self) -> int:
        return reference_of([law.row for law in self.laws()])

    def others(self) -> list[int]:
        reference = self.reference()
        return [index for index in range(len(self.laws())) if index != reference]

    @override
    def fluxes(self, grid: MACGrid, fields: WallFields, axis: Array, velocity: Velocity) -> tuple[Array, Array, Array]:
        """The lumen's and the solid's axial flux (m^3/s), each the weighted volume integral of ``w`` over its region
        (the planes' faces at half weight) over the box's length, exact for a divergence-free field, and their scale."""
        assert isinstance(grid, OpenMACGrid)
        width = band_named(self.constants()["band"]).width * grid.spacing
        weight = grid.weights(velocity[2].dtype)[2]
        inside = fields.signed_distance[2] < 0.5 * width
        w = velocity[2]
        per = grid.cell_volume / (grid.cells[2] * grid.spacing)
        lumen = per * jnp.sum(weight * jnp.where(inside, w, 0.0))
        solid = per * jnp.sum(weight * jnp.where(inside, 0.0, w))
        scale = per * sum((jnp.sum(jnp.abs(c)) for c in velocity), start=jnp.zeros((), dtype=w.dtype))
        return lumen, solid, scale

    # -- the node's interface -------------------------------------------------------------------------------------
    @override
    def initial_state(self) -> NodeFields:
        c = self.constants()
        grid = self.open_grid(c)
        dtype = run_dtype()
        count = len(self.others())
        unit = {f"unit_{x}": jnp.zeros((count, *grid.shape(k)), dtype=dtype) for k, x in enumerate("xyz")}
        held = {**self.initial_fields(), "opening_pressure": jnp.zeros(2, dtype=dtype),
                "opening_flux": jnp.zeros(2, dtype=dtype),
                "opening_state": jnp.asarray(c["initial_opening_state"], dtype=dtype), **unit}
        if not self.metered:
            return node_fields(held)
        z = as_run_array(0.0)
        return books.with_records(self, {**held, "stall_ratio": z, "stall_flagged": z})

    @override
    def stepped(self, p: Mapping[str, Any], state: Mapping[str, Array], boundary_inputs: Mapping[str, Array],
                dt: float) -> tuple[dict[str, Array], StepParts]:
        c = self.constants()
        scheme = self.scheme
        grid = self.open_grid(p)
        dtype = run_dtype()
        fields = self.fields(p, grid)
        sigma = self.resistivity(p, fields)
        operator = self.operator_of(p, grid, sigma)
        stopping = Stopping(tolerance=as_run_array(p["tolerance"]), at_floor=bool(c["at_floor"]), cap=int(c["cap"]),
                            check_every=int(c["check_every"]),
                            window=stagnation_window(grid.cells, int(c["check_every"])))
        preconditioner = preconditioner_named(c["preconditioner"])
        steps = jnp.asarray(state["steps"])
        tap = self.report_row(p, steps)
        magnitude = self.magnitude_of(p, grid, sigma)
        rho = as_run_array(p["density"])
        laws = self.laws()
        rows = [law.row for law in laws]
        reference, others = self.reference(), self.others()
        parameters = as_run_array(p["opening_parameters"])
        law_state = jnp.asarray(state["opening_state"])
        predicted = jnp.asarray(state["opening_pressure"])
        time = steps.astype(dtype) * dt
        instant = time + scheme.drive_instants[0] * dt
        held_units = [(state["unit_x"][k], state["unit_y"][k], state["unit_z"][k]) for k in range(len(others))]
        solved_ports: dict[str, OpenPorts] = {}

        def relative(pressures: Array) -> Velocity:
            return grid.port_force(pressures - pressures[reference], rho)

        def solve(shift: Any, rhs: Velocity, guess: Velocity | None) -> tuple[Velocity, Array]:
            def one(b: Velocity, start: Velocity | None) -> tuple[Velocity, Array]:
                return projected_solve(grid, operator(shift), self.diagonal_of(p, grid, sigma, shift), b,
                                       start if c["scaling_start"] else None, stopping, tap, preconditioner,
                                       magnitude(shift))

            base, report = one(_add(rhs, relative(predicted)), guess)
            units: list[Velocity] = []
            reports = [report]
            for k, j in enumerate(others):
                unit, unit_report = self.unit_field(one, grid.port_force(jnp.zeros(2, dtype=dtype).at[j].set(1.0), rho),
                                                    held_units[k])
                units.append(unit)
                reports.append(unit_report)
            response = jnp.stack([grid.opening_fluxes(unit) for unit in units], axis=1) if units else \
                jnp.zeros((2, 0), dtype=dtype)
            rowed = [law.affine(parameters[i], law_state[i], instant, dt) for i, law in enumerate(laws)]
            pressures, fluxes = port_system(rows, jnp.stack([r[0] for r in rowed]), jnp.stack([r[1] for r in rowed]),
                                            grid.opening_fluxes(base), response, predicted, reference)
            moved = (pressures - pressures[reference]) - (predicted - predicted[reference])
            solved = base
            for k, j in enumerate(others):
                solved = _add(solved, units[k], moved[j])
            solved_ports["ports"] = OpenPorts(pressures, fluxes, base, tuple(units), response, relative(pressures))
            return solved, combined(reports)

        context = StepContext(grid, operator, self.forcing(fields, boundary_inputs, scheme, dtype),
                              bool(c["convection"]), solve, self.kinematic(p), sigma)
        velocity = _velocity(state)
        speed = jnp.max(jnp.stack([jnp.max(jnp.abs(component)) for component in velocity]))
        cfl = speed * dt / grid.spacing
        checked = bool(c["convection"]) and scheme.cfl_line is not None
        stopped_before = jnp.asarray(state["stopped_at"]) >= 0
        past = ~(cfl <= scheme.cfl_limit(p["tube_axis"])) if checked else jnp.asarray(False)
        ended_state, stage = scheme.step(context, {"velocity": _velocity(state)}, time, dt)
        ports = solved_ports["ports"]
        ended_velocity: Velocity = ended_state["velocity"]
        flux = grid.opening_fluxes(ended_velocity)
        law_ended = jnp.stack([law.advanced(parameters[i], law_state[i], flux[i], ports.pressures[i], dt)
                               for i, law in enumerate(laws)])
        report = stage.report
        flagged = (report[2] + report[3] + report[4] > 0).astype(dtype)
        axis = as_run_array(p["tube_axis"])
        axis = axis / jnp.sqrt(jnp.sum(axis * axis))
        lumen, solid, lumen_scale = self.fluxes(grid, fields, axis, ended_velocity)
        ratio = self.ratio(solid, lumen, lumen_scale)
        reynolds = 2.0 * jnp.abs(lumen) / (jnp.pi * as_run_array(p["tube_radius"]) * self.kinematic(p))
        stokes = not bool(c["convection"])
        solved_velocity: Velocity = ended_state["velocity"]
        ended: dict[str, Array] = dict(
            velocity_x=solved_velocity[0], velocity_y=solved_velocity[1], velocity_z=solved_velocity[2],
            steps=steps + 1, solve_report=report,
            solve_report_sum=jnp.asarray(state["solve_report_sum"]) + report,
            solve_flagged=flagged, cfl=cfl.astype(dtype),
            stopped_at=jnp.where(stopped_before, jnp.asarray(state["stopped_at"]),
                                 jnp.where(past, steps + 1, -1)).astype(jnp.int32),
            stopped_cfl=jnp.where(stopped_before, jnp.asarray(state["stopped_cfl"]),
                                  jnp.where(past, cfl, 0.0)).astype(dtype),
            lumen_flux=lumen, bypass_ratio=ratio, bypass_flagged=(jnp.abs(ratio) > BYPASS_LINE).astype(dtype),
            reynolds=reynolds.astype(dtype),
            reynolds_flagged=((reynolds > STOKES_REYNOLDS) & stokes).astype(dtype),
            negative_resistivity=negative_resistivity(fields).astype(dtype),
            opening_pressure=ports.pressures.astype(dtype), opening_flux=flux.astype(dtype),
            opening_state=law_ended.astype(dtype),
            **{f"unit_{x}": (jnp.stack([unit[k] for unit in ports.units]) if ports.units else
                             jnp.asarray(state[f"unit_{x}"])) for k, x in enumerate("xyz")})
        held = node_fields(ended)
        energy: dict[str, Array] = {}
        if self.metered:
            energy = self.open_energy_terms(p, grid, velocity, stage, sigma, fields, dt, ports, flux, law_state,
                                            law_ended, parameters)
            stall = self.stall_ratio(energy)
            held = {**held, "stall_ratio": stall, "stall_flagged": (stall > STALL_LINE).astype(dtype)}
        return held, StepParts(stage, fields, energy)

    def unit_field(self, solve: Any, force: Velocity, start: Velocity) -> tuple[Velocity, Array]:
        """The unit field a unit pressure at one opening drives over the step (``force`` its force), solved from
        ``start`` (the last step's) by the step's own ``solve``, and its report."""
        return solve(force, start)

    # -- the step identity ------------------------------------------------------------------------------------------
    def open_energy_terms(self, p: Mapping[str, Any], grid: OpenMACGrid, before: Velocity, stage: StageParts,
                          sigma: Velocity, fields: WallFields, dt: float, ports: OpenPorts, flux: Array,
                          law_state: Array, law_ended: Array, parameters: Array) -> dict[str, Array]:
        """Each term of the step's identity (J), the openings' laws' work and own books, the fluid's momentum the
        drive is paired with, and the magnitude of the arithmetic forming them (``scale``)."""
        fluid = (1.0 - fields.chi[0], 1.0 - fields.chi[1], 1.0 - fields.chi[2])
        rho, nu = as_run_array(p["density"]), self.kinematic(p)
        solved = stage.solved
        inner, size = grid.inner, grid.inner_magnitude
        laplacian = grid.laplacian(solved)
        minus_laplacian = (-laplacian[0], -laplacian[1], -laplacian[2])
        penalised = (sigma[0] * solved[0], sigma[1] * solved[1], sigma[2] * solved[2])
        forced = _add(stage.rhs, ports.force)
        applied = self.operator_of(p, grid, sigma)(stage.shift)(solved)
        remainder = _add(forced, applied, -1.0)
        gradient = grid.gradient(grid.pressure_of(remainder))
        projected = _add(remainder, gradient, -1.0)
        change = _add(solved, before, -1.0)
        stores = self.scheme.stores_inertia
        zero = jnp.zeros((), dtype=solved[0].dtype)
        laws = self.laws()
        own = [law.books(parameters[i], law_state[i], law_ended[i], flux[i], ports.pressures[i], dt)
               for i, law in enumerate(laws)]
        terms = {
            **self.fluid_momentum(grid, fluid, solved, rho),
            "viscous": dt * rho * nu * inner(minus_laplacian, solved),
            "penalty": dt * rho * inner(penalised, solved),
            "backward_euler": 0.5 * rho * inner(change, change) if stores else zero,
            "convection": -dt * rho * inner(stage.convection, change),
            "krylov": dt * rho * inner(projected, solved),
            "projection": dt * rho * inner(gradient, solved),
            KINETIC_FLUX: dt * rho * inner(stage.convection, before),
            "ports": -dt * ports.pressures * flux,
            "law_books": jnp.stack([jnp.stack([b.storage_start, b.storage_end, b.loss, b.numerical]) for b in own]),
        }
        # The openings' work is judged on its own magnitude, the pressures against the faces' flux in magnitude.
        port_scale = dt * jnp.sum(jnp.abs(ports.pressures) * grid.opening_fluxes_magnitude(solved))
        law_scale = sum((jnp.abs(b.storage_start) + jnp.abs(b.storage_end) + jnp.abs(b.loss) + jnp.abs(b.numerical)
                         for b in own), start=zero)
        scale = rho * (dt * (size(forced, solved) + size(applied, solved) + size(stage.convection, solved)
                             + size(stage.convection, before) + nu * size(minus_laplacian, solved)
                             + size(penalised, solved) + size(gradient, solved))
                       + (size(before, before) + size(solved, solved) + size(before, solved) if stores else 0.0))
        return {**terms, "scale": scale + port_scale + law_scale}

    @override
    def hosted_laws(self) -> tuple[tuple[str, str, str], ...]:
        """The viscosity's and the wall's (as the periodic node's), each opening's law under its own id and class, and
        the kinetic energy convection carries through the openings, one line under ``<node>.openings``, a crossing
        of the boundary (no class)."""
        c = self.constants()
        openings = tuple((opening_role(i), str(law_id), port_law_named(law).energy_class)
                         for i, (law_id, law) in enumerate(zip(c["opening_law_ids"], c["opening_laws"], strict=True)))
        return (*super().hosted_laws(), *openings, (KINETIC_FLUX, f"{self.name}.openings", books.UNMETERED))

    @override
    def storing_laws(self) -> tuple[str, ...]:
        c = self.constants()
        return tuple(str(law_id) for law_id, law in zip(c["opening_law_ids"], self.laws(), strict=True) if law.stores)

    @override
    def law_storage(self, p: Mapping[str, Any], state: Mapping[str, Array]) -> Array | None:
        """Each held law's own storage at ``state``, a row per law: the Windkessels' capacitors; ``None`` where no
        opening's law stores energy."""
        laws = self.laws()
        if not any(law.stores for law in laws) or not self.metered:
            return None
        held = books.deliveries(self.constants())
        rows = books.laws(held)
        storage = jnp.zeros(len(rows), dtype=run_dtype())
        parameters = as_run_array(p["opening_parameters"])
        law_state = jnp.asarray(state["opening_state"])
        for delivery in held:
            if delivery.kind == books.HOSTED and delivery.field.startswith(OPENING):
                index = int(delivery.field.removeprefix(OPENING))
                own = laws[index].books(parameters[index], law_state[index], law_state[index],
                                        jnp.zeros((), dtype=run_dtype()), jnp.zeros((), dtype=run_dtype()), 0.0)
                storage = storage.at[rows.index(delivery.law)].set(own.storage_start)
        return storage

    @override
    def identity_terms(self, p: Mapping[str, Any], held: Sequence[books.LedgerDelivery], state: Mapping[str, Array],
                       boundary_inputs: Mapping[str, Array], ended: Mapping[str, Array], parts: StepParts,
                       dt: float) -> books.StepAccount:
        """The periodic node's identity with the openings: each law's work ``-dt p q`` under its id, the kinetic
        energy's flux under ``<node>.openings``, the scheme's terms and the projection's; and each storing law's own
        books (``StepAccount.stored``)."""
        energy = parts.energy
        dtype = jnp.asarray(energy["krylov"]).dtype
        rows = books.laws(held)
        check_open_deliveries(held)
        work, unattributed, drive_scale = self.drive_work(held, boundary_inputs, rows, dt, energy["momentum"],
                                                          energy["momentum_size"])
        hosted_work: dict[str, Array] = {"viscous": -energy["viscous"], "penalty": -energy["penalty"],
                                         KINETIC_FLUX: energy[KINETIC_FLUX],
                                         **{opening_role(i): energy["ports"][i] for i in range(2)}}
        stored = jnp.zeros((len(rows), len(books.LAW_BOOKS)), dtype=dtype)
        named: set[str] = set()
        for delivery in held:
            if delivery.kind == books.HOSTED:
                row = rows.index(delivery.law)
                work = work.at[row].add(hosted_work[delivery.field])
                named.add(delivery.field)
                if delivery.field.startswith(OPENING):
                    stored = stored.at[row].set(energy["law_books"][int(delivery.field.removeprefix(OPENING))])
        unattributed = unattributed + sum((value for field, value in hosted_work.items() if field not in named),
                                          start=jnp.zeros((), dtype=dtype))
        terms = jnp.stack([energy[name] for name in OPEN_TERMS])
        storing = any(law.stores for law in self.laws())
        return books.StepAccount(work=work, hosted=jnp.zeros((len(rows), 2), dtype=dtype), unattributed=unattributed,
                                 terms=terms, storage_start=self.storage(p, state), storage_end=self.storage(p, ended),
                                 scale=energy["scale"] + drive_scale, operators={},
                                 stored=stored if storing else None)


def combined(reports: Sequence[Array]) -> Array:
    """One report for a step's solves: the iterations and counts summed, the residual the worst."""
    total = sum(reports[1:], start=reports[0])
    worst = jnp.max(jnp.stack([report[1] for report in reports]))
    return total.at[1].set(worst)


def check_open_deliveries(held: Sequence[books.LedgerDelivery]) -> None:
    """:func:`check_deliveries` with the open node's own hosted rows: each opening's law and the kinetic flux's line."""
    plain = [d for d in held if not (d.kind == books.HOSTED and (d.field == KINETIC_FLUX or d.field.startswith(OPENING)))]
    check_deliveries(plain)


__all__ = ["FLOAT32_PORTS", "KINETIC_FLUX", "OPENING", "OpenFlowFields", "OpenFlowNode", "OpenPorts", "check_open_deliveries",
           "combined", "opening_role"]
