"""RK3 on a box open along its axis: the open grid's vessel (``mime.nodes.flow.fvm._open``), its openings held by laws
in the step (``mime.nodes.flow.fvm._open_node``'s), stepped by S1's explicit RK3 (``mime.nodes.flow.fvm._rk3``) with the
openings coupled implicitly at each of its stages (F0's B6/B7, decisions 8, 11 and 14; R5-C-10).

**A stage.** RK3's stage takes, beside the incremental field pressure's gradient, each opening's force at the relative
pressures the last stage solved (the prediction), with the pressure's weight ``alpha_k dt``; its projection is the
field's affine response at the openings (:class:`OpeningStages`): the stage's solution ``u = u_0 + sum_j m_j U_j`` and
its pressure's increment ``phi = phi_0 + sum_j m_j phi_j``, ``U_j`` the unit field of a unit pressure at the ``j``-th
opening that is not the gauge's reference, ``U_j = P[alpha_k dt F_j]`` (``F_j`` its force on its patch's half cells,
``P`` the projector: a fast solve each), ``m_j`` the move of its relative pressure from the prediction, solved with the
laws' rows (``port_system``, the implicit node's). **The moves are pressure increments and are taken as the stage takes
its own**: the prediction's forces enter before the penalty's pointwise solve, as the last pressure's gradient does, and
the moves after it, as the increment ``phi``'s gradient does, so that the openings' pressures and the field's are one
pressure to the scheme (a unit field divided by the penalty, the plane's solid faces damped and the solid behind them
not, kicked the transverse velocity to 0.13 of the mean speed in the fluid in the first steps at a proximal resistance
of 100 times the vessel's, against rounding taken this way [measured while writing, the spike's question 7 point]). **Each law is held at the stage's own instant**, its end ``t_n + c_{k+1} dt`` (``_rk3.STAGE_ENDS``,
where the stage's solution is), its rows backward Euler over ``alpha_k dt`` (a Windkessel's capacitor stepped from the
stage's start to its end), and advanced by the stage's solved flux and pressure: the three stages advance it by ``dt``
(``sum alpha_k = 1``), first order in time (R-8).

**The exchange is workless by construction.** The flow books each law's work ``-sum_k alpha_k dt p_k . q_k`` (its
pressures and its openings' face fluxes at each stage's solution), and the law its own books over the stages, ``sum_k
alpha_k dt p_k . q_k = dE + loss + N``, the same products: their difference is rounding at every step (the law's own
identity, ``mime.core.ledger.law_records``). A law stepped once a step leaves a first-order residual there.

**The start** (decision 14). A step whose pressure is unmade (the run's start, a state given from outside) first makes
it consistent with the velocity and the openings: the first stage's port system, solved on the pressure consistent with
the velocity alone, gives the openings' relative pressures, and the field pressure consistent with the velocity and
those openings' forces is the step's (two fast solves and one port system, in the program). The field sees each
opening's pressure relative to the gauge's reference, so a pressure level many times the vessel's own drop (a
downstream resistance many times the vessel's) never reaches the field's start.

**The backflow term** (decision 12) is explicit from each stage's start velocity, beside the drive, and booked by RK3's
quadrature at the stages' starts, where each of its products is at most zero: its sign check (``backflow_flagged``)
cannot fire under RK3, and route 1's check judges the step.

**The ledger: route 1** (B9, D3). The step books each law's work under its id, the kinetic energy convection carries
through the openings, ``sum_k b_k dt rho <C(u_k), u_k>`` (on the open box the convective power is all of it, so the
scheme's own convective term is zero), under ``<node>.openings``, the backflow term's under ``<node>.backflow``, the
viscosity's and the wall's under their names, and a drive as the periodic RK3 node books it, its attribution claimed;
the split penalty has no formula term, so no identity is claimed and each step is judged to have dissipated, the laws'
storage counted with the flow's (``mime.core.ledger.route_one_recorded``)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, ClassVar, NamedTuple, cast, override

import jax
from jax import Array
from maddening.core.compliance.metadata import NodeMeta, Reference
from maddening.core.params import ParamSpec

from mime.core import NodeFields, Symbol, SymbolKind, as_run_array, jnp, node_fields, run_dtype
from mime.core import ledger as books
from mime.core.ledger import TermOrder
from mime.nodes.flow.fvm import _rk3
from mime.nodes.flow.fvm._band import WallFields, negative_resistivity
from mime.nodes.flow.fvm._explicit import ExplicitFields, cfl_check, delivered_work
from mime.nodes.flow.fvm._grid import Velocity
from mime.nodes.flow.fvm._node import BYPASS_LINE, STOKES_REYNOLDS, WALL_COUNTS, WallCounts
from mime.nodes.flow.fvm._open import OpenMACGrid
from mime.nodes.flow.fvm._open_node import BACKFLOW, KINETIC_FLUX, OPENING, REVERSAL_LINE, OpeningFields, \
    OpenFlowNode, check_open_deliveries, opening_role
from mime.nodes.flow.fvm._open_node import EQUATIONS as OPEN_EQUATIONS
from mime.nodes.flow.fvm._ports import LawBooks, PortSolution
from mime.nodes.flow.fvm._schemes import StageKind, StepContext
from mime.registry import Problem

#: ``jax.lax``, whose hints name a device type from jaxlib's compiled core, which ships no stubs.
_LAX: Any = jax.lax
_STEPPED = "mime.nodes.flow.fvm._open_explicit.OpenExplicitFlowNode.stepped"
#: RK3's numerical terms (its convective term, zero on the open box: its power is all booked as the openings' flux).
SCHEME_TERMS = ("convection",)
_START = "mime.nodes.flow.fvm._open_explicit.OpeningStages.ready"


class OpenExplicitFlowState(ExplicitFields, OpeningFields, closed=True):
    """The open flow stepped by RK3, without the energy ledger: the explicit flow's fields and its openings'."""


class OpenExplicitTreeFlowState(ExplicitFields, OpeningFields, WallCounts, closed=True):
    """The open flow stepped by RK3 walled by a tube tree (a built region), without the energy ledger."""


class StageRecord(NamedTuple):
    """What one stage solved at the openings: the port system's answer, each opening's face flux at the stage's
    solution and its magnitude, the superposition's scale (``OpenFlowNode.superposed_scale``'s), the laws' blocks
    placed over the openings, each law's own books over the stage, and the stage's step ``alpha_k dt``."""

    solution: PortSolution
    flux: Array
    flux_size: Array
    superposed: Array
    blocks: Array
    books: tuple[LawBooks, ...]
    stage_dt: Any


def _add(a: Velocity, b: Velocity, scale: Any = 1.0) -> Velocity:
    return (a[0] + scale * b[0], a[1] + scale * b[1], a[2] + scale * b[2])


def _velocity(state: Mapping[str, Array]) -> Velocity:
    return (state["velocity_x"], state["velocity_y"], state["velocity_z"])


def projection(grid: OpenMACGrid, held: Velocity, stage_dt: Any) -> tuple[Velocity, Array]:
    """A stage's projection of ``held`` (RK3's, ``_rk3.step``'s): the projected field and the pressure's increment."""
    phi = grid.poisson(grid.divergence(held) / stage_dt)
    return _add(held, grid.gradient(phi), -stage_dt), phi


def _no_monolithic_solve(shift: Any, rhs: Velocity, guess: Velocity | None) -> tuple[Velocity, Array]:
    raise TypeError("a scheme of the explicit stage kind called the monolithic projected CG's solve")


class OpeningStages:
    """The openings of one RK3 step on the open box (the module states them): its start, the backflow term, the
    openings' forces at the prediction and each stage's projection with their rows solved, each law advanced at its
    stage (:meth:`advance_laws`), the stages' records kept for the step's books (:attr:`records`)."""

    def __init__(self, node: OpenExplicitFlowNode, p: Mapping[str, Any], grid: OpenMACGrid, fields: WallFields,
                 sigma: Velocity, law_state: Array, predicted: Array, time: Any, dt: Any, convection: bool) -> None:
        self.node, self.grid, self.fields, self.sigma = node, grid, fields, sigma
        self.rho, self.nu = as_run_array(p["density"]), node.kinematic(p)
        self.parameters = as_run_array(p["opening_parameters"])
        self.laws, self.bound = node.laws(), node.law_openings()
        self.rows, self.reference, self.others = node.rows(), node.reference(), node.others()
        self.count = node.opening_count()
        self.law_state, self.predicted = law_state, predicted
        self.time, self.dt, self.convection = time, dt, convection
        dtype = predicted.dtype
        self.unit_forces = [grid.port_force(jnp.zeros(self.count, dtype=dtype).at[j].set(1.0), self.rho)
                            for j in self.others]
        self.records: list[StageRecord] = []

    def ports(self) -> _rk3.StagePorts:
        return _rk3.StagePorts(self.ready, self.explicit, self.held_force, self.project)

    def explicit(self, k: int, velocity: Velocity) -> Velocity:
        """The backflow term at the stage's start velocity (decision 12), zero where none is placed."""
        del k
        return self.node.backflow_force(self.grid, self.fields, velocity)

    def held_force(self, k: int) -> Velocity:
        """The openings' forces at the prediction, the relative pressures the last stage solved."""
        del k
        return self.grid.port_force(self.predicted, self.rho)

    def units(self, stage_dt: Any) -> list[tuple[Velocity, Array]]:
        """Each unit field over the stage and its pressure's increment: the projection of the unit pressure's force
        over the stage, ``P[alpha_k dt F_j]``, taken after the penalty's solve as the stage's own pressure increment
        is (the module)."""
        found: list[tuple[Velocity, Array]] = []
        for force in self.unit_forces:
            found.append(projection(self.grid, cast(Velocity, tuple(stage_dt * force[c] for c in range(3))), stage_dt))
        return found

    def rows_at(self, k: int, stage_dt: Any) -> tuple[Array, Array]:
        """The laws' rows over stage ``k``: backward Euler over ``alpha_k dt`` at the stage's end."""
        return self.node.law_rows(self.parameters, self.law_state, self.time + _rk3.STAGE_ENDS[k] * self.dt, stage_dt)

    def solve(self, k: int, base: Velocity, units: Sequence[tuple[Velocity, Array]], predicted: Array,
              stage_dt: Any) -> tuple[PortSolution, Array]:
        """The stage's port system on its base and unit fields, and the laws' blocks it was solved with."""
        dtype = base[0].dtype
        response = jnp.stack([self.grid.opening_fluxes(unit[0]) for unit in units], axis=1) if units else \
            jnp.zeros((self.count, 0), dtype=dtype)
        a, blocks = self.rows_at(k, stage_dt)
        return self.node.solve_ports(self.rows, a, blocks, self.grid.opening_fluxes(base), response, predicted,
                                     self.reference), blocks

    def project(self, k: int, held: Velocity, stage_dt: Any) -> tuple[Velocity, Array, Velocity]:
        """Stage ``k``'s projection with its openings' rows solved (the module): the solution and the pressure's
        increment, each the base's plus the unit fields' at the solved moves, and the field the penalty was solved at
        (the stage's own: the moves are increments, taken after it)."""
        grid = self.grid
        base, phi = projection(grid, held, stage_dt)
        units = self.units(stage_dt)
        solution, blocks = self.solve(k, base, units, self.predicted, stage_dt)
        velocity = base
        superposed = grid.opening_fluxes_magnitude(base)
        for (unit, unit_phi), j in zip(units, self.others, strict=True):
            moved = solution.moved[j]
            velocity = _add(velocity, unit, moved)
            phi = phi + moved * unit_phi
            superposed = superposed + jnp.abs(moved) * grid.opening_fluxes_magnitude(unit)
        self.advance_laws(grid.opening_fluxes(velocity), grid.opening_fluxes_magnitude(velocity), solution, blocks,
                          superposed, stage_dt)
        self.predicted = solution.relative
        return velocity, phi, held

    def advance_laws(self, flux: Array, flux_size: Array, solution: PortSolution, blocks: Array, superposed: Array,
                     stage_dt: Any) -> None:
        """Each law advanced over the stage by its solved flux and pressure, its books over the stage recorded."""
        node, pressures = self.node, solution.pressures
        ended = jnp.stack([law.advanced(self.parameters[i], self.law_state[i], node.per_law(law, held, flux),
                                        node.per_law(law, held, pressures), stage_dt)
                           for i, (law, held) in enumerate(zip(self.laws, self.bound, strict=True))])
        own = tuple(law.books(self.parameters[i], self.law_state[i], ended[i], node.per_law(law, held, flux),
                              node.per_law(law, held, pressures), stage_dt)
                    for i, (law, held) in enumerate(zip(self.laws, self.bound, strict=True)))
        self.records.append(StageRecord(solution, flux, flux_size, superposed, blocks, own, stage_dt))
        self.law_state = ended

    def ready(self, velocity: Velocity, pressure: Array, ready: Array, force: Velocity) -> Array:
        """The pressure the step starts from (decision 14; the module's start): ``pressure`` where ``ready`` marks it
        made, else consistent with the velocity and the openings, the prediction made with it."""

        def made() -> tuple[Array, Array]:
            driven = _add(force, self.explicit(0, velocity))
            alone = _rk3.consistent_pressure(self.grid, velocity, self.nu, self.sigma, driven, self.convection)
            relative = self.start_pressures(velocity, alone, driven)
            opened = _add(driven, self.grid.port_force(relative, self.rho))
            return _rk3.consistent_pressure(self.grid, velocity, self.nu, self.sigma, opened, self.convection), relative

        held, predicted = _LAX.cond(jnp.asarray(ready) > 0.5, lambda: (pressure, self.predicted), made)
        self.predicted = predicted
        return cast(Array, held)

    def start_pressures(self, velocity: Velocity, pressure: Array, driven: Velocity) -> Array:
        """The openings' relative pressures at the start: the first stage's port system on ``pressure`` (the one
        consistent with the velocity alone), from no prediction."""
        grid, stage_dt = self.grid, _rk3.ALPHA[0] * self.dt
        rhs = _rk3.explicit_rhs(grid, velocity, self.nu, driven, self.convection)
        r = _add(_add(velocity, rhs, _rk3.GAMMA[0] * self.dt), grid.gradient(pressure), -stage_dt)
        held = cast(Velocity, tuple(r[c] / (1.0 + stage_dt * self.sigma[c]) for c in range(3)))
        base, _ = projection(grid, held, stage_dt)
        solution, _ = self.solve(0, base, self.units(stage_dt), jnp.zeros_like(self.predicted), stage_dt)
        return solution.relative


class OpenExplicitParts(NamedTuple):
    """What the route-1 account reads of a step: RK3's record, the fields, and the openings' books."""

    stage: _rk3.ExplicitStage
    fields: WallFields
    energy: Mapping[str, Array]


class OpenExplicitFlowNode(OpenFlowNode):
    """The flow field of a medium in a vessel cut at the box's two z planes, its openings held by laws the step holds,
    stepped by S1's explicit RK3 with the openings coupled at each stage (the module states the step)."""

    STAGE_KIND = StageKind.EXPLICIT_PROJECTED
    SCHEME_FIELDS = ("velocity", "pressure", "pressure_ready")
    LEDGER_TERMS = SCHEME_TERMS
    GRID_TERMS = ()
    GRID_ORDERS: ClassVar[Mapping[str, TermOrder]] = MappingProxyType({})
    #: None: no flag's line is measured on RK3's runs (the periodic RK3 node's, ``_flags`` says so).
    LEDGER_FLAGS = ()
    rechecks = False

    meta = NodeMeta(
        algorithm_id="MIME-NODE-132",
        description="A medium's flow in a vessel open at the box's two z planes stepped by S1's explicit RK3: the "
                    "Cartesian MAC grid open along z, each opening (a patch of a plane) held by a law the step holds at "
                    "each of RK3's stages, coupled implicitly through the stage's affine response.",
        governing_equations=OPEN_EQUATIONS,
        assumptions=(
            "Inside, the periodic RK3 node's (ExplicitFlowNode's) stage: convection and viscosity explicit, the "
            "wall's penalty implicit and pointwise in each stage, the incremental projection each stage",
            "The grid is open along z as the implicit open node's is: the planes' faces evolved half cells at half "
            "weight, the pressure's field part zero on the planes, whose pressures the openings' forces carry",
            "Each stage takes the openings' predicted forces with the pressure's weight alpha_k dt before the penalty's "
            "pointwise solve, and their moves after it, as its own pressure increment: the unit fields are the stage's "
            "projections of a unit pressure's force (fast solves)",
            "Each law's rows are backward Euler over its stage's alpha_k dt at the stage's end instant, the law "
            "advanced by the stage's solved flux and pressure",
            "The run starts from a pressure consistent with its velocity and its openings, made in the program",
            "The time of a step is its count times the timestep, never a sum of timesteps"),
        limitations=(
            "Openings on the box's two z planes only: the region builder bends each cut onto one of them",
            "The laws are first order in time under RK3 (backward Euler per stage); with a wall the flux is first "
            "order in time and the pressure first order (the split penalty)",
            "Stable only within its CFL line and its viscous number; the node checks the CFL number each step",
            "No step identity is claimed: each step is judged to have dissipated (the ledger's first route)",
            "The backflow term is explicit from each stage's start, normal only; its booked work cannot be positive, "
            "so its sign check never fires under RK3",
        ),
        references=(Reference("HarlowWelch1965"), Reference("SpalartMoserRogers1991"), Reference("Westerhof2009")),
    )
    symbols = (
        Symbol(r"p", SymbolKind.STATE, "opening_pressure", "Pa", _STEPPED),
        Symbol(r"q", SymbolKind.STATE, "opening_flux", "m^3/s", _STEPPED),
        Symbol(r"\alpha_{k}", SymbolKind.CONSTANT, "mime.nodes.flow.fvm._rk3.ALPHA", "1", _STEPPED),
    )

    @override
    def param_specs(self) -> dict[str, ParamSpec]:
        """The open node's, without the monolithic solve's tolerance and report (this stage kind has neither)."""
        specs = super().param_specs()
        for key in ("tolerance", "solve_report"):
            specs.pop(key, None)
        return specs

    @override
    def initial_state(self) -> NodeFields:
        c = self.constants()
        grid = self.open_grid(c)
        dtype = run_dtype()
        zero = grid.zeros(dtype)
        given = tuple(jnp.asarray(c[key], dtype=dtype) if key in c else zero[k]
                      for k, key in enumerate(("initial_velocity_x", "initial_velocity_y", "initial_velocity_z")))
        start = self.scheme.start(grid, (given[0], given[1], given[2]))
        velocity: Velocity = start["velocity"]
        openings = jnp.zeros(self.opening_count(), dtype=dtype)
        z = as_run_array(0.0)
        held: dict[str, Array] = dict(
            velocity_x=velocity[0], velocity_y=velocity[1], velocity_z=velocity[2], pressure=start["pressure"],
            pressure_ready=start["pressure_ready"], steps=jnp.asarray(0, dtype=jnp.int32), solve_flagged=z, cfl=z,
            stopped_at=jnp.asarray(-1, dtype=jnp.int32), stopped_cfl=z, lumen_flux=z, bypass_ratio=z,
            bypass_flagged=z, reynolds=z, reynolds_flagged=z, negative_resistivity=z, opening_pressure=openings,
            opening_pressure_relative=openings, opening_flux=openings,
            opening_state=jnp.asarray(c["initial_opening_state"], dtype=dtype), opening_reversed=openings,
            opening_reversed_flagged=z, backflow_work=z, backflow_flagged=z, opening_exchange=openings,
            opening_exchange_flagged=z)
        if self.walled_by_tree:
            held.update(self.wall_report(jnp.zeros(len(WALL_COUNTS), dtype=dtype)))
        if not self.metered:
            return node_fields(held)
        return books.route_one_records(books.with_records(self, held))

    @override
    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        if self.metered:
            ended = books.metered_route_one_step(self, self.own_inputs(), p, state, boundary_inputs, dt, self.stepped)
        else:
            ended = self.stepped(p, state, boundary_inputs, dt)[0]
        return self.held_past_the_line(state, ended)

    def stages(self, p: Mapping[str, Any], grid: OpenMACGrid, fields: WallFields, sigma: Velocity,
               state: Mapping[str, Array], time: Any, dt: Any) -> OpeningStages:
        """The openings of the step from ``state`` (the stages' machinery, :class:`OpeningStages`)."""
        return OpeningStages(self, p, grid, fields, sigma, jnp.asarray(state["opening_state"]),
                             jnp.asarray(state["opening_pressure_relative"]), time, dt,
                             bool(self.constants()["convection"]))

    @override
    def stepped(self, p: Mapping[str, Any], state: Mapping[str, Array], boundary_inputs: Mapping[str, Array],
                dt: float) -> tuple[dict[str, Array], Any]:
        c = self.constants()
        scheme = self.scheme
        grid = self.open_grid(p)
        dtype = run_dtype()
        fields = self.fields(p, grid)
        sigma = self.resistivity(p, fields)
        convection = bool(c["convection"])
        steps = jnp.asarray(state["steps"])
        time = steps.astype(dtype) * dt
        velocity = _velocity(state)
        stages = self.stages(p, grid, fields, sigma, state, time, dt)
        weights = grid.weights(dtype)
        fluid = cast(Velocity, tuple(weights[k] * (1.0 - fields.chi[k]) for k in range(3))) if self.metered else None
        context = StepContext(grid, self.operator_of(p, grid, sigma), self.forcing(fields, boundary_inputs, scheme, dtype),
                              convection, _no_monolithic_solve, self.kinematic(p), sigma, fluid, stages.ports())
        cfl, stopped_at, stopped_cfl = cfl_check(velocity, dt, grid.spacing, scheme.cfl_limit(self.cfl_axis(p)),
                                                 convection, jnp.asarray(state["stopped_at"]),
                                                 jnp.asarray(state["stopped_cfl"]), steps)
        ended_state, stage = scheme.step(context, {"velocity": velocity, "pressure": state["pressure"],
                                                   "pressure_ready": state["pressure_ready"]}, time, dt)
        records = stages.records
        last = records[-1]
        solved: Velocity = ended_state["velocity"]
        flux = last.flux
        axis = self.vessel_axis(p)
        axis = axis / jnp.sqrt(jnp.sum(axis * axis))
        lumen, solid, lumen_scale = self.fluxes(grid, fields, axis, solved)
        ratio = self.ratio(solid, lumen, lumen_scale)
        reynolds = 2.0 * jnp.abs(lumen) / (jnp.pi * self.vessel_radius(p) * self.kinematic(p))
        # The outlet watch (decision 13): an observer, read by nothing above.
        inward = grid.opening_reversed_fluxes(solved, 1.0 - fields.chi[2])
        outward = flux > 0.0
        reversed_share = jnp.where(outward, inward / jnp.where(outward, flux, 1.0), 0.0)
        watched = jnp.max(reversed_share) > float(c.get("reversal_line", REVERSAL_LINE))
        backflow_work = as_run_array(p["density"]) * (stage.explicit if stage.explicit is not None else 0.0)
        exchange = self.opening_exchange(p, grid, fields, solved, flux)
        exchanged = jnp.max(jnp.abs(exchange)) > self.exchange_line(p, grid)
        ended: dict[str, Array] = dict(
            velocity_x=solved[0], velocity_y=solved[1], velocity_z=solved[2], pressure=ended_state["pressure"],
            pressure_ready=ended_state["pressure_ready"], steps=steps + 1, solve_flagged=jnp.zeros((), dtype=dtype),
            cfl=cfl, stopped_at=stopped_at, stopped_cfl=stopped_cfl, lumen_flux=lumen, bypass_ratio=ratio,
            bypass_flagged=(jnp.abs(ratio) > BYPASS_LINE).astype(dtype), reynolds=reynolds.astype(dtype),
            reynolds_flagged=((reynolds > STOKES_REYNOLDS) & (not convection)).astype(dtype),
            negative_resistivity=negative_resistivity(fields).astype(dtype),
            opening_pressure=last.solution.pressures.astype(dtype),
            opening_pressure_relative=last.solution.relative.astype(dtype), opening_flux=flux.astype(dtype),
            opening_state=stages.law_state.astype(dtype), opening_reversed=reversed_share.astype(dtype),
            opening_reversed_flagged=watched.astype(dtype), backflow_work=backflow_work.astype(dtype),
            # Each product of RK3's quadrature of the term is at most zero (the module): the sign check reads it.
            backflow_flagged=(backflow_work > 0.0).astype(dtype), opening_exchange=exchange.astype(dtype),
            opening_exchange_flagged=exchanged.astype(dtype))
        if fields.counts is not None:
            ended.update(self.wall_report(fields.counts))
        energy = self.opening_books(grid, records) if self.metered else {}
        return node_fields(ended), OpenExplicitParts(stage, fields, energy)

    def opening_books(self, grid: OpenMACGrid, records: Sequence[StageRecord]) -> dict[str, Array]:
        """The openings' part of the step's account (J): each law's work ``-sum_k alpha_k dt p_k . q_k``, each law's own
        books over the stages (``mime.core.ledger.LAW_BOOKS``' columns), and the magnitude of their arithmetic. A law's
        own scale holds the face flux's summation floor carried through its block at each stage, as the implicit open
        node's does (``OpenFlowNode.open_energy_terms``)."""
        laws, bound, coupling = self.laws(), self.law_openings(), self.coupling()
        faces = grid.opening_faces()
        dtype = records[0].flux.dtype
        zero = jnp.zeros((), dtype=dtype)
        ports: list[Array] = []
        rows: list[Array] = []
        law_scale = port_scale = zero
        for r in records:
            port_scale = port_scale + r.stage_dt * jnp.sum(jnp.abs(r.solution.pressures) * r.flux_size)
        for i, (law, held) in enumerate(zip(laws, bound, strict=True)):
            ports.append(cast(Array, sum((-r.stage_dt * r.solution.pressures[o] * r.flux[o] for r in records
                                          for o in held), start=zero)))
            own = [r.books[i] for r in records]
            start, end = own[0].storage_start, own[-1].storage_end
            loss = sum((b.loss for b in own), start=zero)
            numerical = sum((b.numerical for b in own), start=zero)
            sizes = jnp.abs(start) + jnp.abs(end) + sum((jnp.abs(b.loss) + jnp.abs(b.numerical) for b in own),
                                                        start=zero)
            law_scale = law_scale + sizes
            if not (law.stores or law.own_books):
                rows.append(jnp.zeros(len(books.LAW_BOOKS), dtype=dtype))
                continue
            floor = zero
            products = zero
            for r in records:
                pushed = [sum((r.blocks[o, j] * r.flux[j] for j in range(len(faces)) if coupling[o][j]), start=zero)
                          for o in range(len(faces))]
                floor = floor + sum((r.stage_dt * jnp.abs(pushed[o]) * (faces[o] + 2) * r.superposed[o] for o in held),
                                    start=zero)
                products = products + sum((jnp.abs(r.stage_dt * r.solution.pressures[o] * r.flux[o]) for o in held),
                                          start=zero)
            rows.append(jnp.stack([start, end, loss, numerical, products + sizes + floor / books.IDENTITY_ULPS]))
        return {"ports": jnp.stack(ports), "law_books": jnp.stack(rows), "scale": port_scale + law_scale}

    @override
    def identity_terms(self, p: Mapping[str, Any], held: Sequence[books.LedgerDelivery], state: Mapping[str, Array],
                       boundary_inputs: Mapping[str, Array], ended: Mapping[str, Array], parts: Any,
                       dt: float) -> books.StepAccount:
        """The step's account on route 1 (the module): each law's work under its id, its own books beside
        (``StepAccount.stored``), the hosted laws' under their names, the kinetic energy carried through the openings,
        a drive's by RK3's quadrature with its attribution claimed (as the periodic RK3 node's), the storage at both
        ends; the scheme's convective term zero, its power all booked as the openings' kinetic flux."""
        stage, energy = parts.stage, parts.energy
        rho = as_run_array(p["density"])
        dtype = jnp.asarray(stage.drive).dtype
        check_open_deliveries(held)
        rows = books.laws(held)
        read = self.drive_read()
        driving = [index for index, d in enumerate(held) if d.kind == books.LOAD and d.field == read]
        if len(driving) > 1:
            raise ValueError(f"{len(driving)} laws deliver the flow's {read}; a waveform is not summed, so one law "
                             f"delivers it")
        work = jnp.zeros(len(rows), dtype=dtype)
        unattributed = jnp.zeros((), dtype=dtype)
        drive = rho * stage.drive
        claimed: tuple[Array, Array] | None = None
        if driving and read not in boundary_inputs:
            delivered, size = delivered_work(self.scheme, boundary_inputs[books.law_input(read, driving[0])], stage,
                                             state, dt)
            work = work.at[rows.index(held[driving[0]].law)].add(rho * delivered)
            claimed = (drive - rho * delivered, 2.0 * rho * size)
        else:
            unattributed = unattributed + drive
        zero = jnp.zeros((), dtype=dtype)
        hosted_work: dict[str, Array] = {
            "viscous": -rho * stage.viscous, "penalty": -rho * stage.penalty, KINETIC_FLUX: rho * stage.convection,
            BACKFLOW: rho * stage.explicit if stage.explicit is not None else zero,
            **{opening_role(i): energy["ports"][i] for i in range(len(self.laws()))}}
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
                                          start=zero)
        return books.StepAccount(work=work, hosted=jnp.zeros((len(rows), 2), dtype=dtype), unattributed=unattributed,
                                 terms=jnp.zeros(len(SCHEME_TERMS), dtype=dtype),
                                 storage_start=self.storage(p, state), storage_end=self.storage(p, ended),
                                 scale=rho * stage.scale + energy["scale"], operators={},
                                 claimed=None if claimed is None else claimed[0],
                                 claimed_scale=None if claimed is None else claimed[1],
                                 stored=stored if self.keeps_books() else None)

    @override
    def run_problem(self, end: int) -> Problem | None:
        """None: the explicit stage keeps no per-step rows of derivative-solve reports."""
        del end
        return None


__all__ = ["OpenExplicitFlowNode", "OpenExplicitFlowState", "OpenExplicitParts", "OpenExplicitTreeFlowState",
           "OpeningStages", "StageRecord", "projection"]
