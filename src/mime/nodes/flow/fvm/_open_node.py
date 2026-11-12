"""The implicit flow node on a box open along its axis: the FVM's field in a vessel cut at the box's two planes, its
openings patches of those planes, each opening's law held in the step (F0's B6/B7; the record's "F2's rows, 2"; F2b's
PRs 2 and 3). A law binds one opening or several (a block over them: a shared distal resistance); a plane holds one
opening or several, split between their centres (``mime.nodes.flow.fvm._open``'s layout).

**A step** is :class:`ImplicitFlowNode`'s (IMEX Euler or its steady limit on the monolithic projected CG), on the open
grid (``mime.nodes.flow.fvm._open``), with its openings coupled implicitly through the field's affine response
(decision 11): the step's solve takes the openings' forces at the pressures the last step solved for relative to the
gauge's reference opening (decision 9; held in the state as such, never formed as a difference of absolute pressures),
then the openings' rows (``mime.nodes.flow.fvm._ports``) are solved with the unit fields' response for each relative
pressure's move, and the solution is the solve's plus each unit field times its move. **The unit
fields are made in the program** (round 5, item 13): one a non-reference opening, the response to a unit pressure there,
held in the state and solved again at every step from the last (a warm start that converges in an iteration or two
while the operator is fixed), so that every step's are those of its own parameters and a derivative passes through
them. Each law then steps its own state with its opening's solved flux and pressure.

**Its step identity** (route 2, paired at the solution ``u'``): ``dKE = W_drive + W_ports + W_KE - D_visc - D_wall -
N_BE - N_conv - N_Krylov - N_proj``: each opening's law's work ``-dt p q`` under its own name (its outward face flux
``q`` and its absolute pressure ``p``: the field's forces are the pressures relative to the reference, whose difference
is the reference's pressure times the net flux, zero to rounding); the kinetic energy convection carries through the
openings, ``W_KE = dt rho <C(u), u>``, the scheme's own convective power (B9, decision 14), as one line under
``<node>.openings`` (its split by opening, exact to the field's divergence, is the ``opening_kinetic_flux`` flag's); ``N_conv = -dt rho <C(u), u' - u>``, convection at mixed time levels; ``N_Krylov`` the solve's
projected remainder against ``u'`` and ``N_proj = dt rho <G p', u'>`` its gradient part, ``p'`` the remainder's
pressure: the half-weight faces keep the projector orthogonal, so ``N_proj`` is rounding (C13), booked apart so that its
flag reads it. A law storing energy of its own (the Windkessel) keeps its own books beside the field's (its storage at
the step's start and end, its loss, its ``N_WK``: ``mime.core.ledger.law_records``), none of them the field's."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, ClassVar, NamedTuple, override

import numpy as np
from jax import Array
from jaxtyping import Float
from maddening.core.compliance.metadata import NodeMeta, Reference
from maddening.core.params import ParamSpec
from typing_extensions import Annotated, ReadOnly, TypedDict

from mime.core import NodeFields, Symbol, SymbolKind, Units, as_run_array, jnp, node_fields, run_dtype
from mime.core import ledger as books
from mime.core._node import rechecked
from mime.core.ledger import TermOrder
from mime.nodes.flow.fvm._band import PathSolid, RegionSolid, TreeSolid, TubeSolid, WallFields, band_named, \
    interior_depth, negative_resistivity
from mime.nodes.flow.fvm._flags import OPEN_FLAGS, STALL_PER_TOLERANCE
from mime.nodes.flow.fvm._grid import MACGrid, Velocity
from mime.nodes.flow.fvm._node import BYPASS_LINE, STOKES_REYNOLDS, WALL_COUNTS, ImplicitFlowNode, StepParts, \
    FlowFields, FlowRecords, WallCounts, check_deliveries
from mime.nodes.flow.fvm._open import OpeningLayout, OpenMACGrid
from mime.nodes.flow.fvm._ports import PARAMETERS, STATES, PortLaw, PortSolution, RowKind, port_law_named, \
    port_system, reference_of, well_posed
from mime.nodes.flow.fvm._schemes import StageParts, StepContext
from mime.nodes.flow.fvm._solve import Stopping, preconditioner_named, projected_solve, recheck_stopping, \
    stagnation_window
from mime.registry import Problem
from mime.scene._region import PATH_ROW

#: The roles the open node's hosted rows take beside its viscosity's and wall's: each opening's law, by its index,
#: and the kinetic energy convection carries through the openings.
KINETIC_FLUX = "kinetic_flux"
#: The open node's numerical terms: the scheme's, and the projection's own at the open faces.
OPEN_TERMS = ("backward_euler", "convection", "krylov", "projection")
OPENING = "opening_"
#: The hosted row of the backflow term's work on the outlets (decision 12): one line under ``<node>.backflow``.
BACKFLOW = "backflow"
#: The backflow term's coefficient (decision 12: beta 1, declared, not fitted).
BACKFLOW_BETA = 1.0
#: The outlet watch's line (decision 13): the share of an opening's net outward flux entering through its patch's
#: inward faces above which the run is flagged (``opening_reversed_flagged``). Placed by FB8's measurement.
REVERSAL_LINE = 0.05
#: Where the outlet watch's line comes from, said beside it.
REVERSAL_MEASURED = "declared, pending FB8's measurement"
#: The solid's share of a lone opening's net flux beyond the band's half width in steady flow, measured at 8 cells
#: across by the stiff interior (#246's FA1 = FB14 arms: 0.4 to 3.7 % at D/h 16 to 8 at 0.1h, 0.01 to 0.4 % at
#: 0.01h): the bound the exchange flag's line is derived from.
LONE_SOLID_SHARE = MappingProxyType({0.1: 0.037, 0.01: 0.004})
#: The exchange flag's line over that bound, scaled to the run (declared).
EXCHANGE_FACTOR = 2.0


def exchange_line(cells_across: float, delta_in: float) -> float:
    """The line on an opening's solid share where its plane holds another opening (``opening_exchange_flagged``):
    ``EXCHANGE_FACTOR`` times the lone opening's measured share at the stiff interior, scaled to the narrowest lumen's
    cells across as the solid's Darcy flux scales with ``(h / R)^2``; derived from the lone opening, not fitted to the
    exchange it flags."""
    return EXCHANGE_FACTOR * LONE_SOLID_SHARE[delta_in] * (8.0 / cells_across) ** 2
#: The float32 flow's distance from float64's with ports, over the largest speed, measured on the built node over the
#: open rig's draws (``tests/flow/open_cases.py``; the probe of the review's fix round): the envelope, not one point.
FLOAT32_PORTS = ("the flow within 7.2e-4 of float64's over the largest speed on 190 drawn open runs inside the CFL line "
                 "(median 3.3e-5, the larger at the higher Reynolds numbers; D/h 8 to 16, Re 1 to 100, a downstream "
                 "resistance to 1e4 times the vessel's, 60 steps), measured on the built node; the open-ports spike's "
                 "1.6e-6 to 3.5e-6 was one point at D/h 16")


#: Why an open vessel's flow bounds no step of its own, said where ``dt='auto'`` is refused and where the CFL check
#: stops a run.
OPEN_STEP = ("the open vessel's flow (FVM(domain=VesselRegion())) has no step rule yet: the CFL number it would predict "
             "from the inflow's peak through the narrowest lumen comes with the region builder, so its dt is given and "
             "its CFL number checked each step")


#: The planes' names, by their indices (``mime.nodes.flow.fvm._open.LOW`` and ``HIGH``).
PLANES = ("low", "high")


def opening_role(index: int) -> str:
    """The hosted row of the node's ``index``-th law (its laws in their order, one or several openings each)."""
    return f"{OPENING}{index}"


class OpeningFields(TypedDict):
    """An open node's openings' fields beside its flow's, with their units (both stage kinds')."""

    #: (``w``'s faces include the two planes: ``velocity_z`` is ``nx ny nz+1``.) The openings' pressures (absolute)
    #: and outward fluxes the last step solved for, and each law's own state (a row a law).
    opening_pressure: ReadOnly[Annotated[Float[Array, "openings"], Units("Pa")]]
    #: Each opening's pressure relative to the gauge's reference opening (zero there): the field's own unknowns, which
    #: the next step's forces take, held apart so no move is formed as a difference of absolute pressures.
    opening_pressure_relative: ReadOnly[Annotated[Float[Array, "openings"], Units("Pa")]]
    opening_flux: ReadOnly[Annotated[Float[Array, "openings"], Units("m^3/s")]]
    opening_state: ReadOnly[Annotated[Float[Array, "laws 1"], Units("1")]]  # noqa: F722 - a jaxtyping shape
    #: The outlet watch (decision 13; an observer, read by nothing in the step): each opening's inward flux over its
    #: net outward flux while that is outward (zero otherwise), and the flag above the line.
    opening_reversed: ReadOnly[Annotated[Float[Array, "openings"], Units("1")]]
    opening_reversed_flagged: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    #: The backflow term's work over the step (J, ``dt rho <f, u'>``), and its sign check: one when it gave energy.
    backflow_work: ReadOnly[Annotated[Float[Array, ""], Units("J")]]  # noqa: F722 - a jaxtyping shape
    backflow_flagged: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    #: Each opening's net flux through its patch's faces at least the band's half width into the solid over its net
    #: flux, where its plane holds another opening (zero elsewhere): the solid between two patches exchanging flow; and
    #: the flag above :func:`exchange_line`.
    opening_exchange: ReadOnly[Annotated[Float[Array, "openings"], Units("1")]]
    opening_exchange_flagged: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape


class OpenFlowFields(OpeningFields):
    """The open node's own fields beside the periodic node's (``FlowState``'s): its openings' and its unit fields."""

    #: The unit fields, one a non-reference opening: the velocity a unit pressure there drives over a step (m/s/Pa).
    unit_x: ReadOnly[Annotated[Float[Array, "units nx ny nz"], Units("m/(s Pa)")]]  # noqa: F722 - a jaxtyping shape
    unit_y: ReadOnly[Annotated[Float[Array, "units nx ny nz"], Units("m/(s Pa)")]]  # noqa: F722 - a jaxtyping shape
    unit_z: ReadOnly[Annotated[Float[Array, "units nx ny nzp1"], Units("m/(s Pa)")]]  # noqa: F722 - a jaxtyping shape


class OpenFlowState(FlowFields, OpenFlowFields, closed=True):
    """The open flow's state without the energy ledger."""


class MeteredOpenFlowState(FlowFields, OpenFlowFields, FlowRecords, books.LawRecords, closed=True):
    """A metered open flow's state: the open flow's fields, the stall detector, the ledger's records and, where a law it
    holds stores energy, that law's own."""


class OpenTreeFlowState(FlowFields, OpenFlowFields, WallCounts, closed=True):
    """An open flow walled by a tube tree (a built region), without the energy ledger: the open flow's fields and the
    wall's report."""


class MeteredOpenTreeFlowState(FlowFields, OpenFlowFields, WallCounts, FlowRecords, books.LawRecords, closed=True):
    """A metered open flow walled by a tube tree."""


class OpenPorts(NamedTuple):
    """What the step's port coupling solved: the port system's answer (the openings' pressures, absolute and relative
    to the reference, each relative pressure's move, the fluxes its rows hold), the solve's own solution before the
    correction, the unit fields, their response, and the forces at the solved pressures."""

    solution: PortSolution
    base: Velocity
    units: tuple[Velocity, ...]
    response: Array
    force: Velocity
    #: The laws' blocks over the stage placed in one over the openings (zero on a flux row's).
    blocks: Array

    @property
    def pressures(self) -> Array:
        return self.solution.pressures


def _velocity(state: Mapping[str, Array]) -> Velocity:
    return (state["velocity_x"], state["velocity_y"], state["velocity_z"])


def _add(a: Velocity, b: Velocity, scale: Any = 1.0) -> Velocity:
    return (a[0] + scale * b[0], a[1] + scale * b[1], a[2] + scale * b[2])


#: The openings' conditions, the periodic node's momentum equation (``ImplicitFlowNode``'s) holding inside.
EQUATIONS = (r"\operatorname{p}_{n+1} = p_{k}, \quad p_{k} = \operatorname{a}_{k} + \sum_{j} \operatorname{B}_{kj} q_{j} \ "
             r"\text{or} \ q_{k} = \operatorname{Q}_{k}")
_STEPPED = "mime.nodes.flow.fvm._open_node.OpenFlowNode.stepped"


class OpenFlowNode(ImplicitFlowNode):
    """The flow field of a medium in a vessel cut at the box's two z planes, its openings patches of them, each held by a
    law the step holds, stepped by IMEX Euler (or its steady limit) on the monolithic projected CG (the module states
    the step)."""

    meta = NodeMeta(
        algorithm_id="MIME-NODE-129",
        description="A medium's flow in a vessel open at the box's two z planes: the periodic flow node's step on the "
                    "Cartesian MAC grid open along z, each opening (a patch of a plane) held by a law the step holds "
                    "(an inflow, a Windkessel, a closed valve, or one law over several openings), coupled implicitly "
                    "through the field's affine response.",
        governing_equations=EQUATIONS,
        assumptions=(
            "Inside, the periodic flow node's (ImplicitFlowNode's) momentum equation and its wall's band",
            "The grid is open along z: the faces on its two planes are evolved half cells, weighted by a half in the "
            "inner product, so the projector stays orthogonal; the pressure's field part is zero on the planes, whose "
            "pressures the openings' forces carry",
            "At an open plane the viscous flux is zero and convection's flux is the plane's own face value",
            "Each opening k's law states its pressure affine in the outward volume fluxes of the openings it binds "
            "over the stage (a, B), or the flux itself (Q); the field's pressure on the opening's patch is its "
            "pressure, seen relative to one reference opening (the gauge)",
            "A plane holding several openings is split between their centres: each of its faces, the solid's "
            "included, belongs to the nearest opening's patch",
            "The unit fields are solved in the program each step from the last, while the operator is fixed"),
        limitations=(
            "Openings on the box's two z planes only: the region builder bends each cut onto one of them",
            "IMEX Euler or its steady limit (RK3 with openings is OpenExplicitFlowNode's)",
            "The backflow term's coefficient is lagged from the step's start (first order in time in the term), the "
            "step's end velocity implicit, so it never gives energy; it acts on the normal component only",
        ),
        references=(Reference("HarlowWelch1965"), Reference("Westerhof2009")),
    )
    symbols = (
        Symbol(r"p", SymbolKind.STATE, "opening_pressure", "Pa", _STEPPED),
        Symbol(r"q", SymbolKind.STATE, "opening_flux", "m^3/s", _STEPPED),
    )

    LEDGER_TERMS = OPEN_TERMS
    GRID_TERMS = ("projection",)
    #: The grid's own term's order (``mime.testing.term_orders``, beside the scheme's ``ledger_orders``): none, said so.
    GRID_ORDERS: ClassVar[Mapping[str, TermOrder]] = MappingProxyType({"projection": TermOrder(
        reason="rounding at any tolerance: the half-weight faces keep the projector orthogonal under the books' inner "
               "product, so the projection's term does not fall with the tolerance (C13); the open_face_projection "
               "flag bounds it")})
    LEDGER_FLAGS = OPEN_FLAGS

    def __init__(self, name: str, timestep: float, *, openings: Sequence[str], opening_laws: Sequence[str],
                 opening_parameters: Any, opening_law_ids: Sequence[str] = (), initial_opening_state: Any = None,
                 opening_planes: Sequence[str] | None = None, opening_centres: Sequence[Sequence[float]] = (),
                 law_openings: Sequence[Sequence[int]] | None = None, backflow_openings: Sequence[int] | None = None,
                 reversal_line: float | None = None, opening_paths: Any = None, **settings: Any) -> None:
        names = tuple(str(o) for o in openings)
        laws = [port_law_named(str(law)) for law in opening_laws]
        planes = tuple(str(plane) for plane in opening_planes) if opening_planes is not None else \
            PLANES if len(names) == 2 else ()
        if len(planes) != len(names) or any(plane not in PLANES for plane in planes):
            raise ValueError(f"each of the openings {names} lies on the box's low or high plane: give opening_planes, "
                             f"one of {PLANES} an opening, not {planes}")
        bound = tuple(tuple(int(i) for i in held) for held in law_openings) if law_openings is not None else \
            tuple((k,) for k in range(len(laws)))
        if len(bound) != len(laws) or sorted(i for held in bound for i in held) != list(range(len(names))):
            raise ValueError(f"each opening of {names} is held by one law: the laws {tuple(opening_laws)} bind "
                             f"{bound}")
        for law, held in zip(laws, bound, strict=True):
            if len(held) != law.openings:
                raise ValueError(f"the port law {law.name!r} binds {law.openings} opening(s), not {len(held)}")
        centres = tuple((float(centre[0]), float(centre[1])) for centre in opening_centres)
        OpeningLayout(tuple(PLANES.index(plane) for plane in planes), centres)  # its own checks
        parameters = np.asarray(opening_parameters, dtype=np.float64)
        if parameters.shape != (len(laws), PARAMETERS):
            raise ValueError(f"each law takes a row of {PARAMETERS} parameters, not {parameters.shape}")
        rows, waveforms = _opening_rows(laws, bound, parameters, len(names))
        reason = well_posed(rows, waveforms)
        if reason:
            raise ValueError(reason)
        if settings.get("periodic_images", "own") != "own" or tuple(settings.get("tube_axis", (0.0, 0.0, 1.0))) != \
                (0.0, 0.0, 1.0):
            raise ValueError("an open box's vessel lies along its z axis (tilted open vessels are not built yet)")
        ids = tuple(str(i) for i in opening_law_ids) or tuple(f"{name}.{'+'.join(names[i] for i in held)}"
                                                             for held in bound)
        backflow = () if backflow_openings is None else tuple(sorted({int(k) for k in backflow_openings}))
        if any(not 0 <= k < len(names) for k in backflow):
            raise ValueError(f"the backflow term's openings are indices of {names}, not {backflow}")
        placed: dict[str, Any] = {"backflow_openings": backflow} if backflow else {}
        if reversal_line is not None:
            placed["reversal_line"] = float(reversal_line)
        if opening_paths is not None:
            paths = np.asarray(opening_paths, dtype=np.float64)
            if paths.ndim != 2 or paths.shape[1] != PATH_ROW:
                raise ValueError(f"the openings' paths are rows of {PATH_ROW} numbers, not {paths.shape}")
            placed["opening_paths"] = paths
        state = np.zeros((len(laws), STATES)) if initial_opening_state is None else np.asarray(initial_opening_state)
        layout = {} if planes == PLANES and not centres and law_openings is None else \
            {"opening_planes": planes, "opening_centres": centres, "law_openings": bound}
        super().__init__(name, timestep, openings=names, opening_laws=tuple(law.name for law in laws),
                         opening_law_ids=ids, opening_parameters=parameters, initial_opening_state=state, **layout,
                         **placed, **settings)

    # -- the pieces of a step ------------------------------------------------------------------------------------
    @override
    def grid(self, p: Mapping[str, Any]) -> MACGrid:
        return self.open_grid(p)

    def layout(self) -> OpeningLayout:
        c = self.constants()
        planes = tuple(PLANES.index(plane) for plane in c.get("opening_planes", PLANES))
        centres = tuple((float(x), float(y)) for x, y in c.get("opening_centres", ()))
        return OpeningLayout(planes, centres)

    def open_grid(self, p: Mapping[str, Any]) -> OpenMACGrid:
        c = self.constants()
        return OpenMACGrid((c["cells_x"], c["cells_y"], c["cells_z"]), p["spacing"], self.layout())

    @override
    def param_specs(self) -> dict[str, ParamSpec]:
        specs = super().param_specs()
        if "opening_paths" in self.constants():
            specs["opening_paths"] = ParamSpec(
                trainable=False, units="1", description="each opening's path from its cut to past its face in the box's "
                                                        "frame, the region builder's (lengths in m, directions unit, the "
                                                        "bend's angle in rad): made at build from the vessel's geometry")
        return specs

    @override
    def solids(self, p: Mapping[str, Any], grid: MACGrid) -> Sequence[TubeSolid | TreeSolid | RegionSolid]:
        if self.walled_by_tree:
            if "opening_paths" in self.constants():
                return [RegionSolid(self.tree_solid(p, grid), PathSolid(as_run_array(p["opening_paths"])))]
            return super().solids(p, grid)
        return [TubeSolid(as_run_array(p["tube_centre"]), as_run_array(p["tube_axis"]), as_run_array(p["tube_radius"]),
                          ((0.0, 0.0, 0.0),))]

    def backflow_openings(self) -> tuple[int, ...]:
        """The openings the backflow term acts on, as the build places it: every one but the upstream, the role the
        region's centreline gives them (round 5, item 14); none for a node made without them."""
        return tuple(self.constants().get("backflow_openings", ()))

    def opening_exchange(self, p: Mapping[str, Any], grid: OpenMACGrid, fields: WallFields, velocity: Velocity,
                         flux: Array) -> Array:
        """Each opening's solid share where its plane holds another opening (:class:`OpenFlowFields`), zero where it
        is alone on its plane or its net flux is zero."""
        layout = grid.layout
        shared = jnp.asarray([0.0 if layout.alone(k) else 1.0 for k in range(grid.opening_count)], dtype=flux.dtype)
        depth = 0.5 * band_named(self.constants()["band"]).width * grid.spacing
        solid = grid.opening_fluxes((velocity[0], velocity[1],
                                     jnp.where(fields.signed_distance[2] >= depth, velocity[2], 0.0)))
        moving = jnp.abs(flux) > 0.0
        return shared * jnp.where(moving, solid / jnp.where(moving, jnp.abs(flux), 1.0), 0.0)

    def exchange_line(self, p: Mapping[str, Any], grid: OpenMACGrid) -> float:
        """:func:`exchange_line` at the run's narrowest lumen and stiff interior."""
        c = self.constants()
        radius = float(np.min(np.asarray(c["tree_radii"]))) if self.walled_by_tree else float(c["tube_radius"])
        return exchange_line(2.0 * radius / float(c["spacing"]), interior_depth(c["interior"]))

    def backflow_coefficient(self, grid: OpenMACGrid, fields: WallFields, velocity: Velocity) -> Array | None:
        """The backflow term's coefficient (decision 12, its lagged implicit form, the term ``-c w'``) from the step's
        start ``velocity``, on the fluid of the outlets' half cells (``1 - chi``, as the drive acts); ``None`` where
        none is placed."""
        openings = self.backflow_openings()
        if not openings:
            return None
        return grid.backflow_coefficient(velocity, openings, BACKFLOW_BETA, 1.0 - fields.chi[2])

    def laws(self) -> list[PortLaw]:
        return [port_law_named(law) for law in self.constants()["opening_laws"]]

    def law_openings(self) -> tuple[tuple[int, ...], ...]:
        """The openings each law binds, in the laws' order."""
        c = self.constants()
        held = c.get("law_openings")
        return tuple(tuple(o) for o in held) if held is not None else tuple((k,) for k in range(len(c["opening_laws"])))

    def opening_count(self) -> int:
        return len(self.constants()["openings"])

    def rows(self) -> list[RowKind]:
        """Each opening's row kind, its law's."""
        rows = [RowKind.FLUX] * self.opening_count()
        for law, held in zip(self.laws(), self.law_openings(), strict=True):
            for opening in held:
                rows[opening] = law.row
        return rows

    def reference(self) -> int:
        return reference_of(self.rows())

    def others(self) -> list[int]:
        reference = self.reference()
        return [index for index in range(self.opening_count()) if index != reference]

    def law_rows(self, parameters: Array, law_state: Array, instant: Any, dt: Any) -> tuple[Array, Array]:
        """The stage's rows over the openings: each opening's ``a`` and the laws' blocks placed in one ``B``."""
        count = self.opening_count()
        dtype = parameters.dtype
        a = jnp.zeros(count, dtype=dtype)
        block = jnp.zeros((count, count), dtype=dtype)
        for index, (law, held) in enumerate(zip(self.laws(), self.law_openings(), strict=True)):
            own_a, own_b = law.affine(parameters[index], law_state[index], instant, dt)
            own_a, own_b = jnp.reshape(own_a, (len(held),)), jnp.reshape(own_b, (len(held), len(held)))
            for i, row in enumerate(held):
                a = a.at[row].set(own_a[i])
                for j, column in enumerate(held):
                    block = block.at[row, column].set(own_b[i, j])
        return a, block

    def coupling(self) -> list[list[bool]]:
        """Which of ``B``'s entries a law places: its openings' with each other's."""
        count = self.opening_count()
        held = [[False] * count for _ in range(count)]
        for openings in self.law_openings():
            for i in openings:
                for j in openings:
                    held[i][j] = True
        return held

    def per_law(self, law: PortLaw, held: tuple[int, ...], values: Array) -> Array:
        """``values`` over the openings as ``law`` takes them: a scalar for one opening, a vector for several."""
        return values[held[0]] if law.openings == 1 else jnp.stack([values[i] for i in held])

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
        openings = self.opening_count()
        unit = {f"unit_{x}": jnp.zeros((count, *grid.shape(k)), dtype=dtype) for k, x in enumerate("xyz")}
        z = as_run_array(0.0)
        held = {**self.initial_fields(), "opening_pressure": jnp.zeros(openings, dtype=dtype),
                "opening_pressure_relative": jnp.zeros(openings, dtype=dtype),
                "opening_flux": jnp.zeros(openings, dtype=dtype),
                "opening_state": jnp.asarray(c["initial_opening_state"], dtype=dtype),
                "opening_reversed": jnp.zeros(openings, dtype=dtype), "opening_reversed_flagged": z,
                "backflow_work": z, "backflow_flagged": z, "opening_exchange": jnp.zeros(openings, dtype=dtype),
                "opening_exchange_flagged": z, **unit}
        if self.walled_by_tree:
            held.update(self.wall_report(jnp.zeros(len(WALL_COUNTS), dtype=dtype)))
        if not self.metered:
            return node_fields(held)
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
        velocity = _velocity(state)
        # The backflow term (decision 12, its lagged implicit form): its coefficient from the step's start, the step's
        # end velocity implicit, a diagonal on the outlets' half cells beside the wall's penalty (zero where none is
        # placed or no face is inward, the operator then the wall's alone).
        lagged = self.backflow_coefficient(grid, fields, velocity)
        stepped_sigma = sigma if lagged is None else (sigma[0], sigma[1], sigma[2] + lagged)
        operator = self.operator_of(p, grid, stepped_sigma)
        stopping = Stopping(tolerance=as_run_array(p["tolerance"]), at_floor=bool(c["at_floor"]), cap=int(c["cap"]),
                            check_every=int(c["check_every"]),
                            window=stagnation_window(grid.cells, int(c["check_every"])))
        if rechecked(p):
            stopping = recheck_stopping(stopping)
        preconditioner = preconditioner_named(c["preconditioner"])
        steps = jnp.asarray(state["steps"])
        tap = self.report_row(p, steps)
        magnitude = self.magnitude_of(p, grid, stepped_sigma)
        rho = as_run_array(p["density"])
        laws, bound = self.laws(), self.law_openings()
        rows = self.rows()
        reference, others = self.reference(), self.others()
        count = self.opening_count()
        parameters = as_run_array(p["opening_parameters"])
        law_state = jnp.asarray(state["opening_state"])
        predicted = jnp.asarray(state["opening_pressure_relative"])
        time = steps.astype(dtype) * dt
        instant = time + scheme.drive_instants[0] * dt
        held_units = [(state["unit_x"][k], state["unit_y"][k], state["unit_z"][k]) for k in range(len(others))]
        solved_ports: dict[str, OpenPorts] = {}

        def solve(shift: Any, rhs: Velocity, guess: Velocity | None) -> tuple[Velocity, Array]:
            def one(b: Velocity, start: Velocity | None) -> tuple[Velocity, Array]:
                return projected_solve(grid, operator(shift), self.diagonal_of(p, grid, stepped_sigma, shift), b,
                                       start if c["scaling_start"] else None, stopping, tap, preconditioner,
                                       magnitude(shift))

            base, report = one(_add(rhs, grid.port_force(predicted, rho)), guess)
            units: list[Velocity] = []
            reports = [report]
            for k, j in enumerate(others):
                unit, unit_report = self.unit_field(one, grid.port_force(jnp.zeros(count, dtype=dtype).at[j].set(1.0),
                                                                         rho), held_units[k])
                units.append(unit)
                reports.append(unit_report)
            response = jnp.stack([grid.opening_fluxes(unit) for unit in units], axis=1) if units else \
                jnp.zeros((count, 0), dtype=dtype)
            a, blocks = self.law_rows(parameters, law_state, instant, dt)
            solution = self.solve_ports(rows, a, blocks, grid.opening_fluxes(base), response, predicted, reference)
            solved = base
            for k, j in enumerate(others):
                solved = _add(solved, units[k], solution.moved[j])
            solved_ports["ports"] = OpenPorts(solution, base, tuple(units), response,
                                              grid.port_force(solution.relative, rho), blocks)
            return solved, combined(reports)

        drive = self.forcing(fields, boundary_inputs, scheme, dtype)
        context = StepContext(grid, operator, drive, bool(c["convection"]), solve, self.kinematic(p), sigma)
        speed = jnp.max(jnp.stack([jnp.max(jnp.abs(component)) for component in velocity]))
        cfl = speed * dt / grid.spacing
        checked = bool(c["convection"]) and scheme.cfl_line is not None
        stopped_before = jnp.asarray(state["stopped_at"]) >= 0
        past = ~(cfl <= scheme.cfl_limit(self.cfl_axis(p))) if checked else jnp.asarray(False)
        ended_state, stage = scheme.step(context, {"velocity": _velocity(state)}, time, dt)
        ports = solved_ports["ports"]
        ended_velocity: Velocity = ended_state["velocity"]
        flux = grid.opening_fluxes(ended_velocity)
        law_ended = jnp.stack([law.advanced(parameters[i], law_state[i], self.per_law(law, held, flux),
                                            self.per_law(law, held, ports.pressures), dt)
                               for i, (law, held) in enumerate(zip(laws, bound, strict=True))])
        report = stage.report
        flagged = (report[2] + report[3] + report[4] > 0).astype(dtype)
        axis = self.vessel_axis(p)
        axis = axis / jnp.sqrt(jnp.sum(axis * axis))
        lumen, solid, lumen_scale = self.fluxes(grid, fields, axis, ended_velocity)
        ratio = self.ratio(solid, lumen, lumen_scale)
        reynolds = 2.0 * jnp.abs(lumen) / (jnp.pi * self.vessel_radius(p) * self.kinematic(p))
        stokes = not bool(c["convection"])
        solved_velocity: Velocity = ended_state["velocity"]
        # The outlet watch (decision 13): an observer, read by nothing above.
        inward = grid.opening_reversed_fluxes(solved_velocity, 1.0 - fields.chi[2])
        outward = flux > 0.0
        reversed_share = jnp.where(outward, inward / jnp.where(outward, flux, 1.0), 0.0)
        watched = jnp.max(reversed_share) > float(c.get("reversal_line", REVERSAL_LINE))
        zero_force = grid.zeros(dtype)
        backflow: Velocity = zero_force if lagged is None else \
            (zero_force[0], zero_force[1], -lagged * solved_velocity[2])
        backflow_work = dt * as_run_array(p["density"]) * grid.inner(backflow, solved_velocity)
        backflow_size = dt * as_run_array(p["density"]) * grid.inner_magnitude(backflow, solved_velocity)
        gave = backflow_work > books.IDENTITY_ULPS * jnp.finfo(dtype).eps * backflow_size
        exchange = self.opening_exchange(p, grid, fields, solved_velocity, flux)
        exchanged = jnp.max(jnp.abs(exchange)) > self.exchange_line(p, grid)
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
            opening_pressure=ports.pressures.astype(dtype),
            opening_pressure_relative=ports.solution.relative.astype(dtype), opening_flux=flux.astype(dtype),
            opening_state=law_ended.astype(dtype), opening_reversed=reversed_share.astype(dtype),
            opening_reversed_flagged=watched.astype(dtype), backflow_work=backflow_work.astype(dtype),
            backflow_flagged=gave.astype(dtype), opening_exchange=exchange.astype(dtype),
            opening_exchange_flagged=exchanged.astype(dtype),
            **{f"unit_{x}": (jnp.stack([unit[k] for unit in ports.units]) if ports.units else
                             jnp.asarray(state[f"unit_{x}"])) for k, x in enumerate("xyz")})
        if fields.counts is not None:
            ended.update(self.wall_report(fields.counts))
        held = node_fields(ended)
        energy: dict[str, Array] = {}
        if self.metered:
            energy = self.open_energy_terms(p, grid, velocity, stage, sigma, fields, dt, ports, flux, law_state,
                                            law_ended, parameters, backflow, stepped_sigma)
            stall = self.stall_ratio(energy)
            line = STALL_PER_TOLERANCE[jnp.dtype(dtype).name] * as_run_array(p["tolerance"])
            held = {**held, "stall_ratio": stall, "stall_flagged": (stall > line).astype(dtype)}
        return held, StepParts(stage, fields, energy)

    def solve_ports(self, rows: Sequence[RowKind], a: Array, blocks: Array, base: Array, response: Array,
                    predicted: Array, reference: int) -> PortSolution:
        """The port system (``mime.nodes.flow.fvm._ports.port_system``) with each law's block in place."""
        return port_system(rows, a, blocks, base, response, predicted, reference, self.coupling())

    def superposed_scale(self, grid: OpenMACGrid, ports: OpenPorts) -> Array:
        """Each opening's superposition's scale (m^3/s): ``h^2 (sum |w_base| + sum_j |m_j| sum |w_unit_j|)`` over its
        patch's faces, the magnitude its face flux is summed from."""
        total = grid.opening_fluxes_magnitude(ports.base)
        for unit, j in zip(ports.units, self.others(), strict=True):
            total = total + jnp.abs(ports.solution.moved[j]) * grid.opening_fluxes_magnitude(unit)
        return total

    def opening_solid_fluxes(self, p: Mapping[str, Any], velocity: Velocity, depth: Any = None) -> Array:
        """The volume flux (m^3/s) out through each opening's faces that lie at least ``depth`` (m) into the solid: by
        default the half band's width, where the lumen's flux ends (:meth:`fluxes`); the whole plane's flux an
        ``Inflow`` pins holds this share (R5-C-02: one masked sum a step)."""
        grid = self.open_grid(p)
        fields = self.fields(p, grid)
        if depth is None:
            depth = 0.5 * band_named(self.constants()["band"]).width * grid.spacing
        distance = fields.signed_distance[2]
        solid = jnp.where(distance >= depth, velocity[2], 0.0)
        return grid.opening_fluxes((velocity[0], velocity[1], solid))

    @override
    def stop_problem(self, state: Mapping[str, Any]) -> Problem | None:
        """The periodic node's, its routes the open vessel's: no ``dt='auto'`` and no steady drive, the step taken from
        the inflow's peak."""
        problem = super().stop_problem(state)
        if problem is None:
            return None
        cfl = float(np.asarray(state["stopped_cfl"]))
        line = float(self.scheme.cfl_limit(self.constants().get("tube_axis", (1.0, 1.0, 1.0))))
        safe = f"dt <= {self.delta_t * line / cfl:.4g} s, " if math.isfinite(cfl) and cfl > 0.0 else "a smaller dt, "
        return dataclasses.replace(problem, action=(
            f"use a timestep under {safe}the line at the speed the flow had reached by then: a flow still speeding up "
            f"needs a smaller one (Poiseuille's centreline at the inflow's peak mean speed, twice it, over the narrowest "
            f"lumen's share of the vessel's area, bounds it); dt='auto' is refused here ({OPEN_STEP}); or the Stokes "
            f"approximation FVM(stokes=True) where convection is negligible"))

    @staticmethod
    @override
    def fluid_momentum(grid: MACGrid, fluid: Velocity, velocity: Velocity, rho: Array) -> dict[str, Array]:
        """The periodic node's, each face weighted as the books' inner product weighs it (the planes' half cells at a
        half): a body force on the open box's fluid does its work over those weights."""
        assert isinstance(grid, OpenMACGrid)
        weights = grid.weights(velocity[0].dtype)
        volume = grid.cell_volume
        momentum = jnp.stack([rho * volume * jnp.sum(weights[c] * fluid[c] * velocity[c]) for c in range(3)])
        size = jnp.stack([rho * volume * jnp.sum(jnp.abs(weights[c] * fluid[c] * velocity[c])) for c in range(3)])
        return {"momentum": momentum, "momentum_size": size}

    def unit_field(self, solve: Any, force: Velocity, start: Velocity) -> tuple[Velocity, Array]:
        """The unit field a unit pressure at one opening drives over the step (``force`` its force), solved from
        ``start`` (the last step's) by the step's own ``solve``, and its report."""
        return solve(force, start)

    # -- the step identity ------------------------------------------------------------------------------------------
    def open_energy_terms(self, p: Mapping[str, Any], grid: OpenMACGrid, before: Velocity, stage: StageParts,
                          sigma: Velocity, fields: WallFields, dt: float, ports: OpenPorts, flux: Array,
                          law_state: Array, law_ended: Array, parameters: Array, backflow: Velocity | None = None,
                          stepped_sigma: Velocity | None = None) -> dict[str, Array]:
        """Each term of the step's identity (J), the openings' laws' work and own books, the fluid's momentum the
        drive is paired with, and the magnitude of the arithmetic forming them (``scale``). ``stepped_sigma`` is the
        diagonal the step solved with (the wall's ``sigma`` and the backflow term's coefficient), ``backflow`` that
        term's force at the solution: the wall's work is booked from ``sigma`` alone, the term's apart."""
        fluid = (1.0 - fields.chi[0], 1.0 - fields.chi[1], 1.0 - fields.chi[2])
        rho, nu = as_run_array(p["density"]), self.kinematic(p)
        solved = stage.solved
        inner, size = grid.inner, grid.inner_magnitude
        laplacian = grid.laplacian(solved)
        minus_laplacian = (-laplacian[0], -laplacian[1], -laplacian[2])
        penalised = (sigma[0] * solved[0], sigma[1] * solved[1], sigma[2] * solved[2])
        forced = _add(stage.rhs, ports.force)
        applied = self.operator_of(p, grid, sigma if stepped_sigma is None else stepped_sigma)(stage.shift)(solved)
        remainder = _add(forced, applied, -1.0)
        gradient = grid.gradient(grid.pressure_of(remainder))
        projected = _add(remainder, gradient, -1.0)
        change = _add(solved, before, -1.0)
        stores = self.scheme.stores_inertia
        zero = jnp.zeros((), dtype=solved[0].dtype)
        laws, bound = self.laws(), self.law_openings()
        own = [law.books(parameters[i], law_state[i], law_ended[i], self.per_law(law, held, flux),
                         self.per_law(law, held, ports.pressures), dt)
               for i, (law, held) in enumerate(zip(laws, bound, strict=True))]
        superposed = self.superposed_scale(grid, ports)
        # Each law's own scale (``mime.core.ledger.LAW_BOOKS``' last column): its terms' magnitudes, and the face flux's
        # summation floor carried through its block. The law steps with its openings' face fluxes, each a sum over its
        # patch's N faces of the superposed field, and its pressures solve its rows at the rows' fluxes; the two differ
        # by at most (N + 2) eps of the superposition's scale S (three sums of N terms, the faces' own rounding), so
        # its identity's residual, dt sum_k (B (q_row - q_face))_k q_face_k, is bounded by dt sum_k |(B q)_k| (N_k + 2)
        # eps S_k, written here in the identity floor's units (IDENTITY_ULPS eps).
        faces = grid.opening_faces()
        pushed = jnp.stack([sum((ports.blocks[i, j] * flux[j] for j in range(len(faces)) if self.coupling()[i][j]),
                                start=jnp.zeros((), dtype=flux.dtype)) for i in range(len(faces))])
        law_rows = [jnp.stack([b.storage_start, b.storage_end, b.loss, b.numerical,
                               sum((jnp.abs(dt * ports.pressures[k] * flux[k]) for k in held),
                                   start=jnp.zeros((), dtype=flux.dtype))
                               + jnp.abs(b.storage_start) + jnp.abs(b.storage_end) + jnp.abs(b.loss)
                               + jnp.abs(b.numerical)
                               + sum((dt * jnp.abs(pushed[k]) * (faces[k] + 2) * superposed[k] for k in held),
                                     start=jnp.zeros((), dtype=flux.dtype)) / books.IDENTITY_ULPS
                               if laws[i].stores or laws[i].own_books else jnp.zeros((), dtype=flux.dtype)])
                    for i, (b, held) in enumerate(zip(own, bound, strict=True))]
        terms = {
            **self.fluid_momentum(grid, fluid, solved, rho),
            "viscous": dt * rho * nu * inner(minus_laplacian, solved),
            "penalty": dt * rho * inner(penalised, solved),
            "backward_euler": 0.5 * rho * inner(change, change) if stores else zero,
            "convection": -dt * rho * inner(stage.convection, change),
            "krylov": dt * rho * inner(projected, solved),
            "projection": dt * rho * inner(gradient, solved),
            KINETIC_FLUX: dt * rho * inner(stage.convection, before),
            BACKFLOW: dt * rho * inner(backflow, solved) if backflow is not None else zero,
            "ports": jnp.stack([sum((-dt * ports.pressures[k] * flux[k] for k in held[1:]),
                                    start=-dt * ports.pressures[held[0]] * flux[held[0]]) for held in bound]),
            "law_books": jnp.stack(law_rows),
        }
        # The openings' work is judged on its own magnitude, the pressures against the faces' flux in magnitude.
        port_scale = dt * jnp.sum(jnp.abs(ports.pressures) * grid.opening_fluxes_magnitude(solved))
        law_scale = sum((jnp.abs(b.storage_start) + jnp.abs(b.storage_end) + jnp.abs(b.loss) + jnp.abs(b.numerical)
                         for b in own), start=zero)
        scale = rho * (dt * (size(forced, solved) + size(applied, solved) + size(stage.convection, solved)
                             + size(stage.convection, before) + nu * size(minus_laplacian, solved)
                             + size(penalised, solved) + size(gradient, solved)
                             + (size(backflow, solved) if backflow is not None else 0.0))
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
        backflow = ((BACKFLOW, f"{self.name}.backflow", books.DISSIPATIVE),) if self.backflow_openings() else ()
        return (*super().hosted_laws(), *openings, (KINETIC_FLUX, f"{self.name}.openings", books.UNMETERED),
                *backflow)

    @override
    def storing_laws(self) -> tuple[str, ...]:
        c = self.constants()
        return tuple(str(law_id) for law_id, law in zip(c["opening_law_ids"], self.laws(), strict=True) if law.stores)

    def keeps_books(self) -> bool:
        """Whether some law the node holds keeps its own books (a storing law, or one over several openings)."""
        return any(law.stores or law.own_books for law in self.laws())

    @override
    def law_storage(self, p: Mapping[str, Any], state: Mapping[str, Array]) -> Array | None:
        """Each held law's own storage at ``state``, a row per law: the Windkessels' capacitors, zero for a law that
        keeps books and stores nothing; ``None`` where no opening's law keeps its own books."""
        laws, bound = self.laws(), self.law_openings()
        if not self.keeps_books() or not self.metered:
            return None
        held = books.deliveries(self.constants())
        rows = books.laws(held)
        storage = jnp.zeros(len(rows), dtype=run_dtype())
        parameters = as_run_array(p["opening_parameters"])
        law_state = jnp.asarray(state["opening_state"])
        for delivery in held:
            if delivery.kind == books.HOSTED and delivery.field.startswith(OPENING):
                index = int(delivery.field.removeprefix(OPENING))
                none = jnp.zeros(() if laws[index].openings == 1 else (len(bound[index]),), dtype=run_dtype())
                own = laws[index].books(parameters[index], law_state[index], law_state[index], none, none, 0.0)
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
                                         KINETIC_FLUX: energy[KINETIC_FLUX], BACKFLOW: energy[BACKFLOW],
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
                                          start=jnp.zeros((), dtype=dtype))
        terms = jnp.stack([energy[name] for name in OPEN_TERMS])
        storing = self.keeps_books()
        return books.StepAccount(work=work, hosted=jnp.zeros((len(rows), 2), dtype=dtype), unattributed=unattributed,
                                 terms=terms, storage_start=self.storage(p, state), storage_end=self.storage(p, ended),
                                 scale=energy["scale"] + drive_scale, operators={},
                                 stored=stored if storing else None)


def _opening_rows(laws: Sequence[PortLaw], bound: Sequence[tuple[int, ...]], parameters: np.ndarray,
                  count: int) -> tuple[list[RowKind], list[tuple[float, ...]]]:
    """Each opening's row kind, and each opening's waveform where its law is a one-opening flux row (its four
    parameters), as ``well_posed`` reads them."""
    rows: list[RowKind] = [RowKind.FLUX] * count
    waveforms: list[tuple[float, ...]] = [(0.0, 0.0, 0.0, 0.0)] * count
    for law, held, row in zip(laws, bound, parameters.tolist(), strict=True):
        for opening in held:
            rows[opening] = law.row
            if law.openings == 1:
                waveforms[opening] = tuple(row)
    return rows, waveforms


def combined(reports: Sequence[Array]) -> Array:
    """One report for a step's solves: the iterations and counts summed, the residual the worst."""
    total = sum(reports[1:], start=reports[0])
    worst = jnp.max(jnp.stack([report[1] for report in reports]))
    return total.at[1].set(worst)


def check_open_deliveries(held: Sequence[books.LedgerDelivery]) -> None:
    """:func:`check_deliveries` with the open node's own hosted rows: each opening's law and the kinetic flux's line."""
    plain = [d for d in held if not (d.kind == books.HOSTED and (d.field in (KINETIC_FLUX, BACKFLOW)
                                                                  or d.field.startswith(OPENING)))]
    check_deliveries(plain)


__all__ = ["BACKFLOW", "BACKFLOW_BETA", "EXCHANGE_FACTOR", "FLOAT32_PORTS", "LONE_SOLID_SHARE", "exchange_line", "KINETIC_FLUX", "OPENING", "OPEN_STEP", "PLANES", "REVERSAL_LINE",
           "REVERSAL_MEASURED", "MeteredOpenFlowState", "MeteredOpenTreeFlowState", "OpenFlowState", "OpenFlowFields", "OpeningFields",
           "OpenFlowNode", "OpenPorts", "OpenTreeFlowState", "check_open_deliveries",
           "combined", "opening_role"]
