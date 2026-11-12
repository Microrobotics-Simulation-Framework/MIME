"""The flow node's Schwarz partner (F5): an MRS body the flow couples by Schwarz, held in the flow's node with the
staggered order the build gives it (the body's dynamics stepped before its medium's flow).

**The order.** At its step the flow (1) composes the step the body has just taken: the body's twist (this step's,
read after the body's own step) closes the tractions the flow prepared for it, and one walled solve adds their field to
the base (``mime.nodes.flow.fvm._schwarz.composed``), so that the flow's stored field and its kinetic energy are the
solved ones; (2) steps its scheme from the composed field, no body's force in it, for the base of the next step (its
drive evaluated from the delivered waveform at that step's instant: a declared schedule's coefficients do not move
within a run); and (3) prepares the next step at the pose the body reached: the seven columns and the response and load
handed to the body (``prepared``). The body reads them as the flow's state at the start of its next step (the lag rule's
planned read of a thing's published state). The arithmetic is the flow-first step's at the start of the next step, since
every input of the preparation is known at the end of this one: exact, not lagged.

**What it holds as state**: the prepared base field, the seven columns, the pose they were made at, the response and
load handed, the free box's field and image term after the step, the preparation's report blocks, the composed
residual and its flag, the handed response's definiteness margin, the body's gap to the wall in cells and its
near-contact window's flag; lagged, the last step's tractions; in an open vessel, the base's unit fields and its
openings' solution (``mime.nodes.flow.fvm._open_node.SCHWARZ_PORTS``), which each product's and the composition's walled
solve eliminate the openings with. The run's first step reads a preparation made in
the flow's initial state at the body's starting pose and the drive at the first step's end, as the build gives them,
from the build's leaves; a differentiated run makes it again from its own (``MimeNode.started``). A restart from saved state carries the preparation; a parameter the preparation read, changed at a fork, makes it again.

**The near field** is the medium's joint near field (``mime.nodes.hydro.near_field.JointDragNode``, on its iterative
route, which holds each member's free part and its inverse), hosted: its constants are the flow's under ``schwarz_``,
its viscosity the flow's own."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from jax import Array
from maddening.core.params import ParamSpec

from mime.core import as_run_array, jnp, run_dtype
from mime.core._node import DIFFERENTIATED, RUN_CONSTANTS_GIVEN
from mime.nodes.flow.fvm._grid import MACGrid, TRUNCATION
from mime.nodes.flow.fvm._schwarz import COLUMNS, SchwarzBody, box_side
from mime.nodes.hydro.near_field._gmres import REPORT_SIZE
from mime.nodes.hydro.near_field._node import JointDragNode

#: The prefix of the flow's constants and fields that are its Schwarz partner's.
PREFIX = "schwarz_"
#: The flow's own constants for its Schwarz partner (the rest under :data:`PREFIX` are the hosted near field's): the
#: bodies, the laws' ids, the grid's origin in the world, the bodies' starting poses and the drive at the first step's
#: end (the initial preparation's), the GMRES's tolerance, the body's extent, the report leaf, the scheme
#: (:data:`SCHEMES`), and the free box's side in cells where it is written (a structural setting; else
#: ``mime.nodes.flow.fvm._schwarz.box_side`` of the extent).
OWN = ("schwarz_members", "schwarz_ids", "schwarz_origin", "schwarz_starts", "schwarz_drive", "schwarz_tolerance",
       "schwarz_extent", "schwarz_report", "schwarz_box", "schwarz_scheme")
#: The schemes a Schwarz body is coupled by (``Drag.schwarz_scheme``): exact, one pass a step from the last step's
#: tractions, and the body reading the flow with nothing delivered.
SCHEMES = ("one_pass", "lagged", "one_way")
#: The hosted near field's constants the flow does not hold under the prefix: its viscosity is the flow's, and its report
#: leaf the flow's own (two blocks, :data:`REPORT_BLOCKS`).
NOT_HOSTED = ("viscosity", "solve_report", "solve_report_rows")
#: The exchange kernel's width at the force points, in cells: the coupling spike's (``sigma = h``); a structural setting.
WIDTH = 1.0
#: The report blocks of the preparation's GMRES: its own, then its products' walled solves'.
REPORT_BLOCKS = 2 * REPORT_SIZE
#: The near-contact window's line (the window spike's, decided): the contact's local radius at least this many cells
#: of the grid that resolves it (the patch's fine cells; the coarse grid's without the patch) ...
WINDOW_RADIUS_CELLS = 20.0
#: ... and the hand-over gap at least this many of them and at most :data:`NEAR_SHARE` of the local radius. A body
#: nearer the wall than the larger of the two is near contact; near contact below the line is flagged, never refused.
WINDOW_GAP_CELLS = 2.0
NEAR_SHARE = 0.1
#: Each Schwarz body's inputs (``schwarz_<b>_<name>``): its pose and twist, read after its own step.
INPUTS = {"pose": ((7,), "m; 1"), "twist": ((6,), "m/s; rad/s")}
#: What the flow hands each body (``schwarz_<b>_<name>``): its response and its load at zero twist.
HANDED = ("resistance", "load")


class SchwarzHost:
    """The Schwarz partners of a flow node ``name`` (one body until the body-set host), built from the flow's constants
    under :data:`PREFIX`: the hosted near field and the free box's side."""

    def __init__(self, name: str, timestep: float, constants: Mapping[str, Any], spacing: float) -> None:
        members = tuple(str(m) for m in constants["schwarz_members"])
        hosted = {key.removeprefix(PREFIX): value for key, value in constants.items()
                  if key.startswith(PREFIX) and key not in OWN}
        tolerance = constants.get("schwarz_tolerance")
        self.near = JointDragNode(f"{name}/near_field", timestep, members=members,
                                  viscosity=float(constants["viscosity"]), solve="gmres",
                                  **({"tolerance": float(tolerance)} if tolerance is not None else {}), **hosted)
        self.spacing = float(spacing)
        self.scheme = str(constants.get("schwarz_scheme", "one_pass"))  # one of SCHEMES, as Drag.schwarz_scheme gives
        given = constants.get("schwarz_box")
        self.side = int(given) if given is not None else box_side(float(constants["schwarz_extent"]), self.spacing,
                                                                  int(-(-TRUNCATION * WIDTH // 1)))

    @property
    def count(self) -> int:
        return len(self.near.members)

    def hosted_params(self) -> dict[str, Any]:
        """The hosted near field's leaves under :data:`PREFIX` (its held free parts and inverses among them)."""
        return {f"{PREFIX}{key}": value for key, value in self.near.params_pytree().items() if key not in NOT_HOSTED}

    def hosted_specs(self) -> dict[str, ParamSpec]:
        return {f"{PREFIX}{key}": spec for key, spec in self.near.param_specs().items() if key not in NOT_HOSTED}

    def near_constants(self, p: Mapping[str, Any]) -> dict[str, Any]:
        """The hosted near field's constants from the flow's ``p``: its own under :data:`PREFIX`, the flow's viscosity,
        and the step's markers (run-constants given, a differentiated run)."""
        mine = {key.removeprefix(PREFIX): value for key, value in p.items() if key.startswith(PREFIX) and key not in OWN}
        markers = {key: p[key] for key in (RUN_CONSTANTS_GIVEN, DIFFERENTIATED) if key in p}
        return {**self.near.constants(), **mine, "viscosity": p["viscosity"], **markers}

    def run_constants(self, params: Mapping[str, Any]) -> dict[str, Array]:
        held = self.near.run_constants(self.near_constants(params))
        return {f"{PREFIX}{key}": value for key, value in held.items()}

    def made_again(self, params: Mapping[str, Any]) -> dict[str, Any]:
        held = self.near.made_again(self.near_constants(params))
        return {f"{PREFIX}{key}": value for key, value in held.items()}

    def body(self, p: Mapping[str, Any], pose: Array) -> SchwarzBody:
        """The body at ``pose`` as the step holds it (``SchwarzBody``): its placement, its free-space near field at the
        flow's viscosity and its preconditioner, from the hosted near field."""
        joint, blocks, transposed = self.near.free_space_system(self.near_constants(p), pose[None, :])
        return SchwarzBody(joint.placed[0], joint.matrix, blocks[0], transposed[0])

    def box_grid(self) -> MACGrid:
        return MACGrid((self.side, self.side, self.side), self.spacing)

    def fields(self, count: int, points: int, grid: MACGrid) -> dict[str, Array]:
        """The partner's state fields at the run's start, zeros (the initial preparation fills the prepared ones)."""
        dtype = run_dtype()
        box = self.box_grid()
        held: dict[str, Array] = {}
        for k, x in enumerate("xyz"):
            held[f"{PREFIX}base_{x}"] = jnp.zeros(grid.shape(k), dtype=dtype)
            held[f"{PREFIX}own_{x}"] = jnp.zeros(box.shape(k), dtype=dtype)
        for b in range(count):
            held[f"{PREFIX}{b}_columns"] = jnp.zeros((3 * points, COLUMNS), dtype=dtype)
            held[f"{PREFIX}{b}_at"] = jnp.zeros(7, dtype=dtype)
            held[f"{PREFIX}{b}_resistance"] = jnp.zeros((6, 6), dtype=dtype)
            held[f"{PREFIX}{b}_load"] = jnp.zeros(6, dtype=dtype)
            held[f"{PREFIX}{b}_margin"] = as_run_array(0.0)
            held[f"{PREFIX}{b}_wall_gap_cells"] = as_run_array(0.0)
            held[f"{PREFIX}{b}_near_contact_window"] = as_run_array(0.0)
            if self.scheme == "lagged":
                held[f"{PREFIX}{b}_tractions"] = jnp.zeros(3 * points, dtype=dtype)
        held[f"{PREFIX}image"] = jnp.zeros(3, dtype=dtype)
        held[f"{PREFIX}report"] = jnp.zeros(REPORT_BLOCKS, dtype=dtype)
        held[f"{PREFIX}residual"] = as_run_array(0.0)
        held[f"{PREFIX}flagged"] = as_run_array(0.0)
        held[f"{PREFIX}indefinite"] = as_run_array(0.0)
        return held


def window(gap_cells: Any, radius_cells: float) -> Any:
    """Whether a body ``gap_cells`` from the wall with a local radius of ``radius_cells`` (both in the resolving grid's
    cells) is near contact below the window's line: nearer than :data:`WINDOW_GAP_CELLS` (the grid does not resolve
    the gap), or within :data:`NEAR_SHARE` of its radius with a radius under :data:`WINDOW_RADIUS_CELLS`."""
    near = gap_cells <= NEAR_SHARE * radius_cells
    return (gap_cells < WINDOW_GAP_CELLS) | (near & (radius_cells < WINDOW_RADIUS_CELLS))


def starts(constants: Mapping[str, Any], count: int) -> list[Array]:
    """Each body's starting pose from the flow's constants (the initial preparation's)."""
    held = jnp.asarray(constants["schwarz_starts"], dtype=run_dtype()).reshape(count, 7)
    return [held[b] for b in range(count)]


def inputs_of(count: int) -> dict[str, tuple[tuple[int, ...], str]]:
    return {f"{PREFIX}{b}_{name}": spec for b in range(count) for name, spec in INPUTS.items()}


def handed_fields(count: int) -> Sequence[str]:
    return tuple(f"{PREFIX}{b}_{name}" for b in range(count) for name in HANDED)


__all__ = ["HANDED", "INPUTS", "NEAR_SHARE", "NOT_HOSTED", "OWN", "PREFIX", "REPORT_BLOCKS", "SCHEMES", "WIDTH",
           "WINDOW_GAP_CELLS", "WINDOW_RADIUS_CELLS", "SchwarzHost", "handed_fields", "inputs_of", "starts", "window"]
