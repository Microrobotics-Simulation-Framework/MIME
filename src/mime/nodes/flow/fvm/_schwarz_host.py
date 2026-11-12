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
residual and its flag, and the handed response's definiteness margin. The run's first step reads a preparation made in
the flow's initial state at the body's starting pose and the drive at the first step's end, as the build gives them.
A restart from saved state carries the preparation; a parameter the preparation read, changed at a fork, makes it again.

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
#: end (the initial preparation's), the GMRES's tolerance, the body's extent, the report leaf, and the free box's side
#: in cells where it is written (a structural setting; else ``mime.nodes.flow.fvm._schwarz.box_side`` of the extent).
OWN = ("schwarz_members", "schwarz_ids", "schwarz_origin", "schwarz_starts", "schwarz_drive", "schwarz_tolerance",
       "schwarz_extent", "schwarz_report", "schwarz_box")
#: The hosted near field's constants the flow does not hold under the prefix: its viscosity is the flow's, and its report
#: leaf the flow's own (two blocks, :data:`REPORT_BLOCKS`).
NOT_HOSTED = ("viscosity", "solve_report", "solve_report_rows")
#: The exchange kernel's width at the force points, in cells: the coupling spike's (``sigma = h``); a structural setting.
WIDTH = 1.0
#: The report blocks of the preparation's GMRES: its own, then its products' walled solves'.
REPORT_BLOCKS = 2 * REPORT_SIZE
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
        held[f"{PREFIX}image"] = jnp.zeros(3, dtype=dtype)
        held[f"{PREFIX}report"] = jnp.zeros(REPORT_BLOCKS, dtype=dtype)
        held[f"{PREFIX}residual"] = as_run_array(0.0)
        held[f"{PREFIX}flagged"] = as_run_array(0.0)
        held[f"{PREFIX}indefinite"] = as_run_array(0.0)
        return held


def starts(constants: Mapping[str, Any], count: int) -> list[Array]:
    """Each body's starting pose from the flow's constants (the initial preparation's)."""
    held = jnp.asarray(constants["schwarz_starts"], dtype=run_dtype()).reshape(count, 7)
    return [held[b] for b in range(count)]


def inputs_of(count: int) -> dict[str, tuple[tuple[int, ...], str]]:
    return {f"{PREFIX}{b}_{name}": spec for b in range(count) for name, spec in INPUTS.items()}


def handed_fields(count: int) -> Sequence[str]:
    return tuple(f"{PREFIX}{b}_{name}" for b in range(count) for name in HANDED)


__all__ = ["HANDED", "INPUTS", "NOT_HOSTED", "OWN", "PREFIX", "REPORT_BLOCKS", "WIDTH", "SchwarzHost", "handed_fields",
           "inputs_of", "starts"]
