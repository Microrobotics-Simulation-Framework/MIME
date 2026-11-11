"""The joint near-field node: one node per medium for the bodies whose drag interacts through it, their joint system
solved together each step, each body's coupled resistance and lagged cross load delivered into its own inputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import cached_property
from typing import Any, cast, override

import jax
import numpy as np
from jax import Array
from jaxtyping import Bool, Float
from maddening.core.compliance.metadata import NodeMeta, Reference
from maddening.core.node import BoundaryInputSpec
from maddening.core.params import ParamSpec
from typing_extensions import Annotated, NotRequired, ReadOnly, TypedDict

from mime.core import MimeNode, NodeFields, Symbol, SymbolKind, Units, as_run_array, jit, jnp, node_input, \
    run_constants_given, run_dtype
from mime.core._node import FLOOR_LIMITED, STAGNATED, rechecked
from mime.nodes.hydro.mrs._node import (
    CONFINED_ASYMMETRY_LIMIT,
    HELD,
    HELD_FROM,
    NEAR_WALL,
    WALL,
    MRSDragNode,
    array_tree,
    floating_leaves,
    path_name,
    least_distance,
)
from mime.core.ledger import IDENTITY_ULPS
from mime.core.solves import dense_solve
from mime.nodes.hydro.mrs._quadrature import asymmetry, held_or_made, scaled_least_eigenvalue, unchanged
from mime.nodes.hydro.mrs._solve import PRECISION
from mime.nodes.hydro.mrs._walls import WallCorrection, wall_correction_named
from mime.nodes.hydro.near_field._contract import (
    OVERLAP_RESOLUTION,
    Placed,
    blocks_of,
    booked_work,
    cross_loads,
    grand_resistance,
    grand_resistance_by_gmres,
    joint_matrix,
    lagged_amplification,
    lagged_gain,
    pair_gaps,
    quadratic_form,
)
from mime.nodes.hydro.near_field._gmres import CAP, REPORT_SIZE, RESTART, Stopping, recheck_stopping
from mime.nodes.hydro.near_field._mrs import CROSS_SHARE, REACH, MRSOperator, Quadrature
from mime.registry import Problem

_lax: Any = cast(Any, jax).lax
_tree_util: Any = cast(Any, jax).tree_util
checkpoint: Any = cast(Any, jax).checkpoint

EQUATIONS = (r"\mathbf{R} = \mathbf{W}^{\top} \mathbf{A}^{-1} \mathbf{M}, \quad "
             r"\mathbf{A}_{ab} = \mathbf{S}_{ab} / \mu + \mathbf{G}_{ab}, \quad "
             r"\mathbf{F}_{a} = -\sum_{b \neq a} \mathbf{R}_{ab} \mathbf{V}_{b}, \quad "
             r"g = \rho\left((\mathbf{B}/\Delta t + \mathbf{R}_{d})^{-1} \mathbf{R}_{o}\right)")
_ADVANCE = "mime.nodes.hydro.near_field._node.JointDragNode.advance"
#: The prefix of a member's constants: ``m<i>_`` for the ``i``-th body in ``members``' order.
MEMBER = "m{}_"
#: The leaf holding a member's free part, made once per build in its frame at unit viscosity (``3N x 3N``).
HELD_MATRIX = "held_matrix"
#: The leaf holding the inverse of a member's free part, made with it: its preconditioner block on the iterative route.
HELD_INVERSE = "held_inverse"
#: The joint solve's routes, ``Drag.MRS(near_field=NearField(solve=...))``: ``"dense"``, one dense LU, at every size by
#: default; ``"gmres"``, a declared choice (decision #330, superseding the takeover line of 16,000 unknowns: no size
#: line, the memory refusal the only bound, :data:`MEMORY_COUNTS`). Neither memory nor time places a line. **Memory**:
#: through the node GMRES holds more than the dense route, not less: a run's peak, MIME's sphere of 882 force points in
#: a line, 1.11 to 1.35 times the dense route's at 3,042 to 21,168 unknowns in either dtype, a radius and a twentieth of
#: one apart (12.08 against 10.14 GiB at 21,168 unknowns in float64, a radius apart). Both routes' peaks grow as about
#: two joint matrices, and GMRES holds its members' inverses besides: the bare solve's factor of two (the LU against
#: GMRES on a matrix handed to them) does not carry to the node. **Time**: the bare solve on a quiet CPU in float64 had
#: the dense LU faster or tied at 3,042, 4,056 and 8,112 unknowns and GMRES faster at 5,292 and from 10,584. Through the
#: node (``built.run``, one run a case on a laptop CPU shared with other work, 16 threads, so the ratios are the load's
#: as much as the methods'): a radius apart, GMRES's step was the shorter at every size, 1.3 to 2.8 times in float64 and
#: 1.2 to 4.1 in float32 (7 to 20 iterations); a twentieth of a radius apart, the dense route's was the shorter at four
#: sizes of five in float64 (GMRES's 1.2 to 1.5 times as long; at 5,292 it was 1.5 times the shorter; 52 to 103
#: iterations, none flagged), and in float32 the two were within a factor of two either way, eight spheres there
#: stopping stagnated in 8 of 48 columns, flagged. The dense route is exact to rounding, held to W3a's gates.
ROUTES = ("dense", "gmres")
#: GMRES's relative tolerance on each solve's true residual, by the run's dtype: an accuracy-bearing numerical setting,
#: ``Drag.MRS(near_field=NearField(solve="gmres", tolerance=...))``. Float64's holds the grand resistance within 1e-10
#: of the dense route's on the gates' configurations. Float32's sits above the floor its own arithmetic leaves the residual at, measured on
#: MIME's sphere (pairs a radius to a twentieth of one apart, a triangle, a line of four, a cube of eight): the least
#: true residual float32 reached, at any restart length, was the residual float32 reads at float64's answer rounded to
#: it, 8e-7 for a pair a radius apart to 6.6e-6 for one a twentieth of one apart, the near-contact line; a floor, not a
#: stall. That margin is a pair's: four spheres in a square at the line (10,584 unknowns) converged at 9.7e-6 and
#: 9.9e-6, inside 1e-5 by about 2 %, so a denser cluster there may stop short, and below the line (a pair at 0.03 of a
#: radius) a solve stops stagnated: flagged and warned of, never silent. The FVM's float32 default takes its solve's
#: floor where 1e-5 is out of reach; this one keeps 1e-5 and flags instead.
DEFAULT_TOLERANCE = {"float64": 1e-10, "float32": 1e-5}
#: A near field's run's peak resident memory past the process's fixed part, by route and the run's dtype, as so many of
#: its joint matrices (``itemsize (3 sum N)^2`` in the run's dtype) and so many of its members' held free parts
#: (``8 sum (3N)^2``: made in float64 whatever the run's dtype; the count takes in what else a member holds in
#: proportion, and on the GMRES route its inverse). Reported at build and refused past the memory a run can have
#: (``drag.near-field-memory``), on either route; :data:`MEMORY_MEASURED` says how they were measured.
MEMORY_COUNTS = {("dense", "float64"): (2.01, 6.0), ("dense", "float32"): (2.11, 5.4),
                 ("gmres", "float64"): (2.01, 10.5), ("gmres", "float32"): (2.20, 7.9)}
#: The mode of a forward run, against a derivative's ``"forward"`` and ``"reverse"`` (:meth:`JointDragNode.memory_estimate`).
RUN = "run"
#: A gradient study's derivative's peak past the memory check (``mime.study``'s, before it compiles the derivative), by
#: route and the run's dtype, as so many joint matrices and held free parts as :data:`MEMORY_COUNTS` counts them, for one
#: member: reverse mode; forward mode along one direction, and per direction past the first. A batch of members holds as
#: many times as much. :data:`DERIVATIVE_MEASURED` says how they were measured.
DERIVATIVE_COUNTS: dict[tuple[str, str], dict[str, tuple[float, float]]] = {
    ("dense", "float64"): {"reverse": (10.79, 0.0), "forward": (3.98, 0.0), "direction": (1.34, 0.0)},
    ("dense", "float32"): {"reverse": (10.68, 0.0), "forward": (3.03, 0.0), "direction": (1.34, 0.0)},
    ("gmres", "float64"): {"reverse": (11.98, 0.0), "forward": (3.08, 3.05), "direction": (1.34, 0.0)},
    ("gmres", "float32"): {"reverse": (10.83, 0.0), "forward": (2.72, 0.58), "direction": (1.34, 0.0)},
}
DERIVATIVE_MEASURED = (
    "measured through Study.gradient on GitHub runners (4 vCPU, 16 GB; probe runs 42923077932, 42924924124 and "
    "42925693316): MIME's sphere at 882 force points, 2 to 4 in a line a radius apart (5,292 to 10,584 unknowns), 2 "
    "steps, the derivative of the bodies' relative velocity in the viscosity, as the process's resident peak past the "
    "check (the kernel's high-water mark reset there), non-negative least squares on the joint matrix's bytes and the "
    "held blocks'. Reverse mode holds 10.3 to 12.0 joint matrices (dense float64 10.3, 10.9 and 10.8 at 2, 3 and 4 "
    "spheres), forward mode along one axis 2.1 to 4.6, a second axis 1.34 more (dense float64, three spheres: declared "
    "for the other routes and dtypes, not measured there); two points at once 1.85 and 1.83 times one (reverse, "
    "forward), counted as twice; 8 and 32 steps 1.02 to 1.09 times 2 (the chunked checkpoints keep a step's inputs). "
    "Every case 0.69 to 1.06 times its estimate. The review's seven spheres in reverse mode failed on one buffer of "
    "11.05 joint matrices, as reverse mode's count has it")
#: The memory refusal's margin on an estimate, the largest measured peak over its estimate among the fits' cases rounded
#: up (1.13, a forward run's on the laptop, :data:`MEMORY_MEASURED`; a derivative's at most 1.06, a runner's forward run
#: 1.04), so that the estimate it compares is at least what each measured case held; and its headroom, the share of the
#: memory a run can have that it keeps free for what the counts do not hold (another process, the operating system's
#: rounding of what is available). Measured on runners: every case the line admitted peaked under it
#: (:data:`DERIVATIVE_MEASURED`'s and the gate's probe, ``tests/near_field/predictions_drag050_fixround.json``).
MEMORY_MARGIN = 1.15
MEMORY_HEADROOM = 0.1
MEMORY_MEASURED = ("measured on a laptop CPU with 16 threads as the process's peak resident memory over a build and two "
                   "runs through built.run, less a two-sphere run's peak (396 unknowns: 0.65 to 0.73 GiB, the "
                   "process's fixed part, its imports and compiles, which the estimate leaves out), least squares over "
                   "MIME's sphere at 882 force points, 2 to 8 spheres in a line a radius and a twentieth of one apart "
                   "(5,292 to 21,168 unknowns), every case 0.92 to 1.13 times its estimate; with 338 force points "
                   "(3,042 unknowns) a run held about twice its estimate, 0.3 GiB more, so the estimate is least sure "
                   "for a small cluster; it is measured, not a bound, and a GPU's is not measured")
#: The stop's causes, as the state's ``stopped_why`` holds them.
WHY_MEMBER, WHY_CLUSTER, WHY_GAIN, WHY_OVERLAP = 1.0, 2.0, 3.0, 4.0
#: The validity line's lower edge, a gap between two bodies' surfaces over the smaller body's equivalent radius (its
#: surface's area over 4 pi, rooted: a sphere's radius): below about a twentieth of a radius the MRS's own error is not
#: monotone in the spacing (S3's two spheres), lubrication's territory. A step with a pair closer raises
#: ``near_contact``.
CONTACT_GAP = 0.05


def member_prefix(index: int) -> str:
    return MEMBER.format(index)


class JointDragState(TypedDict, extra_items=ReadOnly[Float[Array, "..."]]):
    """``resistance_<i>`` and ``background_load_<i>`` for each member (its delivery); the cluster's symmetrised grand
    resistance and the step's starting twists, which the next step judges this one's books by; the step's judgement
    (each member's scaled least eigenvalue and the cluster's, the asymmetry, the gain, the lagged step's amplification
    and its relaxation in steps, the dissipation rate at the starting twists, the step before's books and their lagged
    residual, the least gap between two bodies and which pair) with the flags it raises; with a wall, each body's least
    distance from it; whether the joint solve stopped other than converging, and on the iterative route its report;
    and the stop's record (the step, why, which member and which other, the value, the poses)."""

    resistance: ReadOnly[Float[Array, "k k"]]  # noqa: F722 - a jaxtyping shape
    least_eigenvalue: ReadOnly[Annotated[Float[Array, "n"], Units("1")]]  # noqa: F722 - a jaxtyping shape
    cluster_least: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    asymmetry: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    gain: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    amplification: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    relaxation_steps: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    dissipation: ReadOnly[Annotated[Float[Array, ""], Units("W")]]  # noqa: F722 - a jaxtyping shape
    cluster_dissipation: ReadOnly[Annotated[Float[Array, ""], Units("J")]]  # noqa: F722 - a jaxtyping shape
    lag_residual: ReadOnly[Annotated[Float[Array, ""], Units("J")]]  # noqa: F722 - a jaxtyping shape
    start_twists: ReadOnly[Annotated[Float[Array, "n 6"], Units("m/s; rad/s")]]  # noqa: F722 - a jaxtyping shape
    least_gap: ReadOnly[Annotated[Float[Array, ""], Units("m")]]  # noqa: F722 - a jaxtyping shape
    gap_pair: ReadOnly[Annotated[Float[Array, "2"], Units("1")]]  # noqa: F722 - a jaxtyping shape
    overlap: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    near_contact: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    wall_accepted: ReadOnly[Annotated[Bool[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    indefinite: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    gain_exceeded: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    near_wall: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    asymmetric: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    cluster_gift: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    cluster_gift_steps: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    cross_cells_short: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    solve_flagged: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    #: The iterative route's: the step's primal solve report (``mime.nodes.flow.fvm.REPORT_FIELDS``), and its running
    #: sum.
    solve_report: NotRequired[ReadOnly[Annotated[Float[Array, "6"], Units("1")]]]  # noqa: F722 - a jaxtyping shape
    solve_report_sum: NotRequired[ReadOnly[Annotated[Float[Array, "6"], Units("1")]]]  # noqa: F722 - a jaxtyping shape
    wall_gap: NotRequired[ReadOnly[Annotated[Float[Array, "n"], Units("m")]]]  # noqa: F722 - a jaxtyping shape
    wall_gap_spacings: NotRequired[ReadOnly[Annotated[Float[Array, "n"], Units("1")]]]  # noqa: F722 - a jaxtyping shape
    wall_cells_short: NotRequired[ReadOnly[Annotated[Float[Array, ""], Units("1")]]]  # noqa: F722 - a jaxtyping shape
    steps: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    stopped_at: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    stopped_why: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    stopped_member: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    stopped_other: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    stopped_value: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    stopped_poses: ReadOnly[Annotated[Float[Array, "n 7"], Units("m; 1")]]  # noqa: F722 - a jaxtyping shape


class JointDragInputs(TypedDict, extra_items=ReadOnly[Float[Array, "..."]]):
    """``pose_<i>``, ``twist_<i>`` and ``mass_matrix_<i>`` for each member, from its body's state."""


class JointDragNode(MimeNode):
    """The drag of a medium's interacting MRS bodies, solved as one joint system each step: one joint node per medium,
    the bodies' steps lagged, with a gain check.

    **Its members** are the bodies of ``members``, each with the constructor arguments its per-body
    ``MRSDragNode`` takes, written ``m<i>_<argument>``; the node builds each one's ``MRSDragNode`` (never added to the
    graph) and takes its discretisation from it (:class:`MRSOperator`). The wall's constants are the medium's, once.

    **Each step**: the joint matrix at the members' poses (each one's free part, made once per build in its frame and
    turned to its pose, with its wall's self-image as the per-body node's; each pair's cross-kernel within its reach
    and the wall's image between them), the grand resistance ``R`` with every body's six rigid motions as right-hand
    sides, its symmetric part taken; then each body ``a`` is delivered its own ``R_aa`` into ``resistance`` and its lagged
    cross load ``-sum_b R_ab V_b`` at the step's starting twists into ``background_load``. The cluster's ``R`` and the
    starting twists stay in the state, so that each body's block and load are derived from it and the next step judges
    this one's books.

    **The solve, by its route** (``solve``, :data:`ROUTES`): ``"dense"``, the default at every size, one dense LU of
    the joint matrix, whose factors serve every right-hand side; ``"gmres"``, declared, GMRES on every column at once
    (``mime.nodes.hydro.near_field._gmres``), each body's held free part's inverse its preconditioner block
    (made once per build in its frame, as the free part is, and turned to its pose), each solve's tolerance
    (``tolerance``, :data:`DEFAULT_TOLERANCE` by the run's dtype) relative to its own right-hand side, its stop on the
    true residual, never silent. Its report, the FVM's (``REPORT_FIELDS``): the primal's each step a state field
    (``solve_report``, with its running sum) and a flag (``solve_flagged``: at its cap, stagnated, or a residual not
    finite; the dense route never raises it); the tangent solves' as the tangents of those fields; the transpose solves'
    as the gradient of the ``solve_report`` leaf, zeros the step reads (summed over the run, or a row a step with
    ``solve_report_rows``). **A recheck** (``rechecks``: a gradient study's member whose solve stopped short) re-solves
    with GMRES's cap lifted and its stall restarts raised (:func:`~mime.nodes.hydro.near_field._gmres.recheck_stopping`),
    as the FVM's CG; what is left short then is GMRES's floor, which no cap lifts.

    **Judged before the bodies take it** (each held out of every derivative): the gap between each pair of bodies,
    estimated from their force points and their surfaces' normals there (:func:`pair_gaps`): where one body's points
    lie behind another's surface by more than a quarter of a cell (``OVERLAP_RESOLUTION``), the bodies overlap, their
    joint system describes no fluid between them, and the run stops; where a pair is closer than ``CONTACT_GAP`` of the
    smaller body's equivalent radius, the validity line's lower edge, the step raises ``near_contact``. Then each body's
    ``R_aa`` and the cluster's ``R`` by their least eigenvalue, each row and column scaled by its diagonal, and the
    lagged step's gain ``rho((M/dt + R_d)^-1 R_off)`` from each body's mass matrix as it publishes it, the per-step
    fixed point's contraction (the record's check: sufficient for the lagged step's stability, not necessary, so that
    it refuses some stable inertial clusters, three spheres near contact between about tau/dt 0.05 and 0.11). Where
    two bodies overlap, a body's or the cluster's resistance is not positive definite, or the gain is not below one,
    every delivery is replaced by NaN and the run stops there (``stop_problem`` says which, when and why). Reported:
    the scaled least eigenvalues (the margin), the gain, the lagged step's amplification ``rho((M/dt + R_d)^-1 (M/dt -
    R_off))`` and its relaxation in steps ``1/(1 - rho)``, the cluster's dissipation rate ``V^T R V`` at the starting
    twists, the least gap and its pair, the asymmetry, each body's distance from the wall, and whether a cross-kernel or
    a wall image ran short of its budget.

    **The cluster's books, judged a step late**: once the bodies have taken a step, the next judges the near field's
    own work on them over it, from what it delivered and the twists they reached, as the energy ledger books it on each
    body (:func:`booked_work`): the fluid may not give the cluster energy past the ledger's floor, as the ledger's route
    1 judges a step that claims no identity (``cluster_gift``, counted in ``cluster_gift_steps``, which the run's account
    reads into the near field's law, ``CHECK_CLUSTER``); the fluid's dissipation and its lagged residual are reported.
    **Its reach**: an energy check sees an error that makes the fluid a source, and none that leaves it dissipating.
    Measured on two spheres, one dragged: cross loads delivered three times over make it one to a gap of a radius along
    the line of centres and a third of one across it, twice over to a third of a radius along it; those at wider gaps,
    any delivered at half their value or with their sign flipped, and any error in ``R`` that leaves it definite, leave
    it dissipating and are the force gates' to catch (the delivery's own check against ``R``, the dense assembly, the
    two-sphere series and Rotne and Prager's mobility). The run's last step is judged by no step after it."""

    meta = NodeMeta(
        algorithm_id="MIME-NODE-127",
        description="The joint resistance of a medium's interacting rigid bodies in Stokes flow by the method of "
                    "regularised Stokeslets, each body's coupled resistance and lagged cross load delivered.",
        governing_equations=EQUATIONS,
        assumptions=("Stokes flow: quasi-steady, the limit of vanishing Reynolds number",
                     "Each body's surface by the separated quadrature of the MRS rung (MRSDragNode)",
                     "Bodies coupled through the fluid by each source's regularised Stokeslet, integrated over its "
                     "cells where the pair is near and taken at its force points beyond the reach, and the wall's image "
                     "between them at the force points",
                     "The bodies' steps lagged: each takes its own coupled resistance implicitly and the others' "
                     "loads from the step before",
                     "A medium at rest, so a body held still in it feels no load"),
        references=(Reference("Cortez2005", "the regularised Stokeslet and its method of regularised Stokeslets"),
                    Reference("Smith2018", "the nearest-neighbour discretisation, force points apart from the "
                                           "quadrature")),
    )
    rechecks = True
    floor_stops = True
    symbols = (
        Symbol(r"\mathbf{R}", SymbolKind.STATE, "resistance", "N s/m; N s; N m s", _ADVANCE),
        Symbol(r"\mu", SymbolKind.PARAMETER, "viscosity", "Pa s", _ADVANCE),
        Symbol(r"g", SymbolKind.STATE, "gain", "1", _ADVANCE),
    )

    def __init__(self, name: str, timestep: float, *, members: Sequence[str], viscosity: float, wall_kind: str,
                 wall_kind_version: str, reach: float = REACH, cross_share: float = CROSS_SHARE,
                 solve: str = "dense", tolerance: float | None = None, restart: int = RESTART,
                 cap: int = CAP, solve_report_rows: int = 0, solve_report: object = None, **constants: object) -> None:
        del solve_report  # a node made again from its own arguments: its report leaf is zeros, made below
        held = tuple(str(member) for member in members)
        if not held or len(set(held)) != len(held):
            raise ValueError(f"a joint node couples bodies, each once (the build gives it two or more; one, its tests "
                             f"against the per-body node); it was given {held!r}")
        names = {f"{WALL}{name}" for name in wall_correction_named(wall_kind, wall_kind_version).parameters}
        walls = {key: value for key, value in constants.items() if key in names}
        mine = {i: {key.removeprefix(member_prefix(i)): value for key, value in constants.items()
                    if key.startswith(member_prefix(i))} for i in range(len(held))}
        unknown = set(constants) - set(walls) - {f"{member_prefix(i)}{key}" for i, args in mine.items()
                                                 for key in args}
        if unknown:
            raise ValueError(f"a joint node takes each member's arguments as m<i>_<argument> and the wall's as "
                             f"wall_<name>; it was given {sorted(unknown)}")
        self._members = [MRSDragNode(f"{name}/{body}", timestep, viscosity=viscosity, wall_kind=wall_kind,
                                     wall_kind_version=wall_kind_version, **walls, **cast(dict[str, Any], mine[i]))
                         for i, body in enumerate(held)]
        self._operators = [MRSOperator(member, float(cross_share)) for member in self._members]
        if cast(object, solve) not in ROUTES:  # a script is not checked
            raise ValueError(f"the joint solve's route is one of {ROUTES}, not {solve!r}")
        settings: dict[str, Any] = {} if solve == "dense" else {"solve": solve}
        self._iterative = solve == "gmres"
        if self._iterative:
            if tolerance is None:
                tolerance = DEFAULT_TOLERANCE[str(jnp.dtype(run_dtype()))]
            if not float(tolerance) > 0.0 or int(restart) < 1 or int(cap) < 1 or int(solve_report_rows) < 0:
                raise ValueError(f"the joint solve's GMRES takes a tolerance above zero, a restart and a cap of one "
                                 f"iteration or more and solve_report_rows at or above zero, not {tolerance!r}, "
                                 f"{restart!r}, {cap!r} and {solve_report_rows!r}")
            settings.update(tolerance=float(tolerance), restart=int(restart), cap=int(cap),
                            solve_report_rows=int(solve_report_rows),
                            solve_report=np.zeros((max(int(solve_report_rows), 1), REPORT_SIZE)))
        super().__init__(name, timestep, members=held, viscosity=viscosity, wall_kind=wall_kind,
                         wall_kind_version=wall_kind_version, reach=reach, cross_share=cross_share, **settings,
                         **constants)

    # -- the members --------------------------------------------------------------------------------------------------

    @property
    def members(self) -> tuple[str, ...]:
        return cast(tuple[str, ...], tuple(self.constants()["members"]))

    @property
    def member_nodes(self) -> tuple[MRSDragNode, ...]:
        """Each member's per-body node, as the rung builds it: never in the graph."""
        return tuple(self._members)

    @property
    def unknowns(self) -> int:
        """The joint system's unknowns: three a force point, over every member."""
        return sum(3 * op.count for op in self._operators)

    @property
    def iterative(self) -> bool:
        """Whether the joint system is solved by GMRES (its route, declared when built)."""
        return self._iterative

    @property
    def route(self) -> str:
        """The joint solve's route, ``"dense"`` or ``"gmres"`` (decided when built)."""
        return "gmres" if self._iterative else "dense"

    @override
    def recheck_route(self, cause: str, dtype: str) -> str:
        """A floor: float64 in a float32 run, in float64 a tolerance at or above it; a stall: GMRES's restart length is
        not a setting; and either way the dense route, an LU with no tolerance to stop short of."""
        dense = ("take the dense route, Drag.MRS(near_field=NearField(solve='dense')), the default, an LU with no "
                 "tolerance to stop short of")
        if cause == FLOOR_LIMITED:
            return (f"run in float64, or {dense}" if dtype == "float32" else
                    f"write a tolerance at or above the floor, Drag.MRS(near_field=NearField(solve='gmres', "
                    f"tolerance=...)), or {dense}")
        return dense if cause == STAGNATED else ""

    def memory_estimate(self, mode: str = RUN, directions: int = 1, members: int = 1) -> float:
        """The bytes the node's run holds at its peak past the process's fixed part, in the run's dtype: a forward run
        (``mode`` :data:`RUN`, :data:`MEMORY_COUNTS`), or a gradient study's derivative of ``members`` batched members,
        in ``"forward"`` mode along ``directions`` tangents or in ``"reverse"`` mode (:data:`DERIVATIVE_COUNTS`)."""
        dtype = str(jnp.dtype(run_dtype()))
        if mode == RUN:
            matrices, held = MEMORY_COUNTS[(self.route, dtype)]
        else:
            counts = DERIVATIVE_COUNTS[(self.route, dtype)]
            matrices, held = counts[mode]
            if mode == "forward":
                more, more_held = counts["direction"]
                matrices, held = matrices + (directions - 1) * more, held + (directions - 1) * more_held
            matrices, held = members * matrices, members * held
        return (matrices * _itemsize() * self.unknowns ** 2
                + held * 8 * sum((3 * op.count) ** 2 for op in self._operators))

    def memory_problem(self, mode: str = RUN, directions: int = 1, members: int = 1) -> Problem | None:
        """A joint system whose run would not fit in the memory a run can have (a GPU's limit; on a CPU the smaller of
        the host's available memory and its control group's limit), refused before its first step on either route:
        its estimate (:meth:`memory_estimate`) times :data:`MEMORY_MARGIN` against that memory less
        :data:`MEMORY_HEADROOM` of it. A forward run is checked at build; a gradient study's derivative before the
        study compiles it (:meth:`derivative_problem`)."""
        where, available = device_memory()
        needed = MEMORY_MARGIN * self.memory_estimate(mode, directions, members)
        if available is None or needed <= (1.0 - MEMORY_HEADROOM) * available:
            return None
        what = "run" if mode == RUN else (
            f"derivative in {mode} mode{f' along {directions} directions' if mode == 'forward' and directions > 1 else ''}"
            f"{f' of {members} members at once' if members > 1 else ''}")
        return Problem(
            "drag.near-field-memory",
            f"the medium's near field couples {len(self.members)} bodies ({', '.join(self.members)}) into a joint "
            f"system of {self.unknowns} unknowns, whose {what} on the {self.route} route needs about "
            f"{needed / 2 ** 30:.3g} GiB (its joint matrix and its members' held blocks times the counts a CPU run was "
            f"measured to hold, times {MEMORY_MARGIN:g}, the measurements' largest miss, as the drag's validity line "
            f"says), more than {where}, {available / 2 ** 30:.3g} GiB, less the {MEMORY_HEADROOM:.0%} of it the check "
            f"keeps free",
            "write a coarser spacing on the bodies' drag, Drag.MRS(spacing=...), fewer bodies in the medium"
            + (", the derivative in forward mode along few axes, study.gradient(..., mode='forward'), which holds "
               "fewer matrices than reverse mode, or fewer of the study's points at once" if mode != RUN else "")
            + ", or a machine with more memory")

    def derivative_problem(self, mode: str, directions: int, members: int, steps: int) -> Problem | None:
        """A gradient study's derivative of ``members`` members at once in ``mode`` (``"forward"`` along
        ``directions`` tangents, or ``"reverse"``) over ``steps`` steps that would not fit (:meth:`memory_problem`),
        refused before the study compiles it. The steps do not enter: measured, reverse mode's chunked checkpoints
        hold a step's inputs, not its matrices (:data:`DERIVATIVE_MEASURED`)."""
        del steps
        return self.memory_problem(mode, directions, members)

    def _wall(self) -> WallCorrection:
        return wall_correction_named(self.constants()["wall_kind"], self.constants()["wall_kind_version"])

    def _member_leaf_names(self, index: int) -> tuple[str, ...]:
        names = self._members[index].body_leaf_names()
        return (*names, "force_spacing") if "force_spacing" in self._members[index].constants() else names

    @override
    def param_specs(self) -> dict[str, ParamSpec]:
        found: dict[str, ParamSpec] = {**super().param_specs(), "viscosity": ParamSpec(units="Pa s"),
                                       "reach": ParamSpec(units="1", trainable=False,
                                                          description="a numerical setting: the cross-kernel's "
                                                                      "reach in each source cell's own size"),
                                       "cross_share": ParamSpec(units="1", trainable=False,
                                                                description="held as built: the cross-kernel's "
                                                                            "budget, a share of a source's "
                                                                            "quadrature points, fixes its shape")}
        made = ("made once per build from the member's kind's numbers, its part's placement and its blob, in its frame "
                "at unit viscosity: no quantity of its own")
        for i, member in enumerate(self._members):
            prefix = member_prefix(i)
            specs = member.param_specs()
            for key, spec in specs.items():
                if key in ("viscosity", HELD) or key.startswith(HELD_FROM) or key in self._wall_keys():
                    continue
                found[f"{prefix}{key}"] = spec
            found[f"{prefix}{HELD_MATRIX}"] = ParamSpec(units="1", trainable=False, description=made)
            if self.iterative:
                found[f"{prefix}{HELD_INVERSE}"] = ParamSpec(
                    units="1", trainable=False, description=f"made once per build, the inverse of {prefix}"
                                                            f"{HELD_MATRIX}: the member's preconditioner block, which "
                                                            f"changes the solve's cost and never its answer")
            found.update({f"{prefix}{HELD_FROM}{key}": ParamSpec(
                units=specs[key].units, trainable=False,
                description=f"the value of {prefix}{key} that {prefix}{HELD_MATRIX} was made at, which the step "
                            f"compares {prefix}{key} with: no quantity of its own") for key in member.body_leaf_names()})
        first = self._members[0].param_specs()
        found.update({key: spec for key, spec in first.items() if key in self._wall_keys()})
        if self.iterative:
            found["tolerance"] = ParamSpec(trainable=False, units="1",
                                           description="the joint solve's GMRES: its relative tolerance on each "
                                                       "solve's true residual, an accuracy-bearing setting")
            found["solve_report"] = ParamSpec(trainable=False, units="1",
                                              description="zeros, never moved: the derivative solves' reports come out "
                                                          "as its gradient")
        return found

    def _wall_keys(self) -> set[str]:
        """The node's constants and leaves that are the wall's (``wall_<parameter>``, and a held table's arrays)."""
        wall = self._wall()
        return {f"{WALL}{name}" for name in wall.parameters} | set(self._wall_arguments())

    def _wall_trees(self) -> list[str]:
        constants = self.constants()
        return [f"{WALL}{name}" for name in sorted(self._wall().held) if array_tree(constants[f"{WALL}{name}"])]

    def _wall_arguments(self) -> dict[str, Any]:
        constants = self.constants()
        return {key: leaf for name in self._wall_trees()
                for key, leaf in floating_leaves(name, constants[name]).items()}

    @override
    def params_pytree(self) -> dict[str, Any]:
        inherited: Any = super()
        own = cast(dict[str, Any], inherited.params_pytree())
        held = {f"{member_prefix(i)}{HELD_MATRIX}": as_run_array(matrix) for i, matrix in enumerate(self._held)}
        if self.iterative:
            held.update({f"{member_prefix(i)}{HELD_INVERSE}": as_run_array(inverse)
                         for i, inverse in enumerate(self._held_inverses)})
        made_at = {f"{member_prefix(i)}{HELD_FROM}{key}": own[f"{member_prefix(i)}{key}"]
                   for i, member in enumerate(self._members) for key in member.body_leaf_names()}
        return {**own, **self._wall_arguments(), **held, **made_at}

    @override
    def _step_constants(self, params: Mapping[str, Any] | None) -> dict[str, Any]:
        held = super()._step_constants(params)
        if params is None:
            return held
        for name in self._wall_trees():
            paths, tree = _tree_util.tree_flatten_with_path(held[name])
            held[name] = _tree_util.tree_unflatten(tree, [params.get(f"{name}{path_name(path)}", leaf)
                                                          for path, leaf in paths])
        arguments = set(self._wall_arguments())
        return {key: value for key, value in held.items() if key not in arguments}

    @cached_property
    def _held(self) -> tuple[np.ndarray, ...]:
        """Each member's free part, ``3N x 3N`` at unit viscosity in its frame, made once from the node's constants in
        float64 whatever the run's precision, never traced."""
        found: list[np.ndarray] = []
        with cast(Any, jax).enable_x64(True), cast(Any, jax).ensure_compile_time_eval():
            for member in self._members:
                theta = {key: jnp.asarray(value, dtype=jnp.float64) for key, value in member.built_theta().items()}
                found.append(np.asarray(member.self_interaction(theta), dtype=np.float64))
        return tuple(found)

    @cached_property
    def _held_inverses(self) -> tuple[np.ndarray, ...]:
        """Each member's held free part's inverse, in float64 whatever the run's precision, never traced."""
        return tuple(np.linalg.inv(matrix) for matrix in self._held)

    @cached_property
    def _made_compiled(self) -> tuple[Any, ...]:
        return tuple(jit(member.self_interaction) for member in self._members)

    def _inverse_made(self, index: int) -> Any:
        member = self._members[index]

        def made(theta: Mapping[str, Any]) -> Array:
            free = member.self_interaction(theta)
            return dense_solve(free, jnp.eye(free.shape[-1], dtype=free.dtype))

        return made

    def _theta(self, index: int, p: Mapping[str, Any]) -> dict[str, Any]:
        """A member's leaves its free part is made from, by the member's own names."""
        return {key: p[f"{member_prefix(index)}{key}"] for key in self._members[index].body_leaf_names()}

    def _made_at(self, index: int, p: Mapping[str, Any]) -> dict[str, Any]:
        """The leaves a member's held free part in ``p`` was made at (``m<i>_held_from.<leaf>``), by the member's own
        names: values of the step, never constants of its program (``held_or_made``)."""
        prefix = member_prefix(index)
        return {key: p[f"{prefix}{HELD_FROM}{key}"] for key in self._members[index].body_leaf_names()}

    def _held_or_made(self, index: int, p: Mapping[str, Any]) -> Array:
        member = self._members[index]
        return held_or_made(member.self_interaction, self._made_at(index, p))(
            self._theta(index, p), p[f"{member_prefix(index)}{HELD_MATRIX}"])

    def _held_value(self, index: int, p: Mapping[str, Any]) -> Array:
        """A member's free part at the step's leaves: as its chunk made it, where it did (its run-constant,
        :meth:`run_constants`); else the held one where its leaves are those it was made at, made afresh where one has
        moved (``held_or_made``), or where the step was given no held value."""
        key = f"{member_prefix(index)}{HELD_MATRIX}"
        if key not in p:
            return self._members[index].self_interaction(self._theta(index, p))
        if run_constants_given(p):
            return p[key]
        return self._held_or_made(index, p)

    def _held_inverse_or_made(self, index: int, p: Mapping[str, Any]) -> Array:
        """A member's preconditioner block's inverse at the step's leaves, as :meth:`_held_or_made` chooses the free
        part, held out of every derivative: a preconditioner changes the solve's cost, never its answer, and the solve's
        derivative passes through its operator alone."""
        stop = _lax.stop_gradient
        return stop(held_or_made(self._inverse_made(index), stop(self._made_at(index, p)))(
            stop(self._theta(index, p)), stop(p[f"{member_prefix(index)}{HELD_INVERSE}"])))

    def _held_inverse_value(self, index: int, p: Mapping[str, Any]) -> Array:
        """A member's held free part's inverse at the step's leaves, chosen as :meth:`_held_value` chooses the free
        part."""
        key = f"{member_prefix(index)}{HELD_INVERSE}"
        if key not in p:
            return _lax.stop_gradient(self._inverse_made(index)(_lax.stop_gradient(self._theta(index, p))))
        if run_constants_given(p):
            return _lax.stop_gradient(p[key])
        return self._held_inverse_or_made(index, p)

    @override
    def run_constants(self, params: Mapping[str, Any]) -> dict[str, Array]:
        """Each member's held free part at the run's leaves (:meth:`_held_value`'s choice), and on the iterative route
        its inverse, made once per chunk."""
        found = {f"{member_prefix(i)}{HELD_MATRIX}": self._held_or_made(i, params)
                 for i in range(len(self._members)) if f"{member_prefix(i)}{HELD_MATRIX}" in params}
        found.update({f"{member_prefix(i)}{HELD_INVERSE}": self._held_inverse_or_made(i, params)
                      for i in range(len(self._members)) if f"{member_prefix(i)}{HELD_INVERSE}" in params})
        return found

    @override
    def made_again(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Each member's held free part (and on the iterative route its inverse) made from ``params``' leaves, once,
        where they are not those it was made at, with the leaves it was made at: what the save's resolution rerun gives
        the step, which keeps it there."""
        found: dict[str, Any] = {}
        for i, member in enumerate(self._members):
            prefix = member_prefix(i)
            key = f"{prefix}{HELD_MATRIX}"
            if key not in params:
                continue
            theta = self._theta(i, params)
            if bool(unchanged(theta, member.built_theta())):
                continue
            made = self._made_compiled[i](theta)
            found[key] = made.astype(jnp.asarray(params[key]).dtype)
            inverse = f"{prefix}{HELD_INVERSE}"
            if inverse in params:
                found[inverse] = dense_solve(made, jnp.eye(made.shape[-1], dtype=made.dtype)).astype(
                    jnp.asarray(params[inverse]).dtype)
            found.update({f"{prefix}{HELD_FROM}{name}": value for name, value in theta.items()})
        return found

    # -- state and inputs ---------------------------------------------------------------------------------------------

    def _gapped(self) -> bool:
        """Whether the wall reports each body's distance from it against the members' force spacings."""
        return self._wall().distance is not None and all("force_spacing" in m.constants() for m in self._members)

    def _wall_budgeted(self) -> bool:
        return any(m.wall_quadrature is not None for m in self._members) and self._gapped()

    @override
    def initial_state(self) -> NodeFields:
        count = len(self._members)
        zero = as_run_array(jnp.zeros(()))
        state: dict[str, Array] = {}
        for i in range(count):
            state[f"resistance_{i}"] = as_run_array(jnp.zeros((6, 6)))
            state[f"background_load_{i}"] = as_run_array(jnp.zeros(6))
        state.update(resistance=as_run_array(jnp.zeros((6 * count, 6 * count))),
                     least_eigenvalue=as_run_array(jnp.zeros(count)), cluster_least=zero, asymmetry=zero, gain=zero,
                     amplification=zero, relaxation_steps=zero, dissipation=zero, cluster_dissipation=zero,
                     lag_residual=zero, start_twists=as_run_array(jnp.zeros((count, 6))), least_gap=zero,
                     gap_pair=as_run_array(jnp.zeros(2)), wall_accepted=jnp.asarray(True), indefinite=zero,
                     gain_exceeded=zero, near_wall=zero, asymmetric=zero, cluster_gift=zero, cluster_gift_steps=zero,
                     cross_cells_short=zero, overlap=zero, near_contact=zero, solve_flagged=zero,
                     steps=zero, stopped_at=zero, stopped_why=zero, stopped_member=zero, stopped_other=zero,
                     stopped_value=zero, stopped_poses=as_run_array(jnp.zeros((count, 7))))
        if self.iterative:
            state.update(solve_report=as_run_array(jnp.zeros(REPORT_SIZE)),
                         solve_report_sum=as_run_array(jnp.zeros(REPORT_SIZE)))
        if self._gapped():
            state.update(wall_gap=as_run_array(jnp.zeros(count)), wall_gap_spacings=as_run_array(jnp.zeros(count)))
        if self._wall_budgeted():
            state["wall_cells_short"] = zero
        return state

    @override
    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        found: dict[str, BoundaryInputSpec] = {}
        for i in range(len(self._members)):
            found[f"pose_{i}"] = node_input("m; 1", (7,))
            found[f"twist_{i}"] = node_input("m/s; rad/s", (6,))
            found[f"mass_matrix_{i}"] = node_input("kg; kg m; kg m^2", (6, 6))
        return found

    # -- the step ------------------------------------------------------------------------------------------------------

    def _leaves(self, index: int, p: Mapping[str, Any]) -> dict[str, Any]:
        return {key: p[f"{member_prefix(index)}{key}"] for key in self._member_leaf_names(index)}

    def report_row(self, p: Mapping[str, Any], steps: Any = None) -> Array:
        """The ``solve_report`` row the step's solve writes its derivative solves' reports through: the one row, or with
        ``solve_report_rows`` the step's own (``steps`` the steps taken before it)."""
        rows = int(self.constants()["solve_report_rows"])
        if rows == 0 or steps is None:
            return jnp.asarray(p["solve_report"])[0]
        return jnp.asarray(p["solve_report"])[jnp.clip(jnp.asarray(steps).astype(jnp.int32), 0, rows - 1)]

    def solved(self, p: Mapping[str, Any], poses: Array, steps: Any = None) -> tuple[Any, ...]:
        """The cluster's grand resistance at ``poses`` (``(K, 7)``) from the constants ``p``, not symmetrised; whether
        every wall read was accepted; whether any wall image or cross-kernel ran short; each body's least distance from
        the wall (zero without a wall's distance); the gaps between the bodies, from the same placements
        (:meth:`body_gaps`); and on the iterative route the joint solve's report (``None`` on the dense route), its
        derivative solves' through the ``solve_report`` row of the step (``steps`` taken before it). Checkpointed:
        reverse mode keeps its inputs and solves again."""
        wall = self._wall()
        parameters = {name: p[f"{WALL}{name}"] for name in wall.parameters}
        count = len(self._members)
        leaves = [self._leaves(i, p) for i in range(count)]
        held = [self._held_value(i, p) for i in range(count)]
        operators, distance = self._operators, wall.distance
        iterative = self.iterative
        inverses = [self._held_inverse_value(i, p) for i in range(count)] if iterative else None
        tap = self.report_row(p, steps) if iterative else None
        constants = self.constants()
        restart, cap = (int(constants["restart"]), int(constants["cap"])) if iterative else (0, 0)
        again = rechecked(p)

        def solve(held: Sequence[Array], leaves: Sequence[Mapping[str, Any]], poses: Array, viscosity: Array,
                  parameters: Mapping[str, Any], reach: Array, inverses: Sequence[Array] | None, tap: Array | None,
                  tolerance: Array | None) -> tuple[Any, ...]:
            joint = joint_matrix(operators, leaves, held, poses, viscosity, wall, parameters, reach)
            gaps = jnp.zeros(len(operators), dtype=poses.dtype) if distance is None else jnp.stack(
                [least_distance(distance(cast(Quadrature, placed.own).points, parameters)) for placed in joint.placed])
            if inverses is None or tap is None:
                grand, report = grand_resistance(joint), None
            else:
                turns, held_viscosity = _lax.stop_gradient(poses), _lax.stop_gradient(viscosity)
                blocks = [op.preconditioner(inverse, turns[i], held_viscosity)
                          for i, (op, inverse) in enumerate(zip(operators, inverses, strict=True))]
                transposed = [op.preconditioner(inverse, turns[i], held_viscosity, transpose=True)
                              for i, (op, inverse) in enumerate(zip(operators, inverses, strict=True))]
                stopping = Stopping(tolerance, restart=restart, cap=cap)
                grand, report = grand_resistance_by_gmres(joint, blocks, transposed,
                                                          recheck_stopping(stopping) if again else stopping, tap)
            return (grand, joint.accepted, jnp.any(joint.wall_short), jnp.any(joint.cross_short), gaps,
                    *self.body_gaps(joint.placed), report)

        return checkpoint(solve)(held, leaves, poses, p["viscosity"], parameters, p["reach"], inverses, tap,
                                 p["tolerance"] if iterative else None)

    @staticmethod
    def body_gaps(placed: Sequence[Placed]) -> tuple[Array, Array, Array]:
        """The bodies' gaps at their placements (:func:`pair_gaps`: each ordered pair's least height of one body's
        points above the other's surface, m, and whether one lies behind it beyond the estimate's resolution), and each
        body's equivalent radius, its surface's area over four pi rooted (m). A judgement, held out of every
        derivative."""
        gaps, beyond = pair_gaps(placed)
        radii = jnp.stack([jnp.sqrt(jnp.sum(body.area) / (4.0 * jnp.pi)) for body in placed])
        return _lax.stop_gradient(gaps), _lax.stop_gradient(beyond), _lax.stop_gradient(radii)

    def judged_books(self, state: NodeFields, twists: Array, dt: float) -> tuple[Array, Array, Array]:
        """The step before's books, judged now that the bodies have taken it: the near field's own work on the cluster
        over that step as the energy ledger pairs it (:func:`booked_work`, from the blocks and loads the node delivered
        then, ``state``, and the twists the bodies reached, ``twists``), the fluid's dissipation, minus that work; its
        lagged residual, ``dt V'^T R_off (V' - V)``, by which the books of a lagged step differ from ``dt V'^T R V'`` at
        one velocity; and whether the fluid gave energy past the ledger's floor (``IDENTITY_ULPS`` units of eps on the
        books' scale), as the ledger's route 1 judges a step that claims no identity. Judged only where the step before
        was stepped and not stopped: the first step has no books before it, and a stopped step's deliveries are NaN."""
        count = len(self._members)
        dtype = state["gain"].dtype
        held = cast(Mapping[str, Array], _lax.stop_gradient(dict(state)))
        own = jnp.stack([held[f"resistance_{i}"] for i in range(count)])
        loads = jnp.stack([held[f"background_load_{i}"] for i in range(count)])
        work, scale = booked_work(own, loads, twists, dt)
        dissipated = -work
        _, off = blocks_of(held["resistance"], count)
        after, before = twists.reshape(-1), held["start_twists"].reshape(-1)
        residual = dt * jnp.dot(after, jnp.matmul(off, after - before, precision=PRECISION), precision=PRECISION)
        judged = (state["steps"] >= 1.0) & (state["stopped_at"] == 0.0)
        gift = judged & ~(dissipated >= -IDENTITY_ULPS * jnp.finfo(dtype).eps * scale)
        return (jnp.where(judged, dissipated, 0.0).astype(dtype), jnp.where(judged, residual, 0.0).astype(dtype),
                gift)

    @override
    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        count = len(self._members)
        dtype = state["gain"].dtype
        rest = jnp.asarray((0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0), dtype=dtype)
        poses = jnp.stack([boundary_inputs.get(f"pose_{i}", rest) for i in range(count)])
        twists = jnp.stack([boundary_inputs.get(f"twist_{i}", jnp.zeros(6, dtype=dtype)) for i in range(count)])
        masses = jnp.stack([boundary_inputs.get(f"mass_matrix_{i}", jnp.zeros((6, 6), dtype=dtype))
                            for i in range(count)])
        found, accepted, wall_short, cross_short, gaps, apart, beyond, radii, report = self.solved(
            p, poses, state["steps"])
        found = found.astype(dtype)
        symmetric = 0.5 * (found + found.T)
        own, off = blocks_of(symmetric, count)
        loads = cross_loads(off, twists)
        judged = _lax.stop_gradient(found)
        held_own, held_off = blocks_of(0.5 * (judged + judged.T), count)
        held_masses, held_twists = _lax.stop_gradient(masses), _lax.stop_gradient(twists)
        least = jnp.stack([scaled_least_eigenvalue(judged[6 * i:6 * i + 6, 6 * i:6 * i + 6])
                           for i in range(count)]).astype(dtype)
        cluster = scaled_least_eigenvalue(judged).astype(dtype)
        dropped = asymmetry(judged).astype(dtype)
        gain = lagged_gain(held_own, held_off, held_masses, dt).astype(dtype)
        amplification = lagged_amplification(held_own, held_off, held_masses, dt).astype(dtype)
        form, _ = quadratic_form(0.5 * (judged + judged.T), held_twists)
        dissipated, residual, gift = self.judged_books(state, held_twists, dt)
        nearest = jnp.argmin(apart.reshape(-1))
        pair = jnp.stack([nearest // count, nearest % count]).astype(dtype)
        least_gap = (jnp.min(apart) if count > 1 else jnp.zeros(())).astype(dtype)
        contact = jnp.any(apart < CONTACT_GAP * jnp.minimum(radii[:, None], radii[None, :]))
        overlap_stop = jnp.any(beyond)
        member_stop, cluster_stop, gain_stop = ~(jnp.min(least) > 0.0), ~(cluster > 0.0), ~(gain < 1.0)
        stop = overlap_stop | (accepted & (member_stop | cluster_stop | gain_stop))
        why = jnp.where(overlap_stop, WHY_OVERLAP, jnp.where(member_stop, WHY_MEMBER, jnp.where(
            cluster_stop, WHY_CLUSTER, WHY_GAIN))).astype(dtype)
        value = jnp.where(overlap_stop, least_gap, jnp.where(member_stop, jnp.min(least),
                                                             jnp.where(cluster_stop, cluster, gain))).astype(dtype)
        member = jnp.where(overlap_stop, pair[0], jnp.argmin(least).astype(dtype))
        other = jnp.where(overlap_stop, pair[1], member)
        steps = state["steps"] + 1.0
        first = stop & (state["stopped_at"] == 0.0)
        record = jnp.where(first, jnp.concatenate([steps[None], why[None], member[None], other[None], value[None],
                                                   poses.reshape(-1)]),
                           jnp.concatenate([state["stopped_at"][None], state["stopped_why"][None],
                                            state["stopped_member"][None], state["stopped_other"][None],
                                            state["stopped_value"][None], state["stopped_poses"].reshape(-1)]))
        out: dict[str, Array] = {}
        for i in range(count):
            out[f"resistance_{i}"] = jnp.where(stop, jnp.nan, own[i]).astype(dtype)
            out[f"background_load_{i}"] = jnp.where(stop, jnp.nan, loads[i]).astype(dtype)
        flags = jnp.stack([stop & ~overlap_stop & (member_stop | cluster_stop), stop & ~overlap_stop & gain_stop,
                           dropped > CONFINED_ASYMMETRY_LIMIT, gift, cross_short, overlap_stop, contact]).astype(dtype)
        relaxation = jnp.where(amplification < 1.0,
                               1.0 / (1.0 - jnp.minimum(amplification, 1.0 - jnp.finfo(dtype).eps)), jnp.inf)
        out.update(resistance=symmetric, least_eigenvalue=least, cluster_least=cluster, asymmetry=dropped, gain=gain,
                   amplification=amplification, relaxation_steps=relaxation.astype(dtype),
                   dissipation=form.astype(dtype), cluster_dissipation=dissipated, lag_residual=residual,
                   start_twists=twists.astype(dtype), least_gap=least_gap, gap_pair=pair, wall_accepted=accepted,
                   indefinite=flags[0], gain_exceeded=flags[1], asymmetric=flags[2], cluster_gift=flags[3],
                   cluster_gift_steps=state["cluster_gift_steps"] + flags[3], cross_cells_short=flags[4],
                   overlap=flags[5], near_contact=flags[6], steps=steps, stopped_at=record[0], stopped_why=record[1],
                   stopped_member=record[2], stopped_other=record[3], stopped_value=record[4],
                   stopped_poses=record[5:].reshape(count, 7))
        if report is None:
            out["solve_flagged"] = jnp.zeros((), dtype=dtype)
        else:
            report = report.astype(dtype)
            out.update(solve_report=report, solve_report_sum=jnp.asarray(state["solve_report_sum"]) + report,
                       solve_flagged=(report[2] + report[3] + report[4] > 0).astype(dtype))
        near = jnp.asarray(False)
        if self._gapped():
            spacing = jnp.stack([p[f"{member_prefix(i)}force_spacing"] for i in range(count)]).astype(dtype)
            near = jnp.any(gaps < NEAR_WALL * spacing)
            out.update(wall_gap=gaps.astype(dtype), wall_gap_spacings=(gaps / spacing).astype(dtype))
        out["near_wall"] = near.astype(dtype)
        if self._wall_budgeted():
            out["wall_cells_short"] = wall_short.astype(dtype)
        return out

    # -- the build's checks and report ---------------------------------------------------------------------------------

    def start_inputs(self, start: Mapping[str, Any]) -> dict[str, Array]:
        return {key: as_run_array(value) for key, value in start.items() if key in self.boundary_input_spec()}

    @cached_property
    def _started(self) -> dict[bytes, dict[str, Any]]:
        return {}

    def judged_at(self, start: Mapping[str, Any]) -> dict[str, Any]:
        """The step's judgement at the starting inputs ``start`` with the built leaves, as numbers; kept by its
        inputs, so that the build's check and its report take one step between them."""
        inputs = self.start_inputs(start)
        key = b"".join(np.asarray(inputs[name], dtype=np.float64).tobytes() for name in sorted(inputs))
        held = self._started.get(key)
        if held is None:
            out = self.update(self.initial_state(), inputs, self.delta_t, params=self.params_pytree())
            held = {name: np.asarray(value) for name, value in out.items()}
            self._started[key] = held
        return held

    def built_problems(self, start: Mapping[str, Any] | None = None) -> list[Problem]:
        """Each member's own refusals of its free part (``MRSDragNode.built_problems``); a joint system whose run would
        not fit in the memory a run can have (:meth:`memory_problem`, before any step); and at the bodies' starting
        poses and twists, two bodies that overlap, a body's or the cluster's resistance that is not positive definite,
        or a lagged gain not below one."""
        found: list[Problem] = []
        for body, member in zip(self.members, self._members, strict=True):
            found += [Problem(problem.code, f"{body}: {problem.message}", problem.action)
                      for problem in member.built_problems(None)]
        if found:
            return found
        memory = self.memory_problem()
        if memory is not None:
            return [memory]
        if start is None or not all(f"pose_{i}" in start for i in range(len(self.members))):
            return found
        judged = self.judged_at(start)
        if float(judged["stopped_at"]) > 0.0:
            problem = self.stop_problem(judged)
            if problem is not None:
                found.append(Problem(problem.code, f"at the bodies' starting poses: {problem.message}",
                                     problem.action))
        return found

    def built_lines(self, start: Mapping[str, Any] | None, closest_approach: float | None) -> list[str]:
        """What the build reports of the near field: which bodies it couples, its reach and budget, the solve, its
        size and the gain and margins at the start."""
        counts = ", ".join(f"{body} {op.count}" for body, op in zip(self.members, self._operators, strict=True))
        budgets = ", ".join(f"{body} {op.budget}" for body, op in zip(self.members, self._operators, strict=True))
        share = float(self.constants()["cross_share"])
        constants = self.constants()
        if self.iterative:
            solve = (f"solved each step by GMRES on the cluster's grand resistance's {6 * len(self.members)} "
                     f"right-hand sides at once, as its drags declare (NearField(solve='gmres')), each body's held "
                     f"free part's inverse its preconditioner, each solve to a relative tolerance of "
                     f"{float(constants['tolerance']):.3g} on its true residual (NearField's tolerance; a restart every "
                     f"{int(constants['restart'])} iterations, a cap of {int(constants['cap'])}; a solve that stops "
                     f"short raises solve_flagged and is warned of)")
        else:
            solve = ("solved each step by one dense LU for the cluster's grand resistance, the dense route at every "
                     "size (NearField(solve='gmres') declares GMRES)")
        lines = [f"the medium's near field couples {', '.join(self.members)} in one joint system (force points: "
                 f"{counts}; {self.unknowns} unknowns), {solve}, each body's step lagged: its own coupled 6 x 6 "
                 f"implicit, the others' loads from the step before",
                 f"the cross-kernel between bodies: the source's regularised Stokeslet at its force points where the "
                 f"pair is farther apart than {self.constants()['reach']:g} of the source's cells' own size, "
                 f"integrated over every cell within half that, blended between; integrated over at most {budgets} "
                 f"quadrature points a source ({share:g} of them), the nearest cells first (cross_cells_short where "
                 f"a cell within the reach of the other body was left at its force point)",
                 self._memory_line()]
        if start is not None and all(f"pose_{i}" in start for i in range(len(self.members))):
            judged = self.judged_at(start)
            margins = ", ".join(f"{body} {float(v):.3g}" for body, v in zip(self.members, judged["least_eigenvalue"],
                                                                             strict=True))
            gain = float(judged["gain"])
            first, second = (self.members[int(round(float(i)))] for i in judged["gap_pair"])
            lines.append(f"at the start: the lagged gain {gain:.3g} (at one or above the run stops; the check is "
                         f"sufficient for the lagged step's stability, not necessary), the lagged step's "
                         f"amplification {float(judged['amplification']):.3g} (an error relaxes over about "
                         f"{float(judged['relaxation_steps']):.3g} steps), the scaled least eigenvalue of each body's "
                         f"coupled resistance {margins} and of the cluster's {float(judged['cluster_least']):.3g} (its "
                         f"margin; not above zero, the run stops), and the least gap between two bodies' surfaces, "
                         f"{first} and {second}, {float(judged['least_gap']) * 1e3:.4g} mm as their discretisations "
                         f"estimate it (closer than {CONTACT_GAP:g} of the smaller one's equivalent radius, a step "
                         f"raises near_contact, the validity line's edge)")
            if self.iterative:
                lines.append(self._start_solve_line(judged["solve_report"]))
        return lines

    @staticmethod
    def _start_solve_line(report: np.ndarray) -> str:
        """The starting step's joint solve on the iterative route, as its report says (``REPORT_FIELDS``)."""
        iterations, residual, at_cap, stagnated, not_finite = (float(value) for value in report[:5])
        short = ", ".join(f"{int(count)} {what}" for what, count in (
            ("at the cap", at_cap), ("stagnated", stagnated), ("not finite", not_finite)) if count)
        ending = f"columns stopped short ({short}): the first step raises solve_flagged" if short else \
            "every column converged"
        return (f"at the start: the joint solve took {int(iterations)} iterations to a largest true relative residual "
                f"of {residual:.3g}; {ending}")

    def _memory_line(self) -> str:
        where, available = device_memory()
        room = f"{available / 2 ** 30:.3g} GiB" if available is not None else "which the system does not report"
        return (f"its joint matrix in the run's dtype {_itemsize() * self.unknowns ** 2 / 2 ** 20:.4g} MiB; the run's "
                f"memory on the {self.route} route about {self.memory_estimate() / 2 ** 20:.4g} MiB past the "
                f"process's fixed part, a gradient study's derivative of one point about "
                f"{self.memory_estimate('forward') / 2 ** 20:.4g} MiB in forward mode along one axis and "
                f"{self.memory_estimate('reverse') / 2 ** 20:.4g} MiB in reverse mode (the counts a CPU run was "
                f"measured to hold, as the drag's validity line says), against {where}, {room}, refused past "
                f"{MEMORY_MARGIN:g} times the estimate against {1.0 - MEMORY_HEADROOM:.0%} of it")

    def run_problem(self, end: int) -> Problem | None:
        """A run that would end past the last row of the derivative solves' per-step reports (``solve_report_rows``) is
        refused before it runs: a row is a step's, and a step past the last would be summed into it."""
        rows = int(self.constants().get("solve_report_rows", 0))
        if rows == 0 or end <= rows:
            return None
        return Problem("drag.near-field-report-rows",
                       f"the near field's derivative solves' reports are kept a row a step for {rows} steps, and this "
                       f"run would end at step {end}",
                       "build the node with solve_report_rows at or past the run's steps, or solve_report_rows=0 to sum "
                       "them over the run")

    def stop_problem(self, state: Mapping[str, Any]) -> Problem | None:
        """Why the node stopped a run, from its state at the run's end: the first step at which a body's or the
        cluster's resistance turned indefinite, or the lagged gain reached one; ``None`` where it stopped none."""
        if "stopped_at" not in state or not float(np.asarray(state["stopped_at"])) > 0.0:
            return None
        step = int(round(float(np.asarray(state["stopped_at"]))))
        why = float(np.asarray(state["stopped_why"]))
        value = float(np.asarray(state["stopped_value"]))
        poses = np.asarray(state["stopped_poses"], dtype=np.float64).reshape(len(self.members), 7)
        where = "; ".join(f"{body} at ({', '.join(f'{c * 1e3:.5g}' for c in pose[:3])}) mm"
                          for body, pose in zip(self.members, poses, strict=True))
        when = (f"at its step {step} (t = {step * self.delta_t:.6g} s), and the run stopped there, before the bodies "
                f"took it")
        if why == WHY_OVERLAP:
            first, second = (self.members[int(round(float(np.asarray(state[key]))))]
                             for key in ("stopped_member", "stopped_other"))
            return Problem(
                "drag.near-field-overlap",
                f"{first} and {second} overlap {when} ({where}): a point of {first} lies {-value * 1e3:.4g} mm behind "
                f"{second}'s surface as their discretisations estimate it (deeper than {OVERLAP_RESOLUTION:g} of a "
                f"cell, their estimate's resolution), so that no fluid lies between them and their joint system "
                f"describes none",
                "start the bodies apart, Body(..., initial=State(position=...)), or hold them apart, "
                "Contact.Projection(margin=...)")
        if why == WHY_GAIN:
            return Problem(
                "drag.near-field-gain",
                f"the lagged coupling of {', '.join(self.members)} may grow: its gain, the spectral radius of "
                f"(M/dt + R_d)^-1 R_off, the per-step fixed point's contraction, was {value:.4g} {when} ({where}); "
                f"the check is sufficient for the lagged step's stability, not necessary: overdamped (three bodies "
                f"in a triangle near contact) the step's amplification is the gain and grows, while with inertia it "
                f"lies below the gain, so that the check refuses some stable runs (three spheres near contact between "
                f"about tau/dt 0.05 and 0.11)",
                "write a shorter step, dt=..., where the bodies are inertial enough for it to lower the gain, or hold "
                "the bodies further apart, Contact.Projection(margin=...); the joint step that takes such clusters "
                "is not in this release")
        member = self.members[int(round(float(np.asarray(state["stopped_member"]))))]
        what = f"{member}'s coupled resistance" if why == WHY_MEMBER else "the cluster's grand resistance"
        return Problem(
            "drag.near-field-indefinite",
            f"{what} turned indefinite {when}: its least eigenvalue, each row and column scaled by its diagonal, was "
            f"{value:.3g} ({where}); the bodies' discretisations do not resolve the gap between them, or a wall's "
            f"image there",
            "write a finer spacing on the bodies' drag, Drag.MRS(spacing=...), or hold them further apart, "
            "Contact.Projection(margin=...)")


def device_memory() -> tuple[str, int | None]:
    """The memory a run's arrays can have and its size (``mime.build._maddening.device_memory``)."""
    from mime.build._maddening import device_memory as held

    return held()


def _itemsize() -> int:
    """The bytes of a floating value in the run's dtype (float64 with JAX's x64 mode, else float32)."""
    return int(jnp.zeros(()).dtype.itemsize)


__all__ = ["CONTACT_GAP", "DERIVATIVE_COUNTS", "DERIVATIVE_MEASURED", "EQUATIONS", "HELD_MATRIX", "MEMORY_COUNTS",
           "MEMORY_HEADROOM", "MEMORY_MARGIN", "MEMORY_MEASURED", "ROUTES",
           "JointDragInputs", "JointDragNode", "JointDragState", "member_prefix"]
