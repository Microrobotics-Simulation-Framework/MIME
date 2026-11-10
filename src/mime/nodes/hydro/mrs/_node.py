"""The MRS resistance of a rigid body: its hull kind's samples, placed by the body's pose, solved for the
resistance about its centre of mass at every step."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from functools import cached_property
from typing import Any, Literal, NamedTuple, cast, override

import jax
import numpy as np
from jax import Array
from jaxtyping import Bool, Float
from maddening.core.compliance.metadata import NodeMeta, Reference
from maddening.core.node import BoundaryInputSpec
from maddening.core.params import ParamSpec
from typing_extensions import Annotated, NotRequired, ReadOnly, TypedDict

from mime.core import MimeNode, NodeFields, Symbol, SymbolKind, Units, as_run_array, jit, jnp, node_fields, \
    node_input, quaternion, run_constants_given, typed_view
from mime.nodes.hydro.mrs._quadrature import (
    Chunks,
    padded,
    asymmetry,
    capped_cells,
    cell_moments,
    empty_cells,
    held_or_made,
    scaled_least_eigenvalue,
    separated_matrix,
    separated_resistance,
    turned,
    turned_resistance,
    unchanged,
)
from mime.nodes.hydro._near_singular import blended, nearest_cells, smoothstep, within_reach
from mime.nodes.hydro.mrs._solve import PRECISION, blob_squared, resistance
from mime.nodes.hydro.mrs._walls import Corrected, WallCorrection, cell_image, wall_correction_named
from mime.nodes.hydro.mrs._handover import (
    DEFAULT_ROUGHNESS,
    HANDOVER_SPACINGS,
    WINDOW,
    Search,
    handed_over,
    search_for,
)
from mime.nodes._kinds import held_as_built, kind_from_settings, registered_kind, setting_units, traced_settings
from mime.registry import Problem, Samples, Sampling

EQUATIONS = (r"\mathbf{R} = \mathbf{M}^{\top} (\mathbf{A} + \mathbf{G})^{-1} \mathbf{M}, \quad "
             r"\mathbf{A}_{mn} = \frac{(r^{2} + 2 \epsilon^{2}) \mathbf{I} + \mathbf{r} \mathbf{r}^{\top}}"
             r"{8 \pi \mu (r^{2} + \epsilon^{2})^{3/2}}, \quad \mathbf{r} = \mathbf{x}_{m} - \mathbf{x}_{n}, \quad "
             r"\epsilon^{2} = b^{2} \frac{1}{N} \sum_{n} w_{n}, \quad "
             r"\mathbf{x}_{n} = \mathbf{p} + R(\mathbf{q}) (\mathbf{P} \mathbf{y}_{n} + \mathbf{c})")

#: ``jax.checkpoint``, whose hints leave its result partly unknown to a strict checker, and the pytree functions.
checkpoint: Any = cast(Any, jax).checkpoint
_lax: Any = cast(Any, jax).lax
_ops: Any = cast(Any, jax).ops
_tree_util: Any = cast(Any, jax).tree_util
#: Where the symbol rows' functions are.
_ADVANCE = "mime.nodes.hydro.mrs._node.MRSDragNode.advance"
_SOLVE = "mime.nodes.hydro.mrs._solve.resistance"
#: The prefix of a node constant holding a setting of the body's hull kind, and of one holding a wall's parameter.
SHAPE, WALL = "shape_", "wall_"
#: The share of a separated quadrature's free resistance its symmetric part may drop before the build refuses it
#: (``asymmetry``). A labelled decision of phase 7m-b: at every setting measured there it was below 1e-3 (2e-6 on the
#: sphere at its default, 8e-4 on FL-9 at 206 force points and blob 1.0), and the rounding-dominated solves of the
#: helix reference reached 2e-2; the limit catches the gross failures (each quadrature point planted in its second
#: nearest force point's cell, 3.5e-2), not every solve that rounding has reached, and is no measure of the
#: discretisation's error: every setting of FL-9 measured in free space, blobs 0.1 to 2.0 and refinements 1 to 16, lay at
#: 3.5e-4 to 8e-4 or, on the point forces, at rounding, whatever it missed by (to 72%). That is
#: :data:`QUADRATURE_MOVE_LIMIT`'s (phase 7m-b's fix round, the review's W7).
ASYMMETRY_LIMIT = 1e-2
#: The ``params`` leaf holding, on a separated quadrature, the body-frame quantity made once per build: in free space
#: the resistance about the body's origin (6 x 6), with a wall the free part's matrix (3N x 3N), at unit viscosity.
HELD = "held_in_body_frame"
#: The prefix of the ``params`` leaves beside :data:`HELD` that hold the leaves it was made at, one each
#: (``held_from.blob``, ``held_from.shape_radius``): what the step compares the leaves it is given with, as values of its
#: own rather than constants of its program, so that the program is the same at every value and what writes the held
#: value with its leaves (a study's member, an override, the save's resolution rerun) keeps it.
HELD_FROM = "held_from."
#: The body's least distance from a wall, in force spacings, below which a step on a separated quadrature flags it
#: (``near_wall``; MIME-ANO-130). A labelled decision of phase 7m-b's fix round. In the review's map (FL-9 in the 1/4 in
#: tube, 297 poses: spacings 0.3 to 0.829 mm, five discretisations and 0.3.0's point forces, gaps 0.05 to 0.8 mm) every
#: indefinite resistance, 53 of them, had a gap of at most half a force spacing (the largest exactly half: 0.6 mm at
#: blob 0.5 and the quadrature four times finer, 0.3 mm from the wall); this round mapped 96 more poses between 0.55
#: and one spacing, at eight rolls each (0.6 mm at the default and at that discretisation, 0.829 mm at the default),
#: and every one was definite (the least +0.14, at 0.55), as was the 1 mm sphere at its default from a tenth of a
#: spacing to one. The bound keeps a factor of two over the largest indefinite gap. A flag, never a stop: the stop is
#: the resistance's own eigenvalue.
NEAR_WALL = 1.0
#: On a graded sampling, how far past the body's least distance from the wall, in force spacings, a point of its
#: quadrature still counts as nearest for ``off_patch``: any such point outside the graded patch raises it, so that a
#: face turned towards the wall (a tie, or within a degree or two of one) is judged by all of it, not by whichever of its
#: points the least happens to fall on. Chosen against the flag's controls, before their rates were measured: a box's
#: face turned 89 degrees towards the wall lies within 0.02 force spacings of its least distance where it leaves the
#: patch, and the gate's sphere (a radius of 3.3 force spacings, its contact graded for half a spacing) brings the
#: patch's edge within 0.05 force spacings of the wall about 10 degrees before the edge itself reaches the nearest point.
OFF_PATCH_BAND = 0.05
#: How a graded run's wall self-image measures each cell's distance from the wall when it blends the integrated image
#: in (decision #308): ``"cell"``, in the cell's own size, ``sqrt(area)``, decided and shipped; ``"force_spacing"``, in
#: the force spacing, as a uniform run's. Measured through the node on the gate's sphere, its contact graded for half a
#: force spacing and the image integrated, against Brenner's series at half a spacing: +5.24 % (cell) and +3.33 %
#: (force spacing) at blob 0.1, +1.67 % and -0.04 % at 0.025, the uniform sampling +1.47 % and -2.18 %. The one switch
#: between them; a uniform run reads neither.
GRADED_WALL_MEASURE: Literal["cell", "force_spacing"] = "cell"
#: What the build report says was mapped near a wall, for a wall kind without figures of its own
#: (``WallCorrection.near_wall_text``).
NEAR_WALL_MAPPED = "every indefinite resistance mapped lay within half a force spacing of the wall (MIME-ANO-130)"
#: The gap to a wall, in force spacings, a body keeps with the wall's image at the force points (the default): a confined
#: body's default force spacing keeps its closest approach here (``Drag``'s rule), and a refusal or a stop near the wall
#: names the spacing that does. Design #210's rule, fixed before its measurement: the review's map found one indefinite
#: pose at exactly half a force spacing (0.6 mm, blob 0.5 and the quadrature four times finer), and all 96 poses from
#: 0.55 to one spacing were definite (:data:`NEAR_WALL`). FL-9 in the 1/4 in tube at 0.181 mm, the floor 0.1 mm away and so
#: 0.55 spacings, its eight rolls: definite (+0.74 to +0.77), and 588 to 625 um per turn, within 0.2% of the frozen point's
#: 587 to 624 at either end; a definite resistance is not shown accurate in general.
CONFINED_GAP = 0.55
#: The same gap with the wall's image integrated over each cell, written with ``wall_refinement`` (opt-in, design #210):
#: below it the resistance was measured indefinite (MIME-ANO-130). FL-9 in the 1/4 in tube at the screw's defaults and
#: wall_refinement 2 under the cap, eight rolls at each gap: indefinite at some roll
#: at 0.15 and 0.2 force spacings at 0.4 and 0.829 mm, at 0.15 and 0.25 at 0.6 mm (the largest 0.25, -3.35), definite at every
#: roll at 0.3 and 0.36 at all three. The figure is 1.44 times the largest
#: gap with an indefinite roll, and never below the 0.36 the uncapped image measured (the uncapped: indefinite at 0.15, 0.2
#: and 0.25; definite at 0.3 and 0.36). Definite is not accurate: at 0.36 spacings the speed per turn was 407 to 444 um at
#: 0.829 mm, 451 to 480 at 0.6 mm and 526 to 559 at 0.4 mm under the cap (411 to 460 and 455 to 483 uncapped), against
#: about 600 at script A's frozen point.
INTEGRATED_GAP = 0.36
#: How near the wall, in force spacings of its force point, a cell takes the wall's image integrated over it: beyond
#: WALL_REACH its column is the image at its force point, within WALL_REACH / 2 the integrated one, and between them a
#: smooth blend of the two, so that a cell's column is continuous as it nears the wall. A labelled decision of phase
#: 7m-b's fix round (design #204, the maintainer's cost point), at a bound of 5% on a column: FL-9 in the 1/4 in tube
#: at the screw's defaults and wall_refinement 2, seven poses from the floor to the axis, each cell's column of the
#: image at its force point against the integrated one, the largest difference beyond two spacings was 3.9% (0.4 mm)
#: and 4.2% (0.829 mm), the median 0.9% to 1.5% between one and three; within half a spacing up to 22% and 50%. The
#: difference falls slowly (below 1% for every cell only from about seven spacings at 0.4 mm), so a bound of 1% would
#: integrate nearly every cell of this tube.
WALL_REACH = 2.0
#: The share of a body's cells the step integrates at most (its budget, ``wall_cells``, fixed at build so that the
#: step's shapes are), the cells nearest the wall; a step with more cells than that within WALL_REACH raises
#: ``wall_cells_short`` and takes the image at the force points for the farthest of them. Measured on FL-9's floor at
#: the 0.1 mm margin, eight rolls: within two spacings of the wall lay at most 14.6% of the cells at 0.278 mm, 16.2% at
#: 0.3, 23.7% at 0.4, 44% at 0.6 and 63% at 0.829. A cell's row is padded to :data:`WALL_CELL_CAP`, so a budget pays
#: only below about 0.41 (the kept points over the padded row, at every spacing measured); 0.25 holds the cells within
#: the reach from 0.4 mm down, and a coarser spacing flags that it is short.
WALL_CELL_SHARE = 0.25
#: The most wall points a cell's image is integrated on, in multiples of the mean over the cells, rounded up
#: (:func:`mime.nodes.hydro.mrs._quadrature.capped_cells`): a tensor grid crowds some cells (at 0.4 mm on FL-9 the widest
#: holds 67 points against a mean of 9), and a step reads every target against every cell padded to the widest, so
#: without a cap the budget of cells nearest the wall holds as many reads as the whole wall quadrature. Design #210, on the
#: maintainer's figure; what it does to a column is in MIME-ANO-130.
WALL_CELL_CAP = 2.0
#: What a run on a wall's image holds, in GiB in float64: ``a + b N^2`` fitted by least squares to six measurements of a
#: process's peak resident memory, each one build and a compile and two steps at each of eight floor rolls, FL-9 in the
#: 1/4 in tube with the image at the force points, on a 24-thread laptop CPU (phase 7m-b's second fix round, on design
#: #210's measure): 1.95, 2.11, 2.26, 3.15, 3.65 and 13.64 GiB at 222, 567, 957, 1,714, 1,966 and 4,816 force points
#: (0.829, 0.5, 0.4, 0.3, 0.277 and 0.181 mm). The fit is within 7% of each (-6.6% at 567, +4.8% at 1,714), and left out
#: of it each is predicted within 9%, the largest count within 12%. The solve's matrices, ``3N x 3N``, grow as the square
#: of the count; the rest (the interpreter, the compiled programs, the wall's table) does not. The fix round's line
#: through two points (1.07 GiB at 222 on another measure, 13.6 at 4,816) predicted 3.1 GiB at 1,966, 14% low.
RUN_GIB_AT_NO_POINTS, RUN_GIB_PER_POINT_SQUARED = 1.81, 5.08e-7
#: What the build report's memory figures are: :func:`run_gib`'s fit, as the report states it.
RUN_GIB_BASIS = ("a fit of the square of the force points' count to six measurements of a run's peak on a 24-thread "
                 "laptop CPU, from 222 to 4,816 force points, within 7% of each")
#: A step's whole time at the default near a wall, a measurement for the build's report beside its estimate of the reads:
#: FL-9 on the floor of the 1/4 in tube, float64, the laptop's CPU.
STEP_MEASURED = "the whole step took 45 s at 4,816 force points and 8 s at 1,966, each on a floor"
#: The memory and time of the image at the force points against the image integrated over each cell, at the spacings
#: their gaps ask of one closest approach, for the build's report: FL-9 held 0.1 mm off the floor of the 1/4 in tube,
#: its eight rolls, float64, the laptop's CPU (design #210's measurement; each run's process peak).
WALL_IMAGE_MEMORY = ("FL-9 held 0.1 mm off a 1/4 in tube's floor took 0.277 mm and wall_refinement 2, about 50 s a step "
                     "and a peak of 4.0 GiB, against 0.181 mm with the image at the force points, 45 s a step and "
                     "13.6 GiB, both definite on 8 of 8 rolls (float64, a 24-thread laptop CPU)")
#: The time one read of a wall's table takes, for the build's estimate of a step's cost: measured on a 24-thread laptop
#: CPU in float64 (phase 7m-b's fix round, FL-9 in the 1/4 in tube: 1.36 to 1.52 us over 0.9 to 6.4 million reads a
#: step). An estimate for one machine, stated as one.
READ_SECONDS = 1.5e-6
#: The share of the confined resistance its symmetric part drops above which a step flags it (``asymmetric``): the
#: dropped share bounds the resistance's error from below (its skew part is its distance from the nearest symmetric
#: matrix, as the exact resistance is), so above this the step's resistance is wrong by at least half of it, 2.5%, in
#: the norm the share is taken in. A labelled decision of phase 7m-b's fix round: measured 0.007 to 0.012 on the tube's
#: axis (FL-9 at 0.6 and 0.829 mm, the free part's 4e-4: the image taken at the force points is the larger part), 0.01
#: to 0.04 between 0.55 and one force spacing from the wall at the defaults (to 0.33 at blob 0.5), 0.24 to 0.46 on the
#: floor at 0.829 mm (MIME-ANO-130). A report, never a stop or a refusal: a small share proves nothing (the point
#: forces are symmetric to rounding whatever they miss).
CONFINED_ASYMMETRY_LIMIT = 5e-2
#: How far the free resistance's significant entries (those above a hundredth of their row's and column's diagonal
#: scale) may move, each relative to itself, from a written quadrature_refinement below the rule's to the rule's
#: (``reference_quadrature_counts``) before the build refuses the written one (``drag.mrs-quadrature``): the quadrature's
#: own error, estimated by refining it. A labelled decision of phase 7m-b's fix round, on what predicts the miss
#: (CLAUDE.md, "prefer the metric that predicts"): over fourteen refinements below the rule (FL-9 at 0.829 mm and the
#: 1 mm sphere at 0.25 mm, blobs 0.1 to 1.0, float64) the move predicted the change the rule's quadrature makes in FL-9's
#: free-space speed per turn, and in the sphere's drag and rotational drag, to within 1.25 times, from 0.001% to 71%
#: (the point forces at blob 0.1, which miss the shipped reference by 72%), while the free part's asymmetry, 2e-17 to
#: 5e-4, predicted none of it (the fix round's table, in its pull request); the limit lies between the largest move
#: among those within 2% of the rule (1.8%, the sphere's point forces at blob 0.5) and the smallest beyond it (8.0%,
#: the sphere's at blob 0.25). What it bounds, re-measured over 92 settings in the second fix round (FL-9 at 0.6, 0.829
#: and 1.2 mm, the sphere at 0.25, 0.35 and 0.5 mm, blobs 0.1 to 2.0, every refinement below the rule): the move of
#: the entries above the cut, and so an answer that rests on them. It is an upper-bound proxy, not a two-sided
#: prediction: of the 26 settings with a move above 1%, 25 changed the answer by 0.88 to 1.25 times the move, and
#: below 1% the move mostly over-predicts (the change 0.006 to 1.33 times the move, at moves too small to matter). An answer that rests on an entry under the cut is not bounded: FL-9
#: at 1.2 mm, blob 0.7 and refinement 1 has its propulsive coupling, -R03, at 0.99% of its scale, uncounted, which
#: moved 7.1% and its speed per turn 6.2% against a move of 2.2%, the one setting of the 92 let through with an
#: answer moved past the limit. It is not caught on purpose: no cut both catches it and keeps the refusals tied to the
#: answers on this sample but one fitted to it (0.55% to 0.99%; at 0.5% four settings whose answer moved 0.2% to 3.4%
#: are refused, and judging every entry, those under the cut on a hundredth of their scale, refuses 13), and the
#: setting's answer is dominated by what the limit does not judge, its blob of 0.84 mm: at the rule's own quadrature
#: its speed per turn is 36% below the shipped reference, which the validity line's blob error states.
QUADRATURE_MOVE_LIMIT = 0.05
#: The entries of a resistance the quadrature's move is judged on: those above this share of their row's and column's
#: diagonal scale (an entry zero by the body's symmetry is rounding, and its relative move is noise).
SIGNIFICANT_ENTRY = 1e-2


class Built(NamedTuple):
    """What a node on a separated quadrature makes once per build, in float64 at unit viscosity in the body's frame: the
    held quantity (the resistance in free space, the free part's matrix with a wall); the free part's resistance's
    asymmetry, its least eigenvalue each row and column scaled by its diagonal, and its symmetric part; and, where a
    reference quadrature was given, how far its significant entries move from it (``QUADRATURE_MOVE_LIMIT``)."""

    held: np.ndarray
    asymmetry: float
    least: float
    free: np.ndarray
    moved: float | None


def quadrature_move(found: np.ndarray, reference: np.ndarray) -> float:
    """How far ``found``'s significant entries lie from ``reference``'s, each relative to itself, the largest: an entry is
    significant above :data:`SIGNIFICANT_ENTRY` of its row's and column's diagonal scale in ``reference``."""
    scale = np.sqrt(np.outer(np.abs(np.diag(reference)), np.abs(np.diag(reference))))
    significant = np.abs(reference) > SIGNIFICANT_ENTRY * scale
    return float(np.max(np.abs(found[significant] - reference[significant]) / np.abs(reference[significant])))


class MRSDragState(TypedDict, closed=True):
    resistance: ReadOnly[Float[Array, "6 6"]]  # noqa: F722 - a jaxtyping shape
    background_load: ReadOnly[Annotated[Float[Array, "6"], Units("N; N m")]]
    wall_accepted: ReadOnly[Annotated[Bool[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape


class SeparatedDragState(TypedDict, closed=True):
    """The state of a node on a separated quadrature: the resistance, the background load and whether every read of
    the wall's data was accepted, as on the point forces; the step's judgement of its resistance, its least eigenvalue
    each row and column scaled by its diagonal and the share its symmetric part drops, with the flags they raise; and,
    with a wall, the body's least distance from it (m, and in force spacings, the near-wall flag's measure) and the
    stop's record: the node's count of steps, and the step (zero for none), pose, gap and least eigenvalue at which its
    resistance first turned indefinite. On a graded sampling, ``off_patch``: the body within :data:`NEAR_WALL` force
    spacings of the wall with a point of its quadrature within :data:`OFF_PATCH_BAND` force spacings of its least
    distance outside the graded patch (zero without a wall)."""

    resistance: ReadOnly[Float[Array, "6 6"]]  # noqa: F722 - a jaxtyping shape
    background_load: ReadOnly[Annotated[Float[Array, "6"], Units("N; N m")]]
    wall_accepted: ReadOnly[Annotated[Bool[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    least_eigenvalue: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    asymmetry: NotRequired[ReadOnly[Annotated[Float[Array, ""], Units("1")]]]  # noqa: F722 - a jaxtyping shape
    indefinite: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    near_wall: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    asymmetric: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    wall_gap: NotRequired[ReadOnly[Annotated[Float[Array, ""], Units("m")]]]  # noqa: F722 - a jaxtyping shape
    wall_gap_spacings: NotRequired[ReadOnly[Annotated[Float[Array, ""], Units("1")]]]  # noqa: F722 - a jaxtyping shape
    wall_cells_short: NotRequired[ReadOnly[Annotated[Float[Array, ""], Units("1")]]]  # noqa: F722 - a jaxtyping shape
    off_patch: NotRequired[ReadOnly[Annotated[Float[Array, ""], Units("1")]]]  # noqa: F722 - a jaxtyping shape
    steps: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    stopped_at: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    stopped_pose: ReadOnly[Annotated[Float[Array, "7"], Units("m; 1")]]
    stopped_gap: NotRequired[ReadOnly[Annotated[Float[Array, ""], Units("m")]]]  # noqa: F722 - a jaxtyping shape
    stopped_least: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    film_increment: NotRequired[ReadOnly[Annotated[Float[Array, "6 6"], Units("N s/m; N s; N m s")]]]  # noqa: F722 - a jaxtyping shape
    film_coupling: NotRequired[ReadOnly[Annotated[Float[Array, "3 3"], Units("N s")]]]  # noqa: F722 - a jaxtyping shape
    contact_gap: NotRequired[ReadOnly[Annotated[Float[Array, ""], Units("m")]]]  # noqa: F722 - a jaxtyping shape
    film_residual: NotRequired[ReadOnly[Annotated[Float[Array, ""], Units("1")]]]  # noqa: F722 - a jaxtyping shape
    handover_window: NotRequired[ReadOnly[Annotated[Float[Array, ""], Units("1")]]]  # noqa: F722 - a jaxtyping shape
    film_annulus: NotRequired[ReadOnly[Annotated[Float[Array, ""], Units("1")]]]  # noqa: F722 - a jaxtyping shape
    film_unsolved: NotRequired[ReadOnly[Annotated[Float[Array, ""], Units("1")]]]  # noqa: F722 - a jaxtyping shape
    film_cutoff: NotRequired[ReadOnly[Annotated[Float[Array, ""], Units("m")]]]  # noqa: F722 - a jaxtyping shape


#: The leaves a node with lubrication's hand-over holds for its near contact's search, made when it is built from the
#: hull's force points as built (``mime.nodes.hydro.mrs._handover.Search``), with their units: as the body's hull
#: contact holds its own (``mime.nodes.body._rigid_inertial.SEARCH_LEAVES``), values of the step, not constants of its
#: program; and the film's reach, the body's size.
FILM_LEAVES = {"film_search_apart_squared": "m^2", "film_search_tolerance": "m", "film_reach": "m"}
#: The state fields a node with the hand-over adds (zero where the film is not evaluated).
FILM_FIELDS = ("film_increment", "film_coupling", "contact_gap", "film_residual", "handover_window", "film_annulus",
               "film_unsolved", "film_cutoff")


def film_state(zero: Array) -> dict[str, Array]:
    """The hand-over's state fields at zero (:data:`FILM_FIELDS`)."""
    return {"film_increment": jnp.zeros((6, 6), dtype=zero.dtype), "film_coupling": jnp.zeros((3, 3), dtype=zero.dtype),
            **{key: zero for key in FILM_FIELDS[2:]}}


def stops_where(accepted: Array, least: Array) -> Array:
    """Where a step on a separated quadrature stops the run: the wall's reads accepted, and the resistance's least
    eigenvalue (each row and column scaled by its diagonal) not above zero, NaN included. Where the reads were refused
    the resistance is NaN already, and the run stops on it as on any state that is not finite."""
    return accepted & jnp.logical_not(least > 0.0)


def on_the_cells(weights: Array, owners: np.ndarray, area: Array) -> Array:
    """A wall quadrature's weights scaled, cell by cell, so that each cell's sum is its area on the separated quadrature
    (``area``): the image integrated over a cell is then the mean of the image over its wall points times the area
    the free part and the wrench take, as the image at the force point is the point's value times it. A coarser wall
    quadrature partitions the surface along other edges, and unscaled its cells' areas would differ from the free
    part's by the edges' quantisation (by tens of per cent at a few points to a cell)."""
    held = jnp.asarray(owners)
    wall_area = _ops.segment_sum(weights, held, num_segments=area.shape[0])
    return weights * (area / wall_area)[held]


def reach_weight(gap: Array) -> Array:
    """A cell's weight on its integrated column, from its force point's distance from the wall in force spacings (in a
    graded run, in the measure :data:`GRADED_WALL_MEASURE` selects): 1 within half of :data:`WALL_REACH`, 0 beyond it,
    and a smoothstep between (continuous with its first derivative)."""
    return smoothstep(gap, WALL_REACH)


def near_wall_image(wall: WallCorrection, point: Corrected, force: Array, points: Array, weights: Array, gap: Array,
                    table: np.ndarray | None, budget: int, chunks: Chunks, parameters: Mapping[str, Any],
                    viscosity: Any, eps_squared: Any) -> tuple[Corrected, Array]:
    """The wall's image with the cells nearest the wall integrated (:func:`cell_image`): the image at the force points
    (``point``, each column times its cell's area) for every cell, and for the ``budget`` cells whose force points lie
    nearest the wall (``gap``, in force spacings) their column blended towards the integrated one by
    :func:`reach_weight`, which is zero beyond :data:`WALL_REACH`; so a cell far from the wall costs what it did, and a
    cell's column is continuous as it crosses the reach and as it enters or leaves the budget. ``points`` and
    ``weights`` are the wall quadrature's (world frame; the weights scaled to the cells' areas), ``table`` each cell's
    points of it (:attr:`MRSDragNode.wall_table`) and ``chunks`` the budget's rows of it flattened; with no ``table``
    every cell is integrated and blended, ``chunks`` the whole quadrature's by its owners. Also whether a cell
    within the reach was left out of the budget. A step with no cell within the reach reads no more than the image at
    the force points (a ``cond``, whose two branches agree there, every weight zero). ``gap`` is in force spacings, or
    in a graded run in the measure :data:`GRADED_WALL_MEASURE` selects (each cell's own size, ``sqrt(area)``, as
    shipped): the switch's mechanics are
    ``mime.nodes.hydro._near_singular``'s, shared with the near field's cross-kernel."""
    return within_reach(gap, WALL_REACH,
                        lambda: _integrated_near(wall, point, force, points, weights, gap, table, budget, chunks,
                                                 parameters, viscosity, eps_squared),
                        lambda: (point, jnp.asarray(False)))


def _integrated_near(wall: WallCorrection, point: Corrected, force: Array, points: Array, weights: Array, gap: Array,
                     table: np.ndarray | None, budget: int, chunks: Chunks, parameters: Mapping[str, Any],
                     viscosity: Any, eps_squared: Any) -> tuple[Corrected, Array]:
    """:func:`near_wall_image` where a cell lies within the reach; where none does, the step takes the image at the
    force points alone and reads nothing more."""
    if table is None:
        integrated = cell_image(wall, force, points, weights, chunks, parameters, viscosity, eps_squared)
        matrix = blended(point.matrix, integrated.matrix, reach_weight(gap))
        return Corrected(matrix, point.accepted & integrated.accepted), jnp.asarray(False)
    chosen, short = nearest_cells(gap, budget, WALL_REACH)
    held, mass = padded(points, weights)
    rows = jnp.asarray(table)[chosen].reshape(-1)
    integrated = cell_image(wall, force, held[rows], mass[rows], chunks, parameters, viscosity, eps_squared)
    columns = (3 * chosen[:, None] + jnp.arange(3)[None, :]).reshape(-1)
    near = point.matrix[:, columns]
    matrix = point.matrix.at[:, columns].set(blended(near, integrated.matrix, reach_weight(gap[chosen])))
    return Corrected(matrix, point.accepted & integrated.accepted), short


#: Why the point forces do not take a body past the band where the resistance turns indefinite.
_NO_WAY_OUT = ("The point forces (quadrature_refinement=1) are no way out: they are indefinite near the wall too "
               "(MIME-ANO-130)")


def indefinite_advice(gap: float | None, spacing: float, clear: float, *, at_start: bool = False) -> str:
    """What to change where the resistance was indefinite, from what the node measured there: the body's least
    distance from the wall, ``gap`` (m, below zero where its hull reached through the wall; ``None`` where unknown), the
    force spacing it ran with (m), and ``clear``, the gap in force spacings outside which the resistance was measured
    definite (:data:`CONFINED_GAP`, :data:`INTEGRATED_GAP`). A finer spacing where one clears the gap; where none can
    (the hull on or through the wall) or none is finer than the one it ran with, a margin that holds the body clear
    at this spacing, so that the advice never repeats the spacing it ran with."""
    away = ("start the body further from the wall, Body(..., initial=State(position=...))" if at_start else
            "hold the body further from the wall, Contact.Projection(margin=...)")
    if gap is not None and gap > 0.0 and gap / clear < spacing:
        finer = gap / clear * 1e3
        return (f"write a force spacing of at most {finer:.3g} mm, the body's least distance from the wall there over "
                f"{clear:g} (closer, the resistance was measured indefinite), Drag.MRS(spacing=mm({finer:.3g})); or "
                f"{away}. {_NO_WAY_OUT}")
    margin = clear * spacing * 1e3
    if gap is not None and gap > 0.0 and spacing > 0.0:
        return (f"the body was {gap * 1e3:.4g} mm from the wall, {gap / spacing:.3g} force spacings, outside the band "
                f"where the resistance was measured indefinite, so no finer spacing than the {spacing * 1e3:.4g} mm "
                f"it ran with follows from that band: write a smaller one, Drag.MRS(spacing=mm({spacing * 5e2:.3g})) "
                f"halves it, and report the case (MIME-ANO-130). {_NO_WAY_OUT}")
    reached = (f"the body's hull reached the wall (its least distance {gap * 1e3:.4g} mm), and no force spacing "
               f"resolves a body on or through it" if gap is not None else "the body's distance from the wall is not "
                                                                            "known")
    hold = (f"start the body's hull at least {margin:.3g} mm from the wall, Body(..., initial=State(position=...))"
            if at_start else f"hold it off the wall with a contact, Contact.Projection(margin=mm({margin:.3g}))")
    return (f"{reached}: {hold}, {clear:g} of the force spacing it ran with ({spacing * 1e3:.4g} mm), outside the "
            f"band where the resistance was measured indefinite; with a smaller margin, write a force spacing of at "
            f"most the margin over {clear:g} as well, Drag.MRS(spacing=...). {_NO_WAY_OUT}")


def run_gib(force_points: float) -> float:
    """The memory a run on a wall's image holds at ``force_points``, in GiB in float64 (:data:`RUN_GIB_PER_POINT_SQUARED`):
    an estimate, fitted to six measurements on one machine."""
    return RUN_GIB_AT_NO_POINTS + RUN_GIB_PER_POINT_SQUARED * force_points ** 2


def least_distance(distances: Array) -> Array:
    """The least of a body's points' distances from a wall, held out of every derivative: a report, which a flag reads
    and no edge does."""
    return _lax.stop_gradient(jnp.min(distances))


def judged(found: Array, accepted: Array, gap: Array, within: Any, state: Mapping[str, Array],
           pose: Array) -> NodeFields:
    """A step's resistance judged before the body takes it, as the node's state with a wall: ``found``, the raw
    ``W^T A^-1 M``, its symmetric part delivered where it is positive definite or where the wall's reads were refused,
    and NaN where it is not (the stop: :func:`stops_where`); its least eigenvalue scaled by its diagonal and the share
    its symmetric part drops, each held out of every derivative (:func:`scaled_least_eigenvalue`, ``asymmetry``); the
    flags ``indefinite``, ``near_wall`` (``gap`` below ``within``, m) and ``asymmetric``; and the stop's record, the
    node's count of steps and, kept from the first step that stopped, its count, pose, gap and eigenvalue. Generic in
    the resistance: any square resistance a solve delivers is judged the same way."""
    dtype = pose.dtype
    held = _lax.stop_gradient(found)
    least, dropped = scaled_least_eigenvalue(held).astype(dtype), asymmetry(held).astype(dtype)
    stopped = stops_where(accepted, least)
    near = jnp.asarray(False) if within is None else gap < within
    steps = state["steps"] + 1.0
    first = stopped & (state["stopped_at"] == 0.0)
    record = jnp.where(first, jnp.concatenate([steps[None], least[None], gap.astype(dtype)[None], pose]),
                       jnp.concatenate([state["stopped_at"][None], state["stopped_least"][None],
                                        state.get("stopped_gap", jnp.zeros((), dtype=dtype))[None],
                                        state["stopped_pose"]]))
    flags = jnp.stack([stopped, near, dropped > CONFINED_ASYMMETRY_LIMIT]).astype(dtype)
    return node_fields(SeparatedDragState(
        resistance=jnp.where(stopped, jnp.nan, (found + found.T) / 2.0).astype(dtype),
        background_load=jnp.zeros(6, dtype=dtype), wall_accepted=accepted, least_eigenvalue=least, asymmetry=dropped,
        indefinite=flags[0], near_wall=flags[1], asymmetric=flags[2], wall_gap=gap.astype(dtype), steps=steps,
        stopped_at=record[0], stopped_least=record[1], stopped_gap=record[2], stopped_pose=record[3:]))


class Judgement(NamedTuple):
    """A step's judgement of its resistance on a separated quadrature with a wall, as numbers: whether the wall's reads
    were accepted, the least eigenvalue each row and column scaled by its diagonal, the share its symmetric part
    drops, and the body's least distance from the wall (m; ``None`` without a wall's distance)."""

    accepted: bool
    least: float
    asymmetry: float
    gap: float | None


class MRSDragInputs(TypedDict, closed=True):
    pose: NotRequired[ReadOnly[Float[Array, "7"]]]


def _array_tree(value: object) -> bool:
    """Whether ``value`` is a registered pytree of arrays (a compiled table), not an array or a tuple of numbers."""
    if hasattr(value, "shape"):
        return False
    leaves, tree = _tree_util.tree_flatten(value)
    return tree.num_nodes > 1 and bool(leaves) and all(hasattr(leaf, "shape") and hasattr(leaf, "dtype")
                                                       for leaf in leaves)


def _path_name(path: Sequence[Any]) -> str:
    """A leaf's path in a pytree as dotted names: ``.axes.0.breaks``."""
    return "".join(f".{getattr(key, 'name', getattr(key, 'idx', getattr(key, 'key', key)))}" for key in path)


def _floating_leaves(name: str, value: object) -> dict[str, Any]:
    """The floating arrays of the pytree ``value``, each by ``name`` and its path."""
    return {f"{name}{_path_name(path)}": leaf for path, leaf in _tree_util.tree_flatten_with_path(value)[0]
            if jnp.issubdtype(leaf.dtype, jnp.floating)}


#: The pytree helpers a node holding a wall's table reads its leaves with, for a node that holds one too.
array_tree, path_name, floating_leaves = _array_tree, _path_name, _floating_leaves


@dataclasses.dataclass(frozen=True)
class Body:
    """A body's discretisation in its own frame, on a separated quadrature: its force points, its quadrature's points
    and weights, each cell's area and centroid, and ``eps^2``."""

    force: Array
    points: Array
    weights: Array
    area: Array
    centroid: Array
    eps_squared: Array


class MRSDragNode(MimeNode):
    """The regularised-Stokeslet resistance (6 x 6, world frame, SI) of a rigid body about its centre of mass at its
    pose, from its hull kind's samples on the named ``patches`` at ``counts`` (``mime.nodes.hydro.mrs``'s
    ``resistance``), with the wall correction ``wall_kind`` added; and the background load, zero (the medium is at
    rest).

    The samples are the kind's, in its own frame, from its settings (each a node constant ``shape_<setting>``): a
    number is a ``params`` leaf, so that a study's derivative in it flows through the samples; a direction and an
    integer are held as built (a part's direction is its frame, and turning the body is its pose). With
    ``sampling_chosen`` and ``sampling_members`` (the kind's ``Sampling`` at the rung's spacing, made when the rung
    built the node), they are that choice among its quadrature at ``counts``; without, the quadrature at ``counts``
    itself, as a kind kept from before quasi-uniform sampling gives them. The choice is held as the counts are: a node
    built again with another value of a setting keeps it, and a study's step moves the chosen samples with the shape,
    never choosing again. They are placed in the body's frame by the part's ``part_rotation`` (rows) and
    ``part_centre``, and in the world by the pose. The blob
    is ``blob`` times the samples' mean spacing. The solve is checkpointed: reverse mode keeps the step's inputs and
    solves again, rather than holding every step's factors.

    The wall's correction is the MRS's for the wall kind ``wall_kind`` ``wall_kind_version``
    (``mime.nodes.hydro.mrs.wall_correction_named``), its constants each ``wall_<name>``; ``wall_accepted`` publishes
    whether every read it made was accepted (inside its table's readable range): where it is false the resistance is
    not a number, and the run stops on it.

    **A wall's data is an argument of the compiled step, not a constant of its program.** A held wall constant that is
    a pytree of arrays (the tube's compiled table, expected near 350 MB for a whole tube) has each of its floating
    arrays as a ``params`` leaf of its own, ``wall_<name>.<path>`` (``wall_table.values``, ``wall_table.axes.0.breaks``),
    declared ``trainable=False``: MADDENING passes every ``params`` leaf to the compiled step as an argument, so a table
    of any size adds nothing to the program but its integer and boolean index arrays (each axis's, a few per knot).
    ``GraphManager.trainable_mask()`` leaves the table out; a gradient or a Fisher matrix over every differentiable
    leaf (``fim`` with its default mask) includes it, so a study passes the trainable mask. The step reads the table
    from those leaves (the constant ``wall_<name>`` rebuilt from them), and a call given no ``params`` reads the
    constant itself.

    **On a separated quadrature** (``quadrature_owners`` given: ``mime.nodes.hydro.mrs._quadrature``), the samples are
    force points and the integral is the kind's quadrature at ``quadrature_counts`` (the force points themselves, one to
    a cell, where the counts are not given), each of its points in the cell of the force point ``quadrature_owners``
    names, a map made once at build and held. The blob is ``blob`` times the force points' mean spacing (the
    quadrature's area over their count). The free part is the same at every pose of a rigid body, so it is made once
    per build, in its frame at unit viscosity and in float64, and held as the ``params`` leaf ``held_in_body_frame``
    (an argument of the compiled step, ``trainable=False``): in free space the resistance itself, turned to the pose
    and scaled by the viscosity at each step, with no solve; with a wall the free part's matrix, turned to the pose,
    with the wall's correction added at the force points (each cell's force placed at its force point, the column
    times the cell's area) and solved at each step. The held value is the step's wherever the leaves it was made from
    (the kind's numbers, the part's placement and the blob) are as built, and it is made afresh from the leaves
    wherever one has moved, so that it is never stale; its derivative in those leaves is the made value's in both cases
    (``held_or_made``). The resistance held is the symmetric part of ``W^T A^-1 M``, which is symmetric only to the
    discretisation's error; the build reports what it drops (``asymmetry``).

    **With a wall, each step judges its resistance before the body takes it** (MIME-ANO-130; :class:`SeparatedDragState`):
    its least eigenvalue each row and column scaled by its diagonal, the share its symmetric part drops and the body's
    least distance from the wall over its quadrature (``WallCorrection.distance``), each held out of every derivative.
    A resistance that is not positive definite where the wall's reads were accepted is replaced by NaN, so that the
    run stops at that step (``indefinite``; :meth:`stop_problem` says why, from the state's record of the first such
    step); ``near_wall`` and ``asymmetric`` are reports against :data:`NEAR_WALL` force spacings and
    :data:`CONFINED_ASYMMETRY_LIMIT`. The build reports the body's distance at its start (:meth:`built_lines`)."""

    meta = NodeMeta(
        algorithm_id="MIME-NODE-108",
        description="The resistance of a rigid body in Stokes flow by the method of regularised Stokeslets "
                    "on its hull kind's samples, with a wall's correction added.",
        governing_equations=EQUATIONS,
        assumptions=("Stokes flow: quasi-steady, the limit of vanishing Reynolds number",
                     "The surface sampled by the kind's quadrature, a point force at each sample, regularised by a blob "
                     "of the samples' mean spacing times b",
                     "On a separated quadrature: a traction constant over each force point's cell, the cell the points "
                     "of the kind's finer quadrature nearest it, the wall's correction taken at the force points",
                     "A medium at rest, so the background load is zero"),
        references=(Reference("Cortez2005", "the regularised Stokeslet and its method of regularised Stokeslets"),
                    Reference("Smith2018", "the nearest-neighbour discretisation, force points apart from the "
                                           "quadrature"),
                    Reference("GallagherSmith2021", "the nearest-neighbour discretisation's error and its ties")),
    )
    symbols = (
        Symbol(r"\mathbf{R}", SymbolKind.STATE, "resistance", "N s/m; N s; N m s", _ADVANCE),
        Symbol(r"\mu", SymbolKind.PARAMETER, "viscosity", "Pa s", _SOLVE),
        Symbol(r"b", SymbolKind.PARAMETER, "blob", "1", _ADVANCE),
        Symbol(r"\mathbf{c}", SymbolKind.PARAMETER, "part_centre", "m", _ADVANCE),
        Symbol(r"\mathbf{P}", SymbolKind.PARAMETER, "part_rotation", "1", _ADVANCE),
        Symbol(r"\mathbf{p}", SymbolKind.INPUT, "pose", "m; 1", _ADVANCE),
        Symbol(r"\mathbf{q}", SymbolKind.INPUT, "pose", "m; 1", _ADVANCE),
    )

    def __init__(self, name: str, timestep: float, *, viscosity: float, blob: float, kind: str, kind_version: str,
                 patches: Sequence[str], counts: Sequence[Sequence[int]], part_centre: Sequence[float],
                 part_rotation: Sequence[float], wall_kind: str, wall_kind_version: str,
                 sampling_chosen: np.ndarray | None = None, sampling_members: np.ndarray | None = None,
                 quadrature_counts: Sequence[Sequence[int]] | None = None,
                 quadrature_owners: np.ndarray | None = None, force_spacing: float | None = None,
                 reference_quadrature_counts: Sequence[Sequence[int]] | None = None,
                 reference_quadrature_owners: np.ndarray | None = None,
                 wall_quadrature_counts: Sequence[Sequence[int]] | None = None,
                 wall_quadrature_owners: np.ndarray | None = None, wall_cells: int | None = None,
                 wall_refinement: int | None = None, wall_refinement_written: int | None = None,
                 graded_patch: np.ndarray | None = None, grading: str | None = None,
                 lubrication: bool = False, handover_spacing: float | None = None,
                 roughness_body: float | None = None, roughness_wall: float | None = None,
                 **constants: object) -> None:
        if len(patches) != len(counts) or not patches:
            raise ValueError(f"patches {patches!r} and counts {counts!r} name one count per patch, and one patch or "
                             f"more")
        held = registered_kind(kind, kind_version)
        settings = {key.removeprefix(SHAPE): value for key, value in constants.items() if key.startswith(SHAPE)}
        wall = wall_correction_named(wall_kind, wall_kind_version)
        given = {key.removeprefix(WALL) for key in constants if key.startswith(WALL)}
        unknown = set(constants) - {f"{SHAPE}{name}" for name in settings} - {f"{WALL}{name}" for name in given}
        if unknown or given != set(wall.parameters):
            raise ValueError(f"the node takes its kind's settings as shape_<setting> and the wall's parameters "
                             f"{wall.parameters} as wall_<parameter>; it was given {sorted(constants)}")
        self._kind = kind_from_settings(kind, kind_version, settings)
        unknown_patches = [patch for patch in patches if patch not in self._kind.patches]
        if unknown_patches:
            raise ValueError(f"{kind} has no patches {unknown_patches}; its patches are {sorted(self._kind.patches)}")
        #: The kind's settings the step takes as leaves: its numbers, not its directions or integers.
        self._traced = traced_settings(held, settings)
        held_counts = tuple(tuple(int(c) for c in count) for count in counts)
        if (sampling_chosen is None) != (sampling_members is None):
            raise ValueError("a sampling's choice is its chosen samples and their cells, both or neither")
        chosen: dict[str, np.ndarray] = {}
        if sampling_chosen is not None and sampling_members is not None:
            chosen = {"sampling_chosen": np.asarray(sampling_chosen, dtype=np.int32),
                      "sampling_members": np.asarray(sampling_members, dtype=np.int32)}
        self._sampling = Sampling(tuple(patches), held_counts, "chosen" if chosen else "quadrature",
                                  chosen.get("sampling_chosen"), chosen.get("sampling_members"))
        if quadrature_counts is not None and quadrature_owners is None:
            raise ValueError("a separated quadrature's counts come with its owner map, quadrature_owners")
        separated: dict[str, object] = {}
        self._owners: np.ndarray | None = None
        self._quadrature: tuple[tuple[int, ...], ...] | None = None
        if quadrature_owners is not None:
            self._owners = np.asarray(quadrature_owners, dtype=np.int32)
            self._owners.flags.writeable = False
            separated["quadrature_owners"] = self._owners
            if quadrature_counts is not None:
                self._quadrature = tuple(tuple(int(c) for c in count) for count in quadrature_counts)
                separated["quadrature_counts"] = self._quadrature
        if force_spacing is not None:
            separated["force_spacing"] = float(force_spacing)
        if (reference_quadrature_counts is None) != (reference_quadrature_owners is None):
            raise ValueError("a reference quadrature is its counts and its owner map, both or neither")
        self._reference: tuple[tuple[tuple[int, ...], ...], np.ndarray] | None = None
        if reference_quadrature_counts is not None and reference_quadrature_owners is not None:
            counts_held = tuple(tuple(int(c) for c in count) for count in reference_quadrature_counts)
            owners_held = np.asarray(reference_quadrature_owners, dtype=np.int32)
            owners_held.flags.writeable = False
            self._reference = (counts_held, owners_held)
            separated["reference_quadrature_counts"] = counts_held
            separated["reference_quadrature_owners"] = owners_held
        if (wall_quadrature_counts is None) != (wall_quadrature_owners is None):
            raise ValueError("a wall quadrature is its counts and its owner map, both or neither")
        self._wall_quadrature: tuple[tuple[tuple[int, ...], ...], np.ndarray] | None = None
        if wall_quadrature_counts is not None and wall_quadrature_owners is not None:
            wall_counts = tuple(tuple(int(c) for c in count) for count in wall_quadrature_counts)
            wall_owners = np.asarray(wall_quadrature_owners, dtype=np.int32)
            wall_owners.flags.writeable = False
            self._wall_quadrature = (wall_counts, wall_owners)
            separated["wall_quadrature_counts"] = wall_counts
            separated["wall_quadrature_owners"] = wall_owners
            separated["wall_cells"] = int(wall_cells) if wall_cells is not None else 0
        if wall_refinement is not None:  # how many times finer the wall quadrature is, and the refinement written
            separated["wall_refinement"] = int(wall_refinement)
        if wall_refinement_written is not None:
            separated["wall_refinement_written"] = int(wall_refinement_written)
        self._patch: np.ndarray | None = None
        if graded_patch is not None:
            if quadrature_owners is None:
                raise ValueError("a graded sampling's patch is judged on a separated quadrature's points "
                                 "(quadrature_owners)")
            self._patch = np.asarray(graded_patch, dtype=bool)
            self._patch.flags.writeable = False
            separated["graded_patch"] = self._patch
            separated["grading"] = str(grading or "")
        self._search: Search | None = None
        if lubrication:
            if quadrature_owners is None or handover_spacing is None:
                raise ValueError("lubrication's hand-over runs on a separated quadrature (quadrature_owners) with the "
                                 "force spacing it hands over at (handover_spacing)")
            if wall.opened is None or wall.region is None:
                raise ValueError(f"the wall kind {wall_kind} {wall_kind_version} gives no opening (WallCorrection."
                                 f"opened, region), so lubrication's hand-over cannot open a contact with it")
            separated.update(lubrication=True, handover_spacing=float(handover_spacing),
                             roughness_body=float(DEFAULT_ROUGHNESS if roughness_body is None else roughness_body),
                             roughness_wall=float(DEFAULT_ROUGHNESS if roughness_wall is None else roughness_wall))
        super().__init__(name, timestep, viscosity=viscosity, blob=blob, kind=kind, kind_version=kind_version,
                         patches=tuple(patches), counts=held_counts, part_centre=tuple(float(c) for c in part_centre),
                         part_rotation=tuple(float(c) for c in part_rotation), wall_kind=wall_kind,
                         wall_kind_version=wall_kind_version, **chosen, **separated,
                         **{key: value for key, value in constants.items()})
        if self._owners is not None:
            self._check_owners(self._owners)
        if self._patch is not None and self._owners is not None and self._patch.shape != self._owners.shape:
            raise ValueError(f"graded_patch names whether each of the {len(self._owners)} points of the separated "
                             f"quadrature lies in the graded patch; it has shape {self._patch.shape}")
        if lubrication:
            self._search = self._film_search(wall)

    @property
    def lubricated(self) -> bool:
        """Whether the node hands its near contact over to the film (``lubrication``): the coarse MRS evaluated with
        the contact opened to the hand-over gap, plus the film's increment (``mime.nodes.hydro.mrs._handover``)."""
        return self._search is not None

    def _kind_parameters(self, p: Mapping[str, Any]) -> dict[str, Any]:
        """The kind's settings as its samples and support read them: each a leaf where it is a number, its directions
        and integers held as built."""
        constants = self.constants()
        return {name: p[f"{SHAPE}{name}"] if name in self._traced else constants[f"{SHAPE}{name}"]
                for name in (key.removeprefix(SHAPE) for key in constants if key.startswith(SHAPE))}

    def _film_search(self, wall: WallCorrection) -> Search:
        """The near contact's search, from the kind's force points in its own frame as built, in float64."""
        constants = self.constants()
        region = cast(Any, wall.region)
        plain = self._kind_parameters(constants)
        with cast(Any, jax).enable_x64(True):
            settings = {name: (jnp.asarray(value, dtype=jnp.float64) if isinstance(value, (float, tuple)) else value)
                        for name, value in plain.items()}
            points = np.asarray(self._sampling.samples(self._kind, settings).points, dtype=np.float64)
        shape = region({name: constants[f"{WALL}{name}"] for name in wall.parameters})[0]
        return search_for(self._kind, plain, points, self._sampling.patches, self._sampling.counts,
                          constants["part_rotation"], constants["part_centre"], shape)

    def _film_values(self) -> dict[str, float]:
        """The search's leaves as the node was built (:data:`FILM_LEAVES`), the tolerance for the run's dtype."""
        from mime.core import run_dtype
        from mime.nodes.body._contact import search_tolerance

        search = cast(Search, self._search)
        apart = {"film_search_apart_squared": search.apart_squared} if search.starts > 1 else {}
        return {**apart, "film_search_tolerance": search_tolerance(run_dtype(), search.size),
                "film_reach": search.size}

    @property
    def graded(self) -> bool:
        """Whether the node's sampling is graded (``graded_patch`` given): its wall's self-image then measures each
        cell's distance as :data:`GRADED_WALL_MEASURE` says, and its step flags ``off_patch``."""
        return self._patch is not None

    @override
    def param_specs(self) -> dict[str, ParamSpec]:
        held = registered_kind(self.constants()["kind"], self.constants()["kind_version"])
        shapes = {f"{SHAPE}{name}": ParamSpec(units=setting_units(held, name)) if name in self._traced else
                  ParamSpec(trainable=False, description=held_as_built(held, name))
                  for name in (key.removeprefix(SHAPE) for key in self.constants() if key.startswith(SHAPE))}
        placed = "held as built: where the part sits in the body, its frame, as a direction of the kind is"
        wall = self._wall()
        walls = {f"{WALL}{name}": ParamSpec(units=wall.units.get(name, "1")) if name not in wall.held else
                 ParamSpec(units=wall.units.get(name, "1"), trainable=False,
                           description="held as built: the wall's frame or its data")
                 for name in wall.parameters}
        data = {key: ParamSpec(units="1", trainable=False, description="held as built: the wall's data, an argument "
                                                                      "of the compiled step")
                for key in self._wall_arguments()}
        made = ("made once per build from the kind's numbers, the part's placement and the blob, in the body's frame at "
                "unit viscosity: no quantity of its own")
        held = {HELD: ParamSpec(units="1", trainable=False, description=made)} if self._owners is not None else {}
        if "force_spacing" in self.constants():
            held["force_spacing"] = ParamSpec(units="m", trainable=False,
                                              description="held as built: the force spacing the step judges the body's "
                                                          "distance from the wall against (near_wall), a report")
        if self.lubricated:
            held["handover_spacing"] = ParamSpec(units="m", trainable=False, description=(
                f"held as built: the force spacing at the near contact (the finest graded one where graded), whose "
                f"{HANDOVER_SPACINGS:g} is the gap the contact is opened to before the coarse MRS is evaluated (a "
                f"numerical choice, measured)"))
            held["roughness_body"] = ParamSpec(units="m", description="the roughness of the body's surface, declared")
            held["roughness_wall"] = ParamSpec(units="m", description="the roughness of the wall's surface, declared")
            held.update({key: ParamSpec(units=FILM_LEAVES[key], trainable=False, description=(
                "made when the node is built, from the hull's force points as built: where the near contact's search "
                "starts and when it stops, and the film's reach; outside every derivative"))
                for key in self._film_values()})
        found = {**super().param_specs(), "viscosity": ParamSpec(units="Pa s"), "blob": ParamSpec(units="1"),
                 "part_centre": ParamSpec(units="m", trainable=False, description=placed),
                 "part_rotation": ParamSpec(units="1", trainable=False, description=placed), **shapes, **walls, **data,
                 **held}
        if self._owners is not None:
            found.update({f"{HELD_FROM}{key}": ParamSpec(
                units=found[key].units, trainable=False,
                description=f"the value of {key} that {HELD} was made at, which the step compares {key} with: no "
                            f"quantity of its own") for key in self.body_leaf_names()})
        return found

    def _wall_trees(self) -> list[str]:
        """The node constants of the wall's held parameters that are pytrees of arrays (a compiled table)."""
        constants = self.constants()
        return [f"{WALL}{name}" for name in sorted(self._wall().held) if _array_tree(constants[f"{WALL}{name}"])]

    def _wall_arguments(self) -> dict[str, Any]:
        """The floating arrays of each held wall pytree, by ``wall_<name>.<path>``: the leaves the compiled step takes
        the wall's data as."""
        constants = self.constants()
        return {key: leaf for name in self._wall_trees()
                for key, leaf in _floating_leaves(name, constants[name]).items()}

    @override
    def params_pytree(self) -> dict[str, Any]:
        inherited: Any = super()
        leaves = cast(dict[str, Any], inherited.params_pytree())
        held = {HELD: as_run_array(self.held_in_body_frame),
                **{f"{HELD_FROM}{key}": leaves[key] for key in self.body_leaf_names()}} \
            if self._owners is not None else {}
        film = {key: as_run_array(value) for key, value in self._film_values().items()} if self.lubricated else {}
        return {**leaves, **self._wall_arguments(), **held, **film}

    @override
    def _step_constants(self, params: Mapping[str, Any] | None) -> dict[str, Any]:
        """The step's constants, each held wall pytree rebuilt from its leaves in ``params`` where given."""
        held = super()._step_constants(params)
        if params is None:
            return held
        for name in self._wall_trees():
            paths, tree = _tree_util.tree_flatten_with_path(held[name])
            held[name] = _tree_util.tree_unflatten(tree, [params.get(f"{name}{_path_name(path)}", leaf)
                                                          for path, leaf in paths])
        arguments = set(self._wall_arguments())
        return {key: value for key, value in held.items() if key not in arguments}

    def _wall(self) -> WallCorrection:
        return wall_correction_named(self.constants()["wall_kind"], self.constants()["wall_kind_version"])

    @override
    def initial_state(self) -> NodeFields:
        if self._owners is None:
            return node_fields(MRSDragState(resistance=as_run_array(jnp.zeros((6, 6))),
                                            background_load=as_run_array(jnp.zeros(6)),
                                            wall_accepted=jnp.asarray(True)))
        zero = as_run_array(jnp.zeros(()))
        resistance, load, accepted = as_run_array(jnp.zeros((6, 6))), as_run_array(jnp.zeros(6)), jnp.asarray(True)
        patch = {"off_patch": zero} if self._patch is not None else {}
        if self._free():
            return node_fields(SeparatedDragState(resistance=resistance, background_load=load, wall_accepted=accepted,
                                                  least_eigenvalue=zero, indefinite=zero, near_wall=zero,
                                                  asymmetric=zero, steps=zero, stopped_at=zero,
                                                  stopped_pose=as_run_array(jnp.zeros(7)), stopped_least=zero)) | patch
        state = node_fields(SeparatedDragState(resistance=resistance, background_load=load, wall_accepted=accepted,
                                               least_eigenvalue=zero, asymmetry=zero, indefinite=zero, near_wall=zero,
                                               asymmetric=zero, steps=zero, stopped_at=zero,
                                               stopped_pose=as_run_array(jnp.zeros(7)), stopped_least=zero))
        if self._wall().distance is None:
            return state | patch
        spaced = "force_spacing" in self.constants()
        return state | {"wall_gap": zero, "stopped_gap": zero} | ({"wall_gap_spacings": zero} if spaced else {}) | (
            {"wall_cells_short": zero} if spaced and self._wall_quadrature is not None else {}) | patch | (
            film_state(zero) if self.lubricated else {})

    @override
    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        return {"pose": node_input("m; 1", (7,))}

    @cached_property
    def sample_count(self) -> int:
        """How many samples the node solves at: its choice's, or its quadrature's at ``counts``."""
        if self._sampling.chosen is not None:
            return len(self._sampling.chosen)
        return int(self.samples(self._step_constants(None)).points.shape[0])

    def samples(self, p: Mapping[str, Any]) -> Samples:
        """The kind's samples on every patch, in the body's frame: points, normals and weights."""
        constants = self.constants()
        parameters = {name: p[f"{SHAPE}{name}"] if name in self._traced else constants[f"{SHAPE}{name}"]
                      for name in (key.removeprefix(SHAPE) for key in constants if key.startswith(SHAPE))}
        found = self._sampling.samples(self._kind, parameters)
        dtype = jnp.asarray(p["viscosity"]).dtype
        turn = jnp.asarray(p["part_rotation"], dtype=dtype).reshape(3, 3)
        centre = jnp.asarray(p["part_centre"], dtype=dtype)
        points = jnp.asarray(found.points, dtype=dtype)
        normals = jnp.asarray(found.normals, dtype=dtype)
        weights = jnp.asarray(found.weights, dtype=dtype)
        return Samples(jnp.matmul(points, turn.T, precision=PRECISION) + centre,
                       jnp.matmul(normals, turn.T, precision=PRECISION), weights)

    # -- the separated quadrature ------------------------------------------------------------------------------------

    @property
    def separated(self) -> bool:
        """Whether the node solves on a separated quadrature (``quadrature_owners`` given)."""
        return self._owners is not None

    def _check_owners(self, owners: np.ndarray) -> None:
        cells = self.sample_count
        points = self.quadrature_count
        if owners.shape != (points,) or owners.min() < 0 or owners.max() >= cells:
            raise ValueError(f"quadrature_owners names a force point for each of the quadrature's {points} points, "
                             f"each of the {cells} force points; it has shape {owners.shape}")

    @cached_property
    def quadrature_count(self) -> int:
        """How many points the separated quadrature has: the kind's at ``quadrature_counts``, or the force points'."""
        if self._quadrature is None:
            return self.sample_count
        return int(self._quadrature_samples(self._step_constants(None)).points.shape[0])

    @cached_property
    def chunks(self) -> Chunks:
        """The quadrature taken in chunks by its force points (``mime.nodes.hydro.mrs._quadrature.Chunks``)."""
        if self._owners is None:
            raise ValueError("a node on its samples alone has no separated quadrature")
        return Chunks.of(self._owners, self.sample_count)

    @cached_property
    def wall_budget(self) -> int:
        """How many cells a step integrates the wall's image over at most, the nearest the wall: ``wall_cells`` where
        given, else :data:`WALL_CELL_SHARE` of the cells, rounded up."""
        held = int(self.constants().get("wall_cells") or 0)
        found = held if held > 0 else math.ceil(WALL_CELL_SHARE * self.sample_count)
        return max(1, min(found, self.sample_count))

    @cached_property
    def wall_capped(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """The wall quadrature with no cell over :data:`WALL_CELL_CAP` times the mean: the points the step reads, as
        indices of the quadrature, each point's index among them, whose weight it adds to, and the cell of each point
        read (``capped_cells``); ``None`` without a wall quadrature."""
        if self._wall_quadrature is None:
            return None
        owners = self._wall_quadrature[1]
        widest = int(np.bincount(owners, minlength=self.sample_count).max())
        most = min(widest, math.ceil(WALL_CELL_CAP * len(owners) / self.sample_count))
        kept, runs = capped_cells(owners, self.sample_count, most)
        return kept, runs, owners[kept]

    @cached_property
    def wall_table(self) -> np.ndarray | None:
        """Each cell's points of the capped wall quadrature (:attr:`wall_capped`), ``(N, widest)`` indices padded with
        their count (a point of weight zero); ``None`` without a wall quadrature."""
        if self.wall_capped is None:
            return None
        owners = self.wall_capped[2]
        order = np.argsort(owners, kind="stable")
        counts = np.bincount(owners, minlength=self.sample_count)
        table = np.full((self.sample_count, int(counts.max())), len(owners), dtype=np.int32)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        for cell in range(self.sample_count):
            table[cell, :counts[cell]] = order[starts[cell]:starts[cell] + counts[cell]]
        return table

    @cached_property
    def wall_every(self) -> bool:
        """Whether the step integrates every cell rather than the budget's: where the budget's cells, each padded to the
        widest, would hold as many points as the whole wall quadrature, the whole costs no more."""
        table, capped = self.wall_table, self.wall_capped
        return table is not None and capped is not None and self.wall_budget * table.shape[1] >= len(capped[2])

    @cached_property
    def wall_chunks(self) -> Chunks | None:
        """The integrated cells' points taken in chunks cut for the wall's image's pairs: every cell's (the capped wall
        quadrature by its cells) where the step integrates them all (:attr:`wall_every`), else the budget's,
        ``wall_budget`` rows of the wall table flattened; ``None`` where the wall's image is taken at the force
        points."""
        table, capped = self.wall_table, self.wall_capped
        if table is None or capped is None:
            return None
        if self.wall_every:
            return Chunks.of(capped[2], self.sample_count, pair_values=self._wall().pair_values)
        rows = np.repeat(np.arange(self.wall_budget, dtype=np.int32), table.shape[1])
        return Chunks.of(rows, self.wall_budget, pair_values=self._wall().pair_values)

    def body_leaf_names(self) -> tuple[str, ...]:
        """The leaves the body-frame quantity is made from: the kind's numbers, the part's placement and the blob."""
        return (*(f"{SHAPE}{name}" for name in self._traced), "part_rotation", "part_centre", "blob")

    # -- what a body model composed from this node reads (the near field's MRS: mime.nodes.hydro.near_field) ----------

    @property
    def owners(self) -> np.ndarray | None:
        """The separated quadrature's owner map (``quadrature_owners``), ``None`` on the point forces."""
        return self._owners

    @property
    def wall_quadrature(self) -> tuple[tuple[tuple[int, ...], ...], np.ndarray] | None:
        """The wall quadrature's counts and owner map, ``None`` where the wall's image is taken at the force points."""
        return self._wall_quadrature

    def built_theta(self) -> dict[str, Any]:
        """The body-frame quantity's leaves as built (:meth:`body_leaf_names`, from the node's constants)."""
        return self._built_theta()

    def quadrature_samples(self, p: Mapping[str, Any], counts: tuple[tuple[int, ...], ...] | None = None) -> Samples:
        """The separated quadrature's points and weights in the body's frame, or the kind's quadrature at
        ``counts``."""
        return self._quadrature_samples(p, counts)

    def _quadrature_samples(self, p: Mapping[str, Any],
                            counts: tuple[tuple[int, ...], ...] | None = None) -> Samples:
        """The separated quadrature's points and weights in the body's frame: the kind's quadrature at
        ``quadrature_counts`` (or at ``counts``), placed as the samples are."""
        counts = counts if counts is not None else self._quadrature
        if counts is None:
            return self.samples(p)
        constants = self.constants()
        parameters = {name: p[f"{SHAPE}{name}"] if name in self._traced else constants[f"{SHAPE}{name}"]
                      for name in (key.removeprefix(SHAPE) for key in constants if key.startswith(SHAPE))}
        found = Sampling(self._sampling.patches, counts, "quadrature").samples(self._kind, parameters)
        dtype = jnp.asarray(p["blob"]).dtype
        turn = jnp.asarray(p["part_rotation"], dtype=dtype).reshape(3, 3)
        centre = jnp.asarray(p["part_centre"], dtype=dtype)
        return Samples(jnp.matmul(jnp.asarray(found.points, dtype=dtype), turn.T, precision=PRECISION) + centre, None,
                       jnp.asarray(found.weights, dtype=dtype))

    def body(self, theta: Mapping[str, Any],
             quadrature: tuple[tuple[tuple[int, ...], ...], np.ndarray] | None = None) -> Body:
        """The body's discretisation in its own frame from the leaves ``theta`` (:meth:`body_leaf_names`), on the
        node's separated quadrature or on ``quadrature`` (its counts and owner map)."""
        p = {**theta, "viscosity": jnp.asarray(theta["blob"])}
        force = self.samples(p).points
        counts, owners = quadrature if quadrature is not None else (None, cast(np.ndarray, self._owners))
        held = self._quadrature_samples(p) if counts is None else self._quadrature_samples(p, counts)
        area, centroid = cell_moments(held.points, held.weights, owners, self.sample_count)
        eps_squared = blob_squared(area, theta["blob"])
        return Body(force, held.points, held.weights, area, centroid, eps_squared)

    def self_interaction(self, theta: Mapping[str, Any]) -> Array:
        """The free part's matrix, ``3N x 3N`` at unit viscosity in the body's frame, from the leaves ``theta``."""
        held = self.body(theta)
        return separated_matrix(held.force, held.points, held.weights, self.chunks, held.eps_squared)

    def free_resistance(self, theta: Mapping[str, Any]) -> Array:
        """The resistance in free space about the body's origin, 6 x 6 at unit viscosity in the body's frame: the
        symmetric part of ``W^T A^-1 M``."""
        held = self.body(theta)
        matrix = separated_matrix(held.force, held.points, held.weights, self.chunks, held.eps_squared)
        found = separated_resistance(matrix, held.force, held.area, held.centroid,
                                     jnp.zeros(3, dtype=held.force.dtype))
        return (found + found.T) / 2.0

    def _free(self) -> bool:
        """Whether the wall's correction declares itself zero at every pose (free space's)."""
        return self._wall().adds_nothing

    def _make(self) -> Any:
        return self.free_resistance if self._free() else self.self_interaction

    @cached_property
    def _made_compiled(self) -> Any:
        """``_make`` compiled once per node: what :meth:`made_again` makes the held value with, so that a build saved
        again compiles it no more."""
        return jit(self._make())

    def _built_theta(self) -> dict[str, Any]:
        constants = self.constants()
        return {key: constants[key] for key in self.body_leaf_names()}

    @cached_property
    def _built(self) -> Built:
        """The body-frame quantity made once, from the node's constants in float64 whatever the run's precision (the
        resistance in free space, the free part's matrix with a wall), and what the build judges of the free part's
        resistance (:class:`Built`). Evaluated where it is first asked for, a step's trace included, and never traced
        (``ensure_compile_time_eval``): it is made from constants."""
        with cast(Any, jax).enable_x64(True), cast(Any, jax).ensure_compile_time_eval():
            theta = {key: jnp.asarray(value, dtype=jnp.float64) for key, value in self._built_theta().items()}
            body = self.body(theta)
            matrix = separated_matrix(body.force, body.points, body.weights, self.chunks, body.eps_squared)
            found = separated_resistance(matrix, body.force, body.area, body.centroid, jnp.zeros(3, dtype=jnp.float64))
            symmetric = np.asarray((found + found.T) / 2.0, dtype=np.float64)
            moved = None
            if self._reference is not None:
                reference = self.body(theta, self._reference)
                held = separated_matrix(reference.force, reference.points, reference.weights,
                                        Chunks.of(self._reference[1], self.sample_count), reference.eps_squared)
                again = separated_resistance(held, reference.force, reference.area, reference.centroid,
                                             jnp.zeros(3, dtype=jnp.float64))
                moved = quadrature_move(symmetric, np.asarray((again + again.T) / 2.0, dtype=np.float64))
            kept = symmetric if self._free() else np.asarray(matrix, dtype=np.float64)
            return Built(kept, float(asymmetry(found)), float(scaled_least_eigenvalue(found)), symmetric, moved)

    @property
    def held_in_body_frame(self) -> np.ndarray:
        """The body-frame quantity made once (``HELD``): the resistance in free space, the free part's matrix with a
        wall, at unit viscosity, in float64."""
        return self._built.held

    @property
    def free_asymmetry(self) -> float:
        """The share of the free part's resistance its symmetric part drops (``asymmetry``), as built."""
        return self._built.asymmetry

    def built_problems(self, start: Mapping[str, Any] | None = None) -> list[Problem]:
        """What the build refuses of the resistance on a separated quadrature: a free part whose symmetric part is not
        positive definite, or whose asymmetry is past :data:`ASYMMETRY_LIMIT` (a gross defect, not a measure of the
        discretisation's error: the point forces are symmetric to rounding whatever they miss); and, with a wall, the
        resistance at the body's starting pose (``start``'s ``pose``), where the wall's reads there are refused or
        where it is not positive definite (the wall's image taken at force points that do not resolve it, a body
        within about half a force spacing of the wall, MIME-ANO-130). A pose reached later is the step's to judge: a
        step whose resistance is indefinite stops the run (:meth:`stop_problem`)."""
        if self._owners is None:
            return []
        empty = empty_cells(self._owners, self.sample_count)
        if empty:
            return [Problem("drag.empty-cell", f"{len(empty)} of the {self.sample_count} force points have no point of "
                                               f"the quadrature nearest them (the first {empty[:5]}), so their cells "
                                               f"are empty and the matrix singular",
                            "write a larger quadrature_refinement: Drag.MRS(quadrature_refinement=...)")]
        built = self._built
        dropped, least = built.asymmetry, built.least
        found: list[Problem] = []
        settings = self._settings_text()
        if not (math.isfinite(dropped) and math.isfinite(least) and least > 0.0):
            found.append(Problem("drag.mrs-resistance", f"the free part's resistance ({settings}) is not positive "
                                                        f"definite: its least eigenvalue, scaled, is {least!r}",
                                 "write a smaller blob or a larger quadrature_refinement: Drag.MRS(blob=..., "
                                 "quadrature_refinement=...)"))
        elif dropped > ASYMMETRY_LIMIT:
            found.append(Problem("drag.mrs-resistance", f"the free part's resistance ({settings}) is symmetric only to "
                                                        f"{dropped:.1e} of itself, past {ASYMMETRY_LIMIT!r}: the "
                                                        f"discretisation does not resolve its blob",
                                 "write a smaller blob or a larger quadrature_refinement: Drag.MRS(blob=..., "
                                 "quadrature_refinement=...)"))
        if not found and built.moved is not None and not built.moved <= QUADRATURE_MOVE_LIMIT:
            reference = sum(int(np.prod(count)) for count in self._reference[0]) if self._reference else 0
            found.append(Problem(
                "drag.mrs-quadrature", f"the free part's resistance ({settings}) moves by {built.moved:.1%} of itself "
                                       f"on the quadrature its blob's rule asks for ({reference} points), past "
                                       f"{QUADRATURE_MOVE_LIMIT:.0%}: the quadrature does not resolve the blob, and that "
                                       f"move estimates its error (a resistance symmetric to rounding, as the point "
                                       f"forces' is, can be this far off)",
                "leave quadrature_refinement out, for its rule's (two over the blob, rounded up), or write a larger "
                "blob: Drag.MRS(blob=...)"))
        if found or self._free() or start is None or "pose" not in start:
            return found
        judged = self.judged_at(start["pose"])
        if not judged.accepted:
            found.append(Problem(
                "drag.mrs-resistance", f"the wall's reads at the body's starting pose were refused ({settings}): a force "
                                       f"point lies outside the wall's data, past the wall or past the table's reach",
                "start the body inside the medium's region, clear of the wall: Body(..., initial=State(position=...))"))
        elif judged.gap is not None and judged.gap < 0.0:
            found.append(Problem(
                "drag.mrs-resistance", f"the body's hull reaches through the wall at its starting pose ({settings}): "
                                       f"its least distance from the wall is {judged.gap * 1e3:.4g} mm, and a sample "
                                       f"through the wall is outside the wall's image",
                "start the body inside the medium's region, clear of the wall: Body(..., initial=State(position=...))"))
        elif not (math.isfinite(judged.least) and judged.least > 0.0):
            found.append(Problem(
                "drag.mrs-resistance", f"the resistance at the body's starting pose, the wall's image in it "
                                       f"({settings}), is not positive definite: its least eigenvalue, each row and "
                                       f"column scaled by its diagonal, is {judged.least:.3g}{self._gap_text(judged.gap)}; "
                                       f"the force points do not resolve the wall's image there (MIME-ANO-130)",
                indefinite_advice(judged.gap, self._force_spacing() or 0.0, self._clear_gap(), at_start=True)))
        return found

    def _clear_gap(self) -> float:
        """The gap to the wall, in force spacings, outside the band where this node's resistance was measured indefinite:
        :data:`INTEGRATED_GAP` with the image integrated over its cells, else :data:`CONFINED_GAP`."""
        return CONFINED_GAP if self._wall_quadrature is None else INTEGRATED_GAP

    def _settings_text(self) -> str:
        return (f"{self.sample_count} force points, a quadrature of {self.quadrature_count} points, blob "
                f"{self.constants()['blob']!r}")

    def _force_spacing(self) -> float | None:
        held = self.constants().get("force_spacing")
        return float(held) if held is not None else None

    def _gap_text(self, gap: float | None) -> str:
        spacing = self._force_spacing()
        if gap is None:
            return ""
        share = f", {gap / spacing:.3g} of its force spacing ({spacing * 1e3:.4g} mm)" if spacing else ""
        return f", the body's least distance from the wall {gap * 1e3:.4g} mm{share}"

    def judged_at(self, pose: Any) -> Judgement:
        """The step's judgement of its resistance at ``pose`` with the node's built leaves: whether the wall's reads
        were accepted, the resistance's least eigenvalue (each row and column scaled by its diagonal), the share its
        symmetric part drops, and the body's least distance from the wall (m; ``None`` without a wall's distance).
        Kept by pose, so that the build's check and its report take one step between them."""
        key = np.asarray(pose, dtype=np.float64).tobytes()
        held = self._judgements.get(key)
        if held is None:
            out = self.update(self.initial_state(), {"pose": as_run_array(pose)}, self.delta_t,
                              params=self.params_pytree())
            held = Judgement(bool(out["wall_accepted"]), float(out["least_eigenvalue"]), float(out["asymmetry"]),
                             float(out["wall_gap"]) if "wall_gap" in out else None)
            self._judgements[key] = held
        return held

    @cached_property
    def _judgements(self) -> dict[bytes, Judgement]:
        return {}

    def built_lines(self, start: Mapping[str, Any] | None, closest_approach: float | None) -> list[str]:
        """What the build reports of the body's distance from the wall (never a refusal): its least distance at the
        start against the force spacing and the near-wall bound (:data:`NEAR_WALL`), and the force spacing the
        experiment's closest approach needs (``closest_approach``, m: the margin a contact holds the body's hull off
        the wall by, ``None`` where nothing holds it off)."""
        spacing = self._force_spacing()
        graded = self._graded_lines()
        if self._owners is None or self._free() or self._wall().distance is None or spacing is None or \
                start is None or "pose" not in start:
            return graded
        judged = self.judged_at(start["pose"])
        lines = [f"the body's least distance from the wall at its start is {(judged.gap or 0.0) * 1e3:.4g} mm, "
                 f"{(judged.gap or 0.0) / spacing:.3g} of the force spacing ({spacing * 1e3:.4g} mm): a step with the "
                 f"body within {NEAR_WALL:g} force spacing of the wall raises near_wall, and a step whose resistance "
                 f"is indefinite stops the run (drag.mrs-indefinite); "
                 f"{self._wall().near_wall_text or NEAR_WALL_MAPPED}"]
        integrated: str | None = None
        if closest_approach is None:
            lines.append("nothing in the experiment holds the body off the wall, so its closest approach is the wall "
                         "itself, which no force spacing clears: the default force spacing is the body's family's")
        elif closest_approach <= 0.0:
            lines.append("the experiment holds the body on the wall (a margin of 0), so its closest approach is the "
                         "wall itself, which no force spacing clears")
        else:
            clear = self._clear_gap()
            needed = closest_approach / clear
            lines.append(f"the experiment's closest approach is {closest_approach * 1e3:.4g} mm, the margin a contact "
                         f"holds the body's hull off the wall by: a force spacing of at most {needed * 1e3:.4g} mm "
                         f"keeps it {clear:g} force spacings or more from the wall, outside the band where the "
                         f"resistance was measured indefinite (a spacing not written takes it, or the family's where "
                         f"that is finer){self._memory_at(self.sample_count * (spacing / needed) ** 2)}; this "
                         f"experiment's is {spacing * 1e3:.4g} mm")
            if self._wall_quadrature is None:
                integrated = (f"with the wall's image integrated over each cell (wall_refinement=2 written) the "
                              f"resistance was measured definite closer in, from {INTEGRATED_GAP:g} force spacings, so "
                              f"a force spacing of at most {closest_approach / INTEGRATED_GAP * 1e3:.4g} mm clears this "
                              f"approach, at a step about as long and less memory: {WALL_IMAGE_MEMORY}; the option on "
                              f"a machine with 16 GB of memory or less, since the default was chosen on a step's time "
                              f"alone, not its memory")
        count, reads = self.sample_count, self.wall_reads
        solve = (f"{count} force points, a dense solve of {3 * count} unknowns (the free part's matrix held, "
                 f"{(3 * count) ** 2 * 8 / 2 ** 20:.4g} MiB in float64)")
        if self._wall().cost_text:
            lines.append(f"what a step costs at this force spacing: {solve}, and the wall's image in closed form, "
                         f"with no table to read: {self._wall().cost_text}")
        else:
            lines.append(f"what a step costs at this force spacing: {solve}, and {reads} reads of the wall's table "
                         f"({self._wall_text()}); about {reads * READ_SECONDS:.2g} s for the reads on the 24-thread "
                         f"laptop CPU it was measured on ({READ_SECONDS * 1e6:.2g} us a read; {STEP_MEASURED}); and "
                         f"about {run_gib(count):.2g} GiB for the run in float64 ({RUN_GIB_BASIS})")
        if integrated is not None:
            lines.append(integrated)
        lines.append(self._image_text())
        return graded + lines + self._film_lines(start)

    def _film_lines(self, start: Mapping[str, Any] | None) -> list[str]:
        """What the build reports of lubrication's hand-over: the hand-over gap against the force spacing and, at the
        start, against the near contact's smaller radius of curvature (the film's measured window), the roughness
        cutoff, and who owns the coupling."""
        if not self.lubricated or start is None or "pose" not in start:
            return []
        leaves = self.params_pytree()
        spacing = float(leaves["handover_spacing"])
        handover = HANDOVER_SPACINGS * spacing
        with cast(Any, jax).ensure_compile_time_eval():
            wall = self._wall()
            found = handed_over(cast(Search, self._search), self._kind, self._kind_parameters(leaves),
                                self._sampling.samples(self._kind, self._kind_parameters(leaves)).points,
                                as_run_array(start["pose"]), wall,
                                {name: leaves.get(f"{WALL}{name}", self.constants()[f"{WALL}{name}"])
                                 for name in wall.parameters}, leaves, leaves["viscosity"])
            radius = 1.0 / max(float(found.contact.k1), 1.0 / float(leaves["film_reach"]))
            cutoff, gap = float(found.cutoff), float(found.contact.gap)
        reach = ("within the film's measured window (about 7 %)" if handover <= WINDOW * radius else
                 f"past the film's measured window, {WINDOW:g} of it (about 7 % within it, 15 to 19 % at force spacings "
                 f"of 0.45 to 0.6 of the radius, about 30 % at 0.9): a step there raises handover_window")
        return [f"lubrication's hand-over: the body's near contact is opened to {handover * 1e3:.4g} mm "
                f"({HANDOVER_SPACINGS:g} of the force spacing, {spacing * 1e3:.4g} mm) before the coarse MRS is "
                f"evaluated, plus the thin film's increment; at the start the contact's gap is {gap * 1e3:.4g} mm and "
                f"its smaller radius of curvature {radius * 1e3:.4g} mm, so the hand-over gap is "
                f"{handover / radius:.3g} of it, {reach}",
                f"lubrication's roughness cutoff: {cutoff * 1e6:.4g} um, the root of the sum of the squares of the "
                f"body's {float(leaves['roughness_body']) * 1e6:.4g} um and the wall's "
                f"{float(leaves['roughness_wall']) * 1e6:.4g} um (each a declaration, never fitted): below it the "
                f"film's gap is floored, its resistance stops growing and a contact carries the gap; the near-wall "
                f"resistance owns the hydrodynamic coupling of rotation and translation down to it (film_coupling "
                f"reports the film's share), a contact none"]

    def _graded_lines(self) -> list[str]:
        """What the build reports of a graded sampling: what it was graded towards, as the drag described it, the
        patch's share of the quadrature's points, and how the wall's self-image measures a cell's distance."""
        if self._patch is None:
            return []
        held = int(np.count_nonzero(self._patch))
        measure = "its own size" if GRADED_WALL_MEASURE == "cell" else "the force spacing"
        flag = (f"; a step with the body within {NEAR_WALL:g} force spacing of the wall and a point within "
                f"{OFF_PATCH_BAND:g} force spacings of its least distance outside the patch raises off_patch (the "
                f"sampling is fixed on the body, so a body that turns carries the patch away); the wall's self-image "
                f"measures each cell's distance in {measure}, and a written wall_refinement is from the finest "
                f"spacing" if not self._free() else "")
        return [f"the force points are graded: {self.constants().get('grading')}; {held} of the {len(self._patch)} "
                f"points of the separated quadrature lie in the graded patch, finer than the force spacing{flag}"]

    def _memory_at(self, force_points: float) -> str:
        """The run's memory at ``force_points`` for the report's closest-approach line, as the tube's fit states it; a
        wall with its own cost figures (``WallCorrection.cost_text``) states them in the cost line instead."""
        if self._wall().cost_text:
            return ""
        return f", at about {run_gib(force_points):.2g} GiB a run in float64 ({RUN_GIB_BASIS})"

    def _image_text(self) -> str:
        """Which image the wall adds (``WallCorrection.regularised``): the regularised image of the drag's own blob, with
        its ``eps``, or the wall's own singular one."""
        wall = self._wall()
        if not wall.regularised:
            return (f"the wall's image ({wall.name}): singular, its own, with no blob in it (the drag's blob is not "
                    f"passed into it), its near-wall behaviour and validity line its own")
        with cast(Any, jax).enable_x64(True), cast(Any, jax).ensure_compile_time_eval():
            theta = {key: jnp.asarray(value, dtype=jnp.float64) for key, value in self._built_theta().items()}
            eps = math.sqrt(float(self.body(theta).eps_squared))
        return (f"the wall's image ({wall.name}): regularised, the drag's blob, eps = {eps:.4g} m (the free part's: "
                f"blob {self.constants()['blob']!r} times the cells' mean spacing), so that it is the exact no-slip "
                f"partner of the free part")

    @property
    def wall_reads(self) -> int:
        """The reads of the wall's table a step makes: every force point against every force point (the image at the
        force points), and with a wall quadrature every force point against the points of the budget's cells, each
        padded to the widest cell, both ways for the reciprocal blend."""
        table, capped = self.wall_table, self.wall_capped
        reads = self.sample_count ** 2
        if table is None or capped is None:
            return reads
        held = len(capped[2]) if self.wall_every else self.wall_budget * table.shape[1]
        return reads + 2 * self.sample_count * held

    def _wall_text(self) -> str:
        table = self.wall_table
        if table is None:
            return "the image at the force points"
        which = "every cell" if self.wall_every else f"the {self.wall_budget} cells nearest the wall"
        held = self.constants()
        used, written = held.get("wall_refinement"), held.get("wall_refinement_written")
        finer = (f" {used} times finer than the {'finest graded spacing' if self.graded else 'force spacing'}"
                 if used is not None else "")
        promoted = (f" (wall_refinement={written} written, which leaves a cell of this body with no wall point, so the "
                    f"next finer that leaves none)" if used is not None and written != used else "")
        return (f"the image at the force points, and where a cell lies within {WALL_REACH:g} force spacings of the "
                f"wall integrated over {which} on a wall quadrature{finer}{promoted}, of up to {table.shape[1]} points "
                f"a cell, blended to the point image by the distance")

    def stop_problem(self, state: Mapping[str, Any]) -> Problem | None:
        """Why the node stopped a run, from its state at the run's end: the step whose resistance turned indefinite
        first, which the node replaced by NaN before the body took it, so that no energy went in; ``None`` where it
        stopped none."""
        if "stopped_at" not in state or not float(np.asarray(state["stopped_at"])) > 0.0:
            return None
        step = int(round(float(np.asarray(state["stopped_at"]))))
        pose = np.asarray(state["stopped_pose"], dtype=np.float64)
        gap = float(np.asarray(state["stopped_gap"])) if "stopped_gap" in state else None
        least = float(np.asarray(state["stopped_least"]))
        position = ", ".join(f"{c * 1e3:.5g}" for c in pose[:3])
        turn = ", ".join(f"{c:.5g}" for c in pose[3:])
        if gap is None:
            return Problem(
                "drag.mrs-indefinite",
                f"its resistance turned indefinite at its step {step} (t = {step * self.delta_t:.6g} s), and the run "
                f"stopped there, before the body took it: its least eigenvalue, each row and column scaled by its "
                f"diagonal, was {least:.3g} ({self._settings_text()}), at the pose ({position}) mm, quaternion "
                f"({turn}), with no wall: the discretisation, as the study's leaves moved it, does not resolve its blob",
                "write a smaller blob or a larger quadrature_refinement: Drag.MRS(blob=..., quadrature_refinement=...)")
        return Problem(
            "drag.mrs-indefinite",
            f"its resistance turned indefinite at its step {step} (t = {step * self.delta_t:.6g} s), and the run "
            f"stopped there, before the body took it: its least eigenvalue, each row and column scaled by its "
            f"diagonal, was {least:.3g} ({self._settings_text()}), at the pose ({position}) mm, quaternion ({turn})"
            f"{self._gap_text(gap)}; the force points do not resolve the wall's image there (MIME-ANO-130)",
            indefinite_advice(gap, self._force_spacing() or 0.0, self._clear_gap()))

    def _held_from(self, p: Mapping[str, Any]) -> dict[str, Any]:
        """The leaves the held value in ``p`` was made at (:data:`HELD_FROM`), by the leaf's own name."""
        return {key: p[f"{HELD_FROM}{key}"] for key in self.body_leaf_names()}

    def _held_value(self, p: Mapping[str, Any]) -> Array:
        """The body-frame quantity at the step's leaves: as its chunk made it, where it did (its run-constant,
        :meth:`run_constants`); else the held one where the leaves are those it was made at, made from them where one
        has moved (``held_or_made``) or where the step was given no held value (a call with no ``params``)."""
        if run_constants_given(p):
            return p[HELD]
        theta = {key: p[key] for key in self.body_leaf_names()}
        if HELD not in p:
            return self._make()(theta)
        return held_or_made(self._make(), self._held_from(p))(theta, p[HELD])

    @override
    def run_constants(self, params: Mapping[str, Any]) -> dict[str, Array]:
        """The held body-frame quantity at the run's leaves (:meth:`_held_value`'s choice), made once per chunk."""
        if HELD not in params:
            return {}
        theta = {key: params[key] for key in self.body_leaf_names()}
        return {HELD: held_or_made(self._make(), self._held_from(params))(theta, params[HELD])}

    @override
    def made_again(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """The held value (``HELD``) made from ``params``' leaves, once, where they are not the build's, with the leaves
        it was made at (:data:`HELD_FROM`): what the save's resolution rerun gives the step, which keeps it there."""
        if HELD not in params:
            return {}
        theta = {key: params[key] for key in self.body_leaf_names()}
        if bool(unchanged(theta, self._built_theta())):
            return {}
        return {HELD: self._made_compiled(theta).astype(jnp.asarray(params[HELD]).dtype),
                **{f"{HELD_FROM}{key}": value for key, value in theta.items()}}

    def _separated_step(self, p: Mapping[str, Any], state: NodeFields, pose: Array) -> NodeFields:
        """The step on a separated quadrature. In free space the held resistance turned to the pose (its asymmetry judged at
        build) and its least eigenvalue judged as with a wall, a study's leaves moving it included. With a wall, the free part's matrix turned, the wall's image added and solved: integrated over each cell on the
        wall quadrature (``wall_quadrature_counts``, :func:`cell_image`) where the node has one, else taken at the force
        points, each column times its cell's area. Then the result judged before the body takes it (:func:`judged`): its
        least eigenvalue each row and column scaled by its diagonal, the share its symmetric part drops, and the body's
        least distance from the wall over its quadrature's points, each held out of every derivative. A resistance that
        is not positive definite where the wall's reads were accepted is replaced by NaN, so that the run stops at that
        step rather than taking energy from the drag (``indefinite``, with the step, pose, gap and eigenvalue recorded
        once, at the first such step: :meth:`stop_problem`); ``near_wall`` and ``asymmetric`` are reports."""
        viscosity = p["viscosity"]
        held = self._held_value(p)
        dtype = pose.dtype
        zero = jnp.zeros((), dtype=dtype)
        patch = {"off_patch": zero} if self._patch is not None else {}
        if self._free():
            found = viscosity * turned_resistance(held, quaternion.rotation_matrix(pose[3:]))
            fields = judged(found.astype(dtype), jnp.asarray(True), zero, None, state, pose)
            return node_fields({key: value for key, value in fields.items()
                                if key not in ("asymmetry", "wall_gap", "stopped_gap")}) | patch
        wall = self._wall()
        theta = {key: p[key] for key in self.body_leaf_names()}
        parameters = {name: p[f"{WALL}{name}"] for name in wall.parameters}
        distance, wall_quadrature, wall_capped, wall_chunks = (wall.distance, self._wall_quadrature, self.wall_capped,
                                                               self.wall_chunks)

        spacing = p.get("force_spacing")
        graded, in_patch = self._patch is not None, self._patch
        own_size = graded and GRADED_WALL_MEASURE == "cell"

        def solved(held: Array, theta: Mapping[str, Any], viscosity: Array, pose: Array,
                   parameters: Mapping[str, Any], actual: Mapping[str, Any] | None = None) -> tuple[Array, ...]:
            # with the hand-over, ``parameters`` are the wall opened and ``actual`` the wall's own, which the body's
            # distance from the wall is reported against
            reported = parameters if actual is None else actual
            turn = quaternion.rotation_matrix(pose[3:])
            body = self.body(theta)
            force = pose[:3] + jnp.matmul(body.force, turn.T, precision=PRECISION)
            centroid = pose[:3] + jnp.matmul(body.centroid, turn.T, precision=PRECISION)
            corrected = wall.correction(force, parameters, viscosity, eps_squared=body.eps_squared)
            image = Corrected(corrected.matrix * jnp.repeat(body.area, 3)[None, :], corrected.accepted)
            short = jnp.asarray(False)
            if wall_quadrature is not None and wall_capped is not None and wall_chunks is not None and \
                    distance is not None and spacing is not None:
                cells = self._quadrature_samples({**theta, "viscosity": jnp.asarray(theta["blob"])},
                                                 wall_quadrature[0])
                kept, runs, capped_owners = wall_capped
                weights = _ops.segment_sum(cells.weights, jnp.asarray(runs), num_segments=len(kept))
                measure = jnp.sqrt(body.area) if own_size else spacing
                image, short = near_wall_image(
                    wall, image, force, pose[:3] + jnp.matmul(cells.points[jnp.asarray(kept)], turn.T,
                                                              precision=PRECISION),
                    on_the_cells(weights, capped_owners, body.area), distance(force, parameters) / measure,
                    None if self.wall_every else cast(np.ndarray, self.wall_table), self.wall_budget, wall_chunks,
                    parameters, viscosity, body.eps_squared)
            found = separated_resistance(turned(held, turn) / viscosity + image.matrix, force, body.area, centroid,
                                         pose[:3])
            points = pose[:3] + jnp.matmul(body.points, turn.T, precision=PRECISION)
            if not graded:
                gap = zero if distance is None else least_distance(distance(points, reported))
                return found, image.accepted, gap, short
            if distance is None:
                return found, image.accepted, zero, short, jnp.asarray(False)
            away = distance(points, reported)
            least = least_distance(away)
            band = jnp.inf if spacing is None else OFF_PATCH_BAND * spacing
            outside = jnp.any(~jnp.asarray(in_patch) & (away <= least + band))
            return found, image.accepted, least, short, outside

        film: dict[str, Array] = {}
        if self._search is not None:
            handover = handed_over(self._search, self._kind, self._kind_parameters(p),
                                   self._sampling.samples(self._kind, self._kind_parameters(p)).points.astype(dtype),
                                   pose, wall, parameters, p, viscosity)
            found, accepted, gap, short, *outside = checkpoint(solved)(held, theta, viscosity, pose, handover.opened,
                                                                       parameters)
            found = found + handover.increment
            film = {"film_increment": handover.increment, "film_coupling": handover.increment[:3, 3:],
                    "contact_gap": _lax.stop_gradient(handover.contact.gap), "film_residual": handover.residual,
                    "handover_window": handover.window, "film_annulus": handover.annulus,
                    "film_unsolved": handover.unsolved, "film_cutoff": _lax.stop_gradient(handover.cutoff)}
        else:
            found, accepted, gap, short, *outside = checkpoint(solved)(held, theta, viscosity, pose, parameters)
        fields = judged(found, accepted, gap, None if distance is None or spacing is None else NEAR_WALL * spacing,
                        state, pose) | film
        if graded:
            within = jnp.asarray(False) if distance is None or spacing is None else gap < NEAR_WALL * spacing
            patch = {"off_patch": jnp.where(within & outside[0], jnp.ones((), dtype=dtype), zero)}
        if distance is None:
            return node_fields({key: value for key, value in fields.items()
                                if key not in ("wall_gap", "stopped_gap")}) | patch
        if spacing is None:
            return node_fields(fields) | patch
        extra = {"wall_gap_spacings": (gap / spacing).astype(dtype)}
        if wall_quadrature is not None:
            extra["wall_cells_short"] = jnp.where(short, jnp.ones((), dtype=dtype), zero)
        return node_fields(fields) | extra | patch

    @override
    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        given = typed_view(MRSDragInputs, boundary_inputs)
        pose = given.get("pose", as_run_array(jnp.asarray((0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))))
        if self._owners is not None:
            return self._separated_step(p, state, pose)
        held = self.samples(p)
        points = pose[:3] + jnp.matmul(held.points, quaternion.rotation_matrix(pose[3:]).T, precision=PRECISION)
        wall = self._wall()
        parameters = {name: p[f"{WALL}{name}"] for name in wall.parameters}

        def solved(points: Array, eps_squared: Array, viscosity: Array, centre: Array,
                   parameters: Mapping[str, Any]) -> tuple[Array, Array]:
            corrected = wall.correction(points, parameters, viscosity, eps_squared=eps_squared)
            return resistance(points, eps_squared, viscosity, centre, corrected.matrix), corrected.accepted

        found, accepted = checkpoint(solved)(points, blob_squared(held.weights, p["blob"]), p["viscosity"], pose[:3],
                                             parameters)
        return node_fields(MRSDragState(resistance=found.astype(pose.dtype),
                                        background_load=jnp.zeros(6, dtype=pose.dtype), wall_accepted=accepted))


__all__ = ["ASYMMETRY_LIMIT", "CONFINED_ASYMMETRY_LIMIT", "CONFINED_GAP", "EQUATIONS", "GRADED_WALL_MEASURE", "HELD", "HELD_FROM", "INTEGRATED_GAP", "NEAR_WALL", "NEAR_WALL_MAPPED",
           "OFF_PATCH_BAND", "QUADRATURE_MOVE_LIMIT", "READ_SECONDS", "RUN_GIB_AT_NO_POINTS", "RUN_GIB_PER_POINT_SQUARED", "SHAPE", "SIGNIFICANT_ENTRY", "WALL", "WALL_CELL_CAP",
           "WALL_CELL_SHARE", "WALL_REACH",
           "Body", "Built", "Judgement", "MRSDragInputs", "MRSDragNode", "MRSDragState", "SeparatedDragState", "array_tree",
           "floating_leaves", "judged", "path_name",
           "kind_from_settings", "least_distance", "near_wall_image", "on_the_cells", "quadrature_move",
           "reach_weight", "run_gib", "stops_where"]
