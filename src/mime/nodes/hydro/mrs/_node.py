"""The MRS resistance of a rigid body: its hull kind's samples, placed by the body's pose, solved for the
resistance about its centre of mass at every step."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from functools import cached_property
from typing import Any, NamedTuple, cast, override

import jax
import numpy as np
from jax import Array
from jaxtyping import Bool, Float
from maddening.core.compliance.metadata import NodeMeta, Reference
from maddening.core.node import BoundaryInputSpec
from maddening.core.params import ParamSpec
from typing_extensions import NotRequired, ReadOnly, TypedDict

from mime.core import MimeNode, NodeFields, Symbol, SymbolKind, as_run_array, jnp, node_fields, node_input, \
    quaternion, typed_view
from mime.nodes.hydro.mrs._quadrature import (
    Chunks,
    asymmetry,
    cell_moments,
    empty_cells,
    held_or_made,
    scaled_least_eigenvalue,
    separated_matrix,
    separated_resistance,
    turned,
    turned_resistance,
)
from mime.nodes.hydro.mrs._solve import PRECISION, blob_squared, resistance
from mime.nodes.hydro.mrs._walls import Corrected, WallCorrection, cell_image, wall_correction_named
from mime.registry import Direction, EntryKind, Geometry, Problem, Registered, Samples, Sampling, lookup
from mime.units import Quantity, unit_symbol

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
#: The gap to a wall, in force spacings, below which the resistance was measured indefinite with the wall's image
#: integrated over each cell (MIME-ANO-130), with room: a confined body's default force spacing keeps its closest
#: approach outside it (``Drag``'s rule), and a refusal or a stop near the wall names the spacing that does. A labelled
#: decision of phase 7m-b's fix round (design #204), the review's figure measured with the image in place: FL-9 in the
#: 1/4 in tube at the screw's defaults and wall_refinement 2, eight rolls at each gap, was indefinite at some roll at
#: 0.15, 0.2 and 0.25 force spacings (the largest 0.25, at 0.6 mm) and definite at every roll at 0.3 and 0.36 (0.6 and
#: 0.829 mm) and at 0.25 at 0.4 mm; the bound keeps a factor of 1.44 over the largest. Definite there is not accurate:
#: at 0.829 mm and 0.36 spacings the speed per turn was 411 to 460 um, against about 600 at script A's frozen point.
CONFINED_GAP = 0.36
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
#: the sphere's at blob 0.25).
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
    background_load: ReadOnly[Float[Array, "6"]]
    wall_accepted: ReadOnly[Bool[Array, ""]]  # noqa: F722 - a jaxtyping shape


class SeparatedDragState(TypedDict, closed=True):
    """The state of a node on a separated quadrature: the resistance, the background load and whether every read of
    the wall's data was accepted, as on the point forces; the step's judgement of its resistance, its least eigenvalue
    each row and column scaled by its diagonal and the share its symmetric part drops, with the flags they raise; and,
    with a wall, the body's least distance from it (m, and in force spacings, the near-wall flag's measure) and the
    stop's record: the node's count of steps, and the step (zero for none), pose, gap and least eigenvalue at which its
    resistance first turned indefinite."""

    resistance: ReadOnly[Float[Array, "6 6"]]  # noqa: F722 - a jaxtyping shape
    background_load: ReadOnly[Float[Array, "6"]]
    wall_accepted: ReadOnly[Bool[Array, ""]]  # noqa: F722 - a jaxtyping shape
    least_eigenvalue: ReadOnly[Float[Array, ""]]  # noqa: F722 - a jaxtyping shape
    asymmetry: ReadOnly[Float[Array, ""]]  # noqa: F722 - a jaxtyping shape
    indefinite: ReadOnly[Float[Array, ""]]  # noqa: F722 - a jaxtyping shape
    near_wall: ReadOnly[Float[Array, ""]]  # noqa: F722 - a jaxtyping shape
    asymmetric: ReadOnly[Float[Array, ""]]  # noqa: F722 - a jaxtyping shape
    wall_gap: NotRequired[ReadOnly[Float[Array, ""]]]  # noqa: F722 - a jaxtyping shape
    wall_gap_spacings: NotRequired[ReadOnly[Float[Array, ""]]]  # noqa: F722 - a jaxtyping shape
    steps: NotRequired[ReadOnly[Float[Array, ""]]]  # noqa: F722 - a jaxtyping shape
    stopped_at: NotRequired[ReadOnly[Float[Array, ""]]]  # noqa: F722 - a jaxtyping shape
    stopped_pose: NotRequired[ReadOnly[Float[Array, "7"]]]
    stopped_gap: NotRequired[ReadOnly[Float[Array, ""]]]  # noqa: F722 - a jaxtyping shape
    stopped_least: NotRequired[ReadOnly[Float[Array, ""]]]  # noqa: F722 - a jaxtyping shape


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


def _registered_kind(kind: str, version: str) -> Registered:
    held = lookup(EntryKind.GEOMETRY, kind, version)
    if held is None or not issubclass(held.cls, Geometry):
        raise ValueError(f"no geometry kind {kind} {version} is registered; import the package that registers it")
    return held


def kind_from_settings(kind: str, version: str, settings: Mapping[str, object]) -> Geometry:
    """The registered geometry kind ``kind`` made again from its settings in SI, as the node holds them."""
    held = _registered_kind(kind, version)
    made: dict[str, object] = {}
    for setting in dataclasses.fields(cast(Any, held.cls)):
        value = settings[setting.name]
        declared = held.settings.get(setting.name)
        if declared is not None and declared.dimension is not None:
            made[setting.name] = Quantity(cast(Any, value), declared.dimension)
        else:
            made[setting.name] = value
    return cast(Geometry, cast(Any, held.cls)(**made))


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
                 wall_quadrature_owners: np.ndarray | None = None, **constants: object) -> None:
        if len(patches) != len(counts) or not patches:
            raise ValueError(f"patches {patches!r} and counts {counts!r} name one count per patch, and one patch or "
                             f"more")
        held = _registered_kind(kind, kind_version)
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
        self._traced = tuple(name for name, value in settings.items()
                             if isinstance(value, float) and _kind_of(held, name) is not Direction)
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
        super().__init__(name, timestep, viscosity=viscosity, blob=blob, kind=kind, kind_version=kind_version,
                         patches=tuple(patches), counts=held_counts, part_centre=tuple(float(c) for c in part_centre),
                         part_rotation=tuple(float(c) for c in part_rotation), wall_kind=wall_kind,
                         wall_kind_version=wall_kind_version, **chosen, **separated,
                         **{key: value for key, value in constants.items()})
        if self._owners is not None:
            self._check_owners(self._owners)

    @override
    def param_specs(self) -> dict[str, ParamSpec]:
        held = _registered_kind(self.constants()["kind"], self.constants()["kind_version"])
        shapes = {f"{SHAPE}{name}": ParamSpec(units=_units_of(held, name)) if name in self._traced else
                  ParamSpec(trainable=False, description="held as built: a direction of the kind is its part's frame")
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
        return {**super().param_specs(), "viscosity": ParamSpec(units="Pa s"), "blob": ParamSpec(units="1"),
                "part_centre": ParamSpec(units="m", trainable=False, description=placed),
                "part_rotation": ParamSpec(units="1", trainable=False, description=placed), **shapes, **walls, **data,
                **held}

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
        held = {HELD: as_run_array(self.held_in_body_frame)} if self._owners is not None else {}
        return {**cast(dict[str, Any], inherited.params_pytree()), **self._wall_arguments(), **held}

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
        if self._free():
            return node_fields(SeparatedDragState(resistance=resistance, background_load=load, wall_accepted=accepted,
                                                  least_eigenvalue=zero, asymmetry=zero, indefinite=zero,
                                                  near_wall=zero, asymmetric=zero))
        state = node_fields(SeparatedDragState(resistance=resistance, background_load=load, wall_accepted=accepted,
                                               least_eigenvalue=zero, asymmetry=zero, indefinite=zero, near_wall=zero,
                                               asymmetric=zero, steps=zero, stopped_at=zero,
                                               stopped_pose=as_run_array(jnp.zeros(7)), stopped_least=zero))
        if self._wall().distance is None:
            return state
        return state | {"wall_gap": zero, "stopped_gap": zero} | (
            {"wall_gap_spacings": zero} if "force_spacing" in self.constants() else {})

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
    def wall_chunks(self) -> Chunks | None:
        """The wall quadrature taken in chunks by its force points, cut for the wall's image's pairs; ``None`` where
        the wall's image is taken at the force points."""
        if self._wall_quadrature is None:
            return None
        return Chunks.of(self._wall_quadrature[1], self.sample_count, pair_values=self._wall().pair_values)

    def body_leaf_names(self) -> tuple[str, ...]:
        """The leaves the body-frame quantity is made from: the kind's numbers, the part's placement and the blob."""
        return (*(f"{SHAPE}{name}" for name in self._traced), "part_rotation", "part_centre", "blob")

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
        finer = self._finer(judged.gap)
        if not judged.accepted:
            found.append(Problem(
                "drag.mrs-resistance", f"the wall's reads at the body's starting pose were refused ({settings}): a force "
                                       f"point lies outside the wall's data, past the wall or past the table's reach",
                "start the body inside the medium's region, clear of the wall: Body(..., initial=State(position=...))"))
        elif not (math.isfinite(judged.least) and judged.least > 0.0):
            found.append(Problem(
                "drag.mrs-resistance", f"the resistance at the body's starting pose, the wall's image in it "
                                       f"({settings}), is not positive definite: its least eigenvalue, each row and "
                                       f"column scaled by its diagonal, is {judged.least:.3g}{self._gap_text(judged.gap)}; "
                                       f"the force points do not resolve the wall's image there (MIME-ANO-130)",
                f"write a force spacing of at most {finer:.3g} mm, the body's least distance from the wall over "
                f"{CONFINED_GAP:g} (closer, the resistance was measured indefinite), Drag.MRS(spacing=mm({finer:.3g})); or "
                f"start the body further from the wall, Body(..., initial=State(position=...)). The point forces "
                f"(quadrature_refinement=1) are no way out: they are indefinite near the wall too (MIME-ANO-130)"))
        return found

    def _finer(self, gap: float | None) -> float:
        """The force spacing (mm) that keeps a body ``gap`` (m) from the wall outside the band where the resistance was
        measured indefinite: the gap over :data:`CONFINED_GAP`; the node's own where the gap is unknown or none."""
        spacing = self._force_spacing() or 0.0
        return (gap / CONFINED_GAP if gap is not None and gap > 0.0 else spacing) * 1e3

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
        symmetric part drops, and the body's least distance from the wall (m; ``None`` without a wall's distance)."""
        out = self.update(self.initial_state(), {"pose": as_run_array(pose)}, self.delta_t, params=self.params_pytree())
        return Judgement(bool(out["wall_accepted"]), float(out["least_eigenvalue"]), float(out["asymmetry"]),
                         float(out["wall_gap"]) if "wall_gap" in out else None)

    def built_lines(self, start: Mapping[str, Any] | None, closest_approach: float | None) -> list[str]:
        """What the build reports of the body's distance from the wall (never a refusal): its least distance at the
        start against the force spacing and the near-wall bound (:data:`NEAR_WALL`), and the force spacing the
        experiment's closest approach needs (``closest_approach``, m: the margin a contact holds the body's hull off
        the wall by, ``None`` where nothing holds it off)."""
        spacing = self._force_spacing()
        if self._owners is None or self._free() or self._wall().distance is None or spacing is None or \
                start is None or "pose" not in start:
            return []
        judged = self.judged_at(start["pose"])
        lines = [f"the body's least distance from the wall at its start is {(judged.gap or 0.0) * 1e3:.4g} mm, "
                 f"{(judged.gap or 0.0) / spacing:.3g} of the force spacing ({spacing * 1e3:.4g} mm): a step with the "
                 f"body within {NEAR_WALL:g} force spacing of the wall raises near_wall, and a step whose resistance "
                 f"is indefinite stops the run (drag.mrs-indefinite); every indefinite resistance mapped lay within "
                 f"half a force spacing of the wall (MIME-ANO-130)"]
        if closest_approach is None:
            lines.append("nothing in the experiment holds the body off the wall, so its closest approach is the wall "
                         "itself, which no force spacing clears: the default force spacing is the body's family's")
        else:
            lines.append(f"the experiment's closest approach is {closest_approach * 1e3:.4g} mm, the margin a contact "
                         f"holds the body's hull off the wall by: a force spacing of at most "
                         f"{closest_approach / CONFINED_GAP * 1e3:.4g} mm keeps it {CONFINED_GAP:g} force spacings or "
                         f"more from the wall, outside the band where the resistance was measured indefinite (a "
                         f"spacing not written takes it, or the family's where that is finer); this experiment's is "
                         f"{spacing * 1e3:.4g} mm")
        count, reads = self.sample_count, self.wall_reads
        lines.append(f"what a step costs at this force spacing: {count} force points, a dense solve of {3 * count} "
                     f"unknowns (the free part's matrix held, {(3 * count) ** 2 * 8 / 2 ** 20:.4g} MiB in float64), "
                     f"and {reads} reads of the wall's table ({self._wall_text()}); about {reads * READ_SECONDS:.2g} s "
                     f"for the reads on the 24-thread laptop CPU the fix round of phase 7m-b measured "
                     f"({READ_SECONDS * 1e6:.2g} us a read)")
        return lines

    @property
    def wall_reads(self) -> int:
        """The reads of the wall's table a step makes: every force point against every force point with the image at
        the force points, else against every point of the wall quadrature, both ways for the reciprocal blend."""
        if self._wall_quadrature is None:
            return self.sample_count ** 2
        return 2 * self.sample_count * len(self._wall_quadrature[1])

    def _wall_text(self) -> str:
        if self._wall_quadrature is None:
            return "the image at the force points"
        return (f"the image integrated over each cell on a wall quadrature of {len(self._wall_quadrature[1])} "
                f"points")

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
        finer = self._finer(gap)
        return Problem(
            "drag.mrs-indefinite",
            f"its resistance turned indefinite at its step {step} (t = {step * self.delta_t:.6g} s), and the run "
            f"stopped there, before the body took it: its least eigenvalue, each row and column scaled by its "
            f"diagonal, was {least:.3g} ({self._settings_text()}), at the pose ({position}) mm, quaternion ({turn})"
            f"{self._gap_text(gap)}; the force points do not resolve the wall's image there (MIME-ANO-130)",
            f"write a force spacing of at most {finer:.3g} mm, the body's least distance from the wall there over "
            f"{CONFINED_GAP:g} (closer, the resistance was measured indefinite), Drag.MRS(spacing=mm({finer:.3g})); or "
            f"hold the body further from the wall, Contact.Projection(margin=...). The point forces "
            f"(quadrature_refinement=1) are no way out: they are indefinite near the wall too (MIME-ANO-130)")

    def _held_value(self, p: Mapping[str, Any]) -> Array:
        """The body-frame quantity at the step's leaves: the held one where they are as built (``held_or_made``),
        made from them where one has moved or where the step was given no held value (a call with no ``params``)."""
        theta = {key: p[key] for key in self.body_leaf_names()}
        if HELD not in p:
            return self._make()(theta)
        return held_or_made(self._make(), self._built_theta())(theta, p[HELD])

    def _separated_step(self, p: Mapping[str, Any], state: NodeFields, pose: Array) -> NodeFields:
        """The step on a separated quadrature. In free space the held resistance turned to the pose, judged at build.
        With a wall, the free part's matrix turned, the wall's image added and solved: integrated over each cell on the
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
        if self._free():
            found = viscosity * turned_resistance(held, quaternion.rotation_matrix(pose[3:]))
            built = self._built
            return node_fields(SeparatedDragState(
                resistance=found.astype(dtype), background_load=jnp.zeros(6, dtype=dtype),
                wall_accepted=jnp.asarray(True), least_eigenvalue=jnp.asarray(built.least, dtype=dtype),
                asymmetry=jnp.asarray(built.asymmetry, dtype=dtype), indefinite=zero, near_wall=zero, asymmetric=zero))
        wall = self._wall()
        theta = {key: p[key] for key in self.body_leaf_names()}
        parameters = {name: p[f"{WALL}{name}"] for name in wall.parameters}
        distance, wall_quadrature, wall_chunks = wall.distance, self._wall_quadrature, self.wall_chunks

        def solved(held: Array, theta: Mapping[str, Any], viscosity: Array, pose: Array,
                   parameters: Mapping[str, Any]) -> tuple[Array, Array, Array]:
            turn = quaternion.rotation_matrix(pose[3:])
            body = self.body(theta)
            force = pose[:3] + jnp.matmul(body.force, turn.T, precision=PRECISION)
            centroid = pose[:3] + jnp.matmul(body.centroid, turn.T, precision=PRECISION)
            if wall_quadrature is None or wall_chunks is None:
                corrected = wall.correction(force, parameters, viscosity)
                image = Corrected(corrected.matrix * jnp.repeat(body.area, 3)[None, :], corrected.accepted)
            else:
                cells = self._quadrature_samples({**theta, "viscosity": jnp.asarray(theta["blob"])},
                                                 wall_quadrature[0])
                image = cell_image(wall, force, pose[:3] + jnp.matmul(cells.points, turn.T, precision=PRECISION),
                                   on_the_cells(cells.weights, wall_quadrature[1], body.area), wall_chunks,
                                   parameters, viscosity)
            found = separated_resistance(turned(held, turn) / viscosity + image.matrix, force, body.area, centroid,
                                         pose[:3])
            points = pose[:3] + jnp.matmul(body.points, turn.T, precision=PRECISION)
            return found, image.accepted, zero if distance is None else least_distance(distance(points, parameters))

        found, accepted, gap = checkpoint(solved)(held, theta, viscosity, pose, parameters)
        spacing = p.get("force_spacing")
        fields = judged(found, accepted, gap, None if distance is None or spacing is None else NEAR_WALL * spacing,
                        state, pose)
        if distance is None:
            return node_fields({key: value for key, value in fields.items() if key not in ("wall_gap", "stopped_gap")})
        if spacing is None:
            return node_fields(fields)
        return node_fields(fields) | {"wall_gap_spacings": (gap / spacing).astype(dtype)}

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
            corrected = wall.correction(points, parameters, viscosity)
            return resistance(points, eps_squared, viscosity, centre, corrected.matrix), corrected.accepted

        found, accepted = checkpoint(solved)(points, blob_squared(held.weights, p["blob"]), p["viscosity"], pose[:3],
                                             parameters)
        return node_fields(MRSDragState(resistance=found.astype(pose.dtype),
                                        background_load=jnp.zeros(6, dtype=pose.dtype), wall_accepted=accepted))


def _kind_of(held: Registered, name: str) -> object:
    declared = held.settings.get(name)
    return None if declared is None else declared.kind


def _units_of(held: Registered, name: str) -> str:
    declared = held.settings.get(name)
    return "1" if declared is None or declared.dimension is None else unit_symbol(declared.dimension)


__all__ = ["ASYMMETRY_LIMIT", "CONFINED_ASYMMETRY_LIMIT", "CONFINED_GAP", "EQUATIONS", "HELD", "NEAR_WALL",
           "QUADRATURE_MOVE_LIMIT", "READ_SECONDS", "SHAPE", "SIGNIFICANT_ENTRY", "WALL", "Body", "Built", "Judgement",
           "MRSDragInputs", "MRSDragNode", "MRSDragState", "SeparatedDragState", "judged", "kind_from_settings",
           "least_distance", "on_the_cells", "quadrature_move", "stops_where"]
