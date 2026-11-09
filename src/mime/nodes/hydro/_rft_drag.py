"""Resistive force theory: a slender body's resistance from local drag coefficients along its centreline, with no
interaction between its segments (Gray & Hancock 1955; Lighthill 1976).

**The model.** Each point of the centreline, read at ``counts`` points a segment (each the middle of its share of the
segment's arc, ``ds = L / N``), pushes the fluid with ``f = (xi_par t t^T + xi_perp (I - t t^T)) (v - u) ds``, ``t``
its tangent and ``v`` its velocity in the body's rigid motion, ``u`` the medium's (zero: a medium at rest). In free
space the body's resistance is the sum of those local terms, ``R = sum_i M_i^T B_i M_i`` (``M_i`` the point's rigid
motions about the centre of mass, ``B_i`` its local drag), positive semidefinite by construction. **With a wall** the
points' forces reach one another through the wall's image ``G`` (the wall kind's correction at the points, per unit
point force), so ``(I + B G) f = B M`` and ``R = M^T f``: segments still feel no interaction through the free fluid,
only through the wall, and a body whose local drag is large against its true resistance (a vessel-sized helix) can turn
this resistance indefinite, which the step judges.

**The coefficients**, by the setting ``coefficients``, each from the medium's viscosity and the segment's numbers as JAX
values (never at build): ``lighthill`` (Lighthill 1976, as the literature quotes it), ``xi_par = 2 pi mu / ln(2 q /
r)``, ``xi_perp = 4 pi mu / (ln(2 q / r) + 1/2)``, ``q = 0.09 Lambda``, ``Lambda`` the segment's wavelength (a helix's
arc length a turn) and ``r`` its filament radius; ``spheroid``, a slender prolate spheroid's leading order,
``xi_par = 2 pi mu / (ln(L / r) - 1/2)``, ``xi_perp = 4 pi mu / (ln(L / r) + 1/2)``, ``L`` the centreline's whole
length, which gives a prolate spheroid's translational resistance to leading order in ``1 / ln(L / r)``.

**Judged at every step**, held out of every derivative: the resistance's least eigenvalue, each row and column scaled
by its diagonal. A straight body's (``SEMIDEFINITE_KINDS``: the capsule and the prolate spheroid) is judged against minus
the semidefinite floor of the run's dtype, a zero diagonal scaled by its block's largest (``floored_least_eigenvalue``):
its spin about itself meets no drag, so the resistance is semidefinite and its least eigenvalue that spin's rounding.
Every other kind's (a helix's) is definite, and is judged as the MRS rung judges, against zero (``scaled_least_eigenvalue``).
Below the line, where the wall's reads were accepted, the delivery is
replaced by NaN so that the run stops there (``indefinite``; ``on_indefinite="flag"`` delivers it and raises the flag
alone, the form a differentiated run takes)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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

from mime.core import MimeNode, NodeFields, Symbol, SymbolKind, Units, as_run_array, jnp, node_fields, node_input, \
    quaternion, typed_view
from mime.nodes.hydro.mrs._node import SHAPE, WALL, array_tree, floating_leaves, path_name
from mime.nodes.hydro.mrs._quadrature import floored_least_eigenvalue, scaled_least_eigenvalue, semidefinite_floor
from mime.nodes.hydro.mrs._solve import PRECISION, force_sums, rigid_motions
from mime.nodes.hydro.mrs._walls import WallCorrection, wall_correction_named
from mime.nodes._kinds import kind_from_settings, registered_kind, setting_units, traced_settings
from mime.registry import Centreline, Problem, centreline_points
from mime.units import Magnitude

_tree_util: Any = cast(Any, jax).tree_util
_lax: Any = cast(Any, jax).lax

EQUATIONS = (r"\mathbf{R} = \mathbf{M}^{\top} (\mathbf{I} + \mathbf{B} \mathbf{G})^{-1} \mathbf{B} \mathbf{M}, \quad "
             r"\mathbf{B}_{i} = \left(\xi_{\parallel} \mathbf{t}_{i} \mathbf{t}_{i}^{\top} + \xi_{\perp} "
             r"(\mathbf{I} - \mathbf{t}_{i} \mathbf{t}_{i}^{\top})\right) \Delta s")
_ADVANCE = "mime.nodes.hydro._rft_drag.RFTDragNode.advance"
#: The coefficient sets, by name: what ``coefficients`` may name.
COEFFICIENT_SETS = ("lighthill", "spheroid")
#: Lighthill's ``q`` as a share of the wavelength along the filament.
LIGHTHILL_Q = 0.09
#: What the step does where the resistance is indefinite: stop the run (NaN), or deliver it and flag it.
ON_INDEFINITE = ("stop", "flag")
#: The rft rung's default spacing of its points along the centreline, in filament radii: the integrand of its
#: resistance is smooth along the centreline (a helix's turn spans many radii), so the midpoint rule converges long
#: before this; a convention, not a measurement of the rung's error, which its validity lines state.
RFT_POINTS_PER_RADIUS = 1.0


@dataclass(frozen=True)
class Measured:
    """Where one of the rung's validity lines holds: a kind's family as its misses against the MRS rung on the same
    bodies were measured (``docs/data/slender_validity``, the record): its shape's ratios, each held to
    ``SHAPE_TOLERANCE`` (a helix's pitch over its radius and its turns), the range of its slenderness (its centreline's
    length over its filament radius) the points spanned, the wall kinds measured, and with a wall the gap measured
    between the filament and the wall, in filament radii: the one gap, held to ``GAP_TOLERANCE``, since a wider or a
    narrower one was not measured. Each line is the largest miss measured over those
    points, an estimate against the MRS at its resolution, never a bound: the swimming speed's (``None`` for a straight
    body, which does not swim) and the resistance's largest entry, each row and column scaled by the MRS's diagonal."""

    kind: str
    shape: Mapping[str, float]
    slenderness: tuple[float, float]
    walls: tuple[str, ...]
    gap: float
    speed: float | None
    resistance: float


#: The kinds whose line model's resistance is semidefinite, a straight centreline's spin about itself meeting no drag: their
#: least eigenvalue is judged at a floor. Declared here per kind; the near-field contract's own field is the later home.
SEMIDEFINITE_KINDS = ("capsule", "ellipsoid")
#: How close a body's shape ratios must be to a measured family's for its line to hold.
SHAPE_TOLERANCE = 1e-6
#: How close the gap between the filament and a wall must be, in shares of the measured gap, for a family's line to hold
#: (declared, not fitted: half a filament radius at the 44 radii measured).
GAP_TOLERANCE = 1e-2
#: The families the rung's validity lines were measured over, by the rule the spec states (each line the largest miss
#: over its points, rounded up to two significant figures; ``docs/data/slender_validity``, held to the record by
#: ``tests/oracles/test_rft_validity.py``), against the MRS rung on the same bodies (a prolate spheroid's against its
#: closed form). With the tube's table, the helices at gaps of 19, 10.7 and 7.3 filament radii were indefinite (the
#: rung refuses them) and claim no line.
MEASURED: tuple[Measured, ...] = (
    Measured("capsule", {}, (35.33333333333333, 68.66666666666667), ('free_space',), 0.0, None, 1.0),  # rod_l35, rod_l69
    Measured("ellipsoid", {}, (20.0, 200.0), ('free_space',), 0.0, None, 1.0),  # spheroid_ab10, spheroid_ab30, spheroid_ab100
    Measured("helix", {"pitch/radius": 2.0, "turns": 2.0}, (131.8763323790246, 131.8763323790246), ('free_space',), 0.0, 0.28, 1.4),  # tight_rf030
    Measured("helix", {"pitch/radius": 4.0, "turns": 2.5}, (46.55239722796466, 372.4191778237173), ('free_space',), 0.0, 0.11, 2.1),  # p4_rf060, p4_rf030, p4_rf015, p4_rf0075
    Measured("helix", {"pitch/radius": 4.0, "turns": 2.5}, (93.10479445592932, 93.10479445592932), ('tube',), 44.0, 0.12, 15.0),  # tube_k012
    Measured("helix", {"pitch/radius": 8.0, "turns": 1.5}, (76.29325651881105, 76.29325651881105), ('free_space',), 0.0, 0.065, 0.7),  # open_rf030
)


def shape_ratios(kind: str, settings: Mapping[str, Any]) -> dict[str, Any]:
    """The ratios a family of ``kind`` is held to: a helix's pitch over its radius and its turns; none for another."""
    if kind == "helix":
        return {"pitch/radius": settings["pitch"] / settings["radius"], "turns": settings["turns"]}
    return {}


def measured_family(kind: str, settings: Mapping[str, Any], slenderness: Any, wall: str) -> tuple[Array, Array]:
    """Whether the body lies in a family the lines were measured over (the first in ``MEASURED`` that holds), and with a
    wall the gap that family was measured at (zero in free space or in none), as JAX values of its settings and its
    slenderness: the step judges them at the leaves it is given, so that a study's member, a leaf written into a
    compiled chunk and a derivative's perturbed build are each judged at their own, and its program is the same at
    every value (the two-builds check)."""
    ratios = shape_ratios(kind, settings)
    held, gap = jnp.asarray(False), jnp.zeros_like(jnp.asarray(slenderness))
    for family in reversed(MEASURED):
        if family.kind != kind or wall not in family.walls or set(family.shape) != set(ratios):
            continue
        low, high = family.slenderness
        holds = (low * (1.0 - SHAPE_TOLERANCE) <= slenderness) & (slenderness <= high * (1.0 + SHAPE_TOLERANCE))
        for key, value in family.shape.items():
            holds = holds & (jnp.abs(ratios[key] - value) <= SHAPE_TOLERANCE * abs(value))
        held, gap = held | holds, jnp.where(holds, 0.0 if wall == "free_space" else family.gap, gap)
    return held, gap


def drag_coefficients(name: str, viscosity: Any, filament_radius: Any, wavelength: Any, length: Any) -> tuple[Any, Any]:
    """``xi_par`` and ``xi_perp`` (N s/m^2, the force a unit length per unit velocity along and across the tangent) of
    the set ``name``, from the viscosity, the filament's radius, the segment's wavelength (``lighthill``) and the
    centreline's whole length (``spheroid``), as JAX values."""
    if name == "lighthill":
        held = jnp.log(2.0 * LIGHTHILL_Q * wavelength / filament_radius)
        return 2.0 * math.pi * viscosity / held, 4.0 * math.pi * viscosity / (held + 0.5)
    if name == "spheroid":
        held = jnp.log(length / filament_radius)
        return 2.0 * math.pi * viscosity / (held - 0.5), 4.0 * math.pi * viscosity / (held + 0.5)
    raise ValueError(f"coefficients {name!r} is not one of {COEFFICIENT_SETS}")


def coefficient_problem(name: str, centreline: Centreline) -> str | None:
    """Why the set ``name`` cannot be taken on ``centreline``: a set that reads a wavelength on a segment that declares
    none, or a segment with no filament radius; ``None`` where it can."""
    if name not in COEFFICIENT_SETS:
        return f"coefficients={name!r} is not one of {COEFFICIENT_SETS}"
    if any(segment.filament_radius is None for segment in centreline.segments):
        return "a segment of its centreline declares no filament radius, which every coefficient set reads"
    if name == "lighthill" and any(segment.wavelength is None for segment in centreline.segments):
        return ("Lighthill's coefficients read each segment's wavelength (a helix's arc length a turn), and a segment "
                "of its centreline declares none (a straight one): write coefficients='spheroid'")
    return None


def default_coefficients(centreline: Centreline) -> str:
    """Lighthill's where every segment declares a wavelength, else the slender spheroid's."""
    return "lighthill" if all(segment.wavelength is not None for segment in centreline.segments) else "spheroid"


class RFTDragState(TypedDict, closed=True):
    resistance: ReadOnly[Float[Array, "6 6"]]  # noqa: F722 - a jaxtyping shape
    background_load: ReadOnly[Float[Array, "6"]]
    wall_accepted: ReadOnly[Annotated[Bool[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    least_eigenvalue: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    indefinite: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    unmeasured: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    steps: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    stopped_at: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    stopped_least: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    stopped_pose: ReadOnly[Annotated[Float[Array, "7"], Units("m; 1")]]  # noqa: F722 - a jaxtyping shape


class RFTDragInputs(TypedDict, closed=True):
    pose: NotRequired[ReadOnly[Float[Array, "7"]]]


class RFTDragNode(MimeNode):
    """The resistive-force-theory resistance (6 x 6, world frame, SI) of a rigid slender body about its centre of mass
    at its pose, from its hull kind's centreline read at ``counts`` points a segment, with the wall kind
    ``wall_kind``'s correction between the points; and the background load, zero (a medium at rest).

    The kind's settings are node constants ``shape_<setting>``, its numbers ``params`` leaves (so a study's derivative
    in them flows through the centreline and the coefficients), its directions held as built; the centreline is placed
    in the body by ``part_rotation`` (rows) and ``part_centre`` and in the world by the pose. The wall's constants are
    ``wall_<name>``, a held pytree's floating arrays each a ``params`` leaf, as the MRS node holds them. The
    coefficients are the set ``coefficients`` (:func:`drag_coefficients`), computed in the step from the viscosity and
    the kind's leaves. The step judges its resistance (the module's docstring): ``indefinite`` stops the run (NaN)
    unless ``on_indefinite`` is ``"flag"``; and whether the body lies within the points its validity lines were
    measured over, from its leaves (:meth:`measured_here`; the ``spacing`` the counts were made with is one), so that
    ``unmeasured`` is no value of the build's."""

    meta = NodeMeta(
        algorithm_id="MIME-NODE-126",
        description="The resistance of a rigid slender body in Stokes flow by resistive force theory: local drag "
                    "coefficients along its centreline, with a wall's image between its points.",
        governing_equations=EQUATIONS,
        assumptions=("Stokes flow: quasi-steady, the limit of vanishing Reynolds number",
                     "A slender body: its drag local along its centreline, no interaction between its segments "
                     "through the free fluid",
                     "A medium at rest, so the background load is zero",
                     "A wall reaches the points through its image at each point, per unit point force"),
        limitations=("Its segments do not interact, so a body coiled on itself (a helix of several turns) is resisted "
                     "more than it is: measured against the MRS in the rung's validity lines, never quoted",
                     "With a wall, a body whose local drag is large against its true resistance can turn the "
                     "resistance indefinite; the step judges it and stops the run there"),
        references=(Reference("Lighthill1976", "the local drag coefficients along a slender filament"),
                    Reference("GrayHancock1955", "resistive force theory: drag local along the filament")),
    )
    symbols = (Symbol(r"\mathbf{R}", SymbolKind.STATE, "resistance", "N s/m; N s; N m s", _ADVANCE),)

    def __init__(self, name: str, timestep: float, *, viscosity: float, kind: str, kind_version: str,
                 counts: Sequence[int], part_centre: Sequence[float], part_rotation: Sequence[float],
                 coefficients: str, wall_kind: str, wall_kind_version: str, spacing: float,
                 on_indefinite: str = "stop", **constants: object) -> None:
        if coefficients not in COEFFICIENT_SETS:
            raise ValueError(f"coefficients {coefficients!r} is not one of {COEFFICIENT_SETS}")
        if on_indefinite not in ON_INDEFINITE:
            raise ValueError(f"on_indefinite {on_indefinite!r} is not one of {ON_INDEFINITE}")
        held = registered_kind(kind, kind_version)
        settings = {key.removeprefix(SHAPE): value for key, value in constants.items() if key.startswith(SHAPE)}
        wall = wall_correction_named(wall_kind, wall_kind_version)
        given = {key.removeprefix(WALL) for key in constants if key.startswith(WALL)}
        unknown = set(constants) - {f"{SHAPE}{name}" for name in settings} - {f"{WALL}{name}" for name in given}
        if unknown or given != set(wall.parameters):
            raise ValueError(f"the node takes its kind's settings as shape_<setting> and the wall's parameters "
                             f"{wall.parameters} as wall_<parameter>; it was given {sorted(constants)}")
        self._kind = kind_from_settings(kind, kind_version, settings)
        self._traced = traced_settings(held, settings)
        self._centreline = self._kind.centreline(cast(Mapping[str, Magnitude], settings))
        held_counts = tuple(int(c) for c in counts)
        if len(held_counts) != len(self._centreline.segments) or min(held_counts) < 1:
            raise ValueError(f"counts {held_counts} name one count of one or more for each of the centreline's "
                             f"{len(self._centreline.segments)} segments")
        problem = coefficient_problem(coefficients, self._centreline)
        if problem is not None:
            raise ValueError(problem)
        super().__init__(name, timestep, viscosity=viscosity, kind=kind, kind_version=kind_version,
                         counts=held_counts, part_centre=tuple(float(c) for c in part_centre),
                         part_rotation=tuple(float(c) for c in part_rotation), coefficients=coefficients,
                         wall_kind=wall_kind, wall_kind_version=wall_kind_version, spacing=float(spacing),
                         on_indefinite=on_indefinite, **{key: value for key, value in constants.items()})

    # -- the leaves ----------------------------------------------------------------------------------------------------

    @override
    def param_specs(self) -> dict[str, ParamSpec]:
        held = registered_kind(self.constants()["kind"], self.constants()["kind_version"])
        shapes = {f"{SHAPE}{name}": ParamSpec(units=setting_units(held, name)) if name in self._traced else
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
        spacing = ParamSpec(units="m", trainable=False, description="held as built: the points' spacing along the "
                                                                    "centreline the counts were made with, which the "
                                                                    "step reads to judge whether the body lies within "
                                                                    "the points its validity lines were measured at")
        return {**super().param_specs(), "viscosity": ParamSpec(units="Pa s"), "spacing": spacing,
                "part_centre": ParamSpec(units="m", trainable=False, description=placed),
                "part_rotation": ParamSpec(units="1", trainable=False, description=placed), **shapes, **walls, **data}

    def _wall(self) -> WallCorrection:
        return wall_correction_named(self.constants()["wall_kind"], self.constants()["wall_kind_version"])

    def _wall_trees(self) -> list[str]:
        constants = self.constants()
        return [f"{WALL}{name}" for name in sorted(self._wall().held) if array_tree(constants[f"{WALL}{name}"])]

    def _wall_arguments(self) -> dict[str, Any]:
        constants = self.constants()
        return {key: leaf for name in self._wall_trees() for key, leaf in floating_leaves(name, constants[name]).items()}

    @override
    def params_pytree(self) -> dict[str, Any]:
        inherited: Any = super()
        return {**cast(dict[str, Any], inherited.params_pytree()), **self._wall_arguments()}

    @override
    def _step_constants(self, params: Mapping[str, Any] | None) -> dict[str, Any]:
        """The step's constants, each held wall pytree rebuilt from its leaves in ``params`` where given."""
        held = super()._step_constants(params)
        if params is None:
            return held
        for name in self._wall_trees():
            paths, tree = _tree_util.tree_flatten_with_path(held[name])
            held[name] = _tree_util.tree_unflatten(tree, [params.get(f"{name}{path_name(path)}", leaf)
                                                          for path, leaf in paths])
        arguments = set(self._wall_arguments())
        return {key: value for key, value in held.items() if key not in arguments}

    @override
    def initial_state(self) -> NodeFields:
        zero = as_run_array(jnp.zeros(()))
        return node_fields(RFTDragState(resistance=as_run_array(jnp.zeros((6, 6))),
                                        background_load=as_run_array(jnp.zeros(6)), wall_accepted=jnp.asarray(True),
                                        least_eigenvalue=zero, indefinite=zero, unmeasured=zero, steps=zero,
                                        stopped_at=zero,
                                        stopped_least=zero, stopped_pose=as_run_array(jnp.zeros(7))))

    @override
    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        return {"pose": node_input("m; 1", (7,))}

    # -- the model -----------------------------------------------------------------------------------------------------

    @cached_property
    def point_count(self) -> int:
        return int(sum(self.constants()["counts"]))

    def parameters(self, p: Mapping[str, Any]) -> dict[str, Any]:
        """The kind's parameters at the step: its numbers from their leaves, its other settings as built."""
        constants = self.constants()
        return {name: p[f"{SHAPE}{name}"] if name in self._traced else constants[f"{SHAPE}{name}"]
                for name in (key.removeprefix(SHAPE) for key in constants if key.startswith(SHAPE))}

    def local_drag(self, p: Mapping[str, Any]) -> tuple[Array, Array, Array]:
        """The centreline's points and tangents in the body's frame ``(n, 3)`` each, and each point's local drag
        ``B_i`` ``(n, 3, 3)`` (N s/m: the coefficients times its arc), from the step's leaves."""
        dtype = jnp.asarray(p["viscosity"]).dtype
        parameters = self.parameters(p)
        counts = self.constants()["counts"]
        read = centreline_points(self._centreline, parameters, counts, dtype)
        lengths = [jnp.asarray(segment.length(parameters), dtype=dtype) for segment in self._centreline.segments]
        whole = sum(lengths[1:], lengths[0])
        pars: list[Any] = []
        perps: list[Any] = []
        for segment, count in zip(self._centreline.segments, counts, strict=True):
            filament = cast(Any, segment.filament_radius)(parameters)
            wavelength = None if segment.wavelength is None else segment.wavelength(parameters)
            par, perp = drag_coefficients(self.constants()["coefficients"], p["viscosity"], filament, wavelength,
                                          whole)
            pars.append(jnp.broadcast_to(jnp.asarray(par, dtype=dtype), (count,)))
            perps.append(jnp.broadcast_to(jnp.asarray(perp, dtype=dtype), (count,)))
        par, perp = jnp.concatenate(pars), jnp.concatenate(perps)
        turn = jnp.asarray(p["part_rotation"], dtype=dtype).reshape(3, 3)
        points = jnp.matmul(read.point, turn.T, precision=PRECISION) + jnp.asarray(p["part_centre"], dtype=dtype)
        tangents = jnp.matmul(read.tangent, turn.T, precision=PRECISION)
        along = tangents[:, :, None] * tangents[:, None, :]
        eye = jnp.eye(3, dtype=dtype)
        blocks = ((par[:, None, None] * along + perp[:, None, None] * (eye - along)) * read.arc[:, None, None])
        return points, tangents, blocks

    def resistance_at(self, p: Mapping[str, Any], pose: Array) -> tuple[Array, Array]:
        """The resistance at ``pose`` (its symmetric part) and whether every read of the wall's data was accepted. The
        local drag ``B`` is block diagonal, so ``B M`` and ``B G`` are taken point by point (three terms a sum), and
        ``M^T f`` over the points in pieces of at most ``DEPTH`` terms added in order (``mrs.force_sums``): a product
        over the 3n unknowns taken whole is blocked by the L1 data cache of the core XLA:CPU's Eigen first ran on, so its
        last bits would depend on where a process started (MIME-ANO-125)."""
        points, _, blocks = self.local_drag(p)
        turn = quaternion.rotation_matrix(pose[3:])
        world = pose[:3] + jnp.matmul(points, turn.T, precision=PRECISION)
        local = jnp.einsum("ab,nbc,dc->nad", turn, blocks, turn, precision=PRECISION)
        motions = rigid_motions(world, pose[:3])
        count = world.shape[0]
        pushed = jnp.einsum("nab,nbk->nak", local, motions.reshape(count, 3, 6), precision=PRECISION).reshape(
            3 * count, 6)
        wall = self._wall()
        if wall.adds_nothing:
            found, accepted = force_sums(motions, pushed), jnp.asarray(True)
        else:
            parameters = {name: p[f"{WALL}{name}"] for name in wall.parameters}
            image = wall.correction(world, parameters, p["viscosity"], jnp.zeros((), dtype=world.dtype))
            fed_back = jnp.einsum("nab,nbk->nak", local, image.matrix.reshape(count, 3, 3 * count),
                                  precision=PRECISION).reshape(3 * count, 3 * count)
            forces = jnp.linalg.solve(jnp.eye(3 * count, dtype=world.dtype) + fed_back, pushed)
            found, accepted = force_sums(motions, forces), image.accepted
        return (found + found.T) / 2.0, accepted

    def wall_gap(self, p: Mapping[str, Any], pose: Array) -> Array:
        """The least gap between the filament and the wall at ``pose``, in filament radii, from the wall kind's distance
        at the centreline's points (held out of every derivative: a report); a large number in free space."""
        wall = self._wall()
        if wall.adds_nothing or wall.distance is None:
            return jnp.asarray(jnp.inf, dtype=pose.dtype)
        points, _, _ = self.local_drag(p)
        world = pose[:3] + jnp.matmul(points, quaternion.rotation_matrix(pose[3:]).T, precision=PRECISION)
        parameters = {name: p[f"{WALL}{name}"] for name in wall.parameters}
        filament = jnp.min(jnp.stack([jnp.asarray(cast(Any, segment.filament_radius)(self.parameters(p)))
                                      for segment in self._centreline.segments]))
        return _lax.stop_gradient((jnp.min(wall.distance(world, parameters)) - filament) / filament).astype(pose.dtype)

    def judged(self, found: Array, dtype: Any) -> tuple[Array, float]:
        """The resistance's least eigenvalue as the kind's line model is judged, and the floor it is judged against
        (minus it): a straight body's, at the semidefinite floor; any other kind's, definite, against zero."""
        if self.constants()["kind"] in SEMIDEFINITE_KINDS:
            return floored_least_eigenvalue(found).astype(dtype), semidefinite_floor(dtype)
        return scaled_least_eigenvalue(found).astype(dtype), 0.0

    def measured_here(self, p: Mapping[str, Any]) -> tuple[Array, Array]:
        """Whether the body lies within the points the rung's validity lines were measured over, and with a wall the gap
        its family was measured at (filament radii; zero in free space or outside them), from the step's leaves: its
        family (:func:`measured_family`) by its shape and its slenderness (its centreline's length over its thinnest
        filament radius), at the default coefficients and a spacing no coarser than ``RFT_POINTS_PER_RADIUS`` a filament
        radius. A report that no derivative passes: comparisons, and the gap a selection between the families'
        measured numbers."""
        parameters = self.parameters(p)
        dtype = jnp.asarray(p["viscosity"]).dtype
        segments = self._centreline.segments
        lengths = [jnp.asarray(segment.length(parameters), dtype=dtype) for segment in segments]
        thinnest = jnp.min(jnp.stack([jnp.asarray(cast(Any, segment.filament_radius)(parameters), dtype=dtype)
                                      for segment in segments]))
        held = self.constants()
        family, gap = measured_family(held["kind"], parameters, sum(lengths[1:], lengths[0]) / thinnest,
                                      held["wall_kind"])
        fine = jnp.asarray(p["spacing"], dtype=dtype) <= thinnest / RFT_POINTS_PER_RADIUS * (1.0 + 1e-9)
        found = family & fine & (held["coefficients"] == default_coefficients(self._centreline))
        return found, gap.astype(dtype)

    def gap_unmeasured(self, p: Mapping[str, Any], pose: Array, measured: Array) -> Array:
        """Whether the wall's gap at ``pose`` is other than ``measured``, the one its family was measured at (none is,
        in free space): a narrower or a wider gap was not measured."""
        return (measured > 0.0) & (jnp.abs(self.wall_gap(p, pose) - measured) > GAP_TOLERANCE * measured)

    @override
    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        given = typed_view(RFTDragInputs, boundary_inputs)
        pose = given.get("pose", as_run_array(jnp.asarray((0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))))
        dtype = pose.dtype
        found, accepted = self.resistance_at(p, pose)
        found = found.astype(dtype)
        least, floor = self.judged(_lax.stop_gradient(found), dtype)
        indefinite = accepted & jnp.logical_not(least > -floor)
        stop = indefinite if self.constants()["on_indefinite"] == "stop" else jnp.asarray(False)
        measured, gap = self.measured_here(p)
        unmeasured = jnp.logical_not(measured) | self.gap_unmeasured(p, pose, gap)
        steps = state["steps"] + 1.0
        first = indefinite & (state["stopped_at"] == 0.0)
        record = jnp.where(first, jnp.concatenate([steps[None], least[None], pose]),
                           jnp.concatenate([state["stopped_at"][None], state["stopped_least"][None],
                                            state["stopped_pose"]]))
        return node_fields(RFTDragState(
            resistance=jnp.where(stop, jnp.nan, found), background_load=jnp.zeros(6, dtype=dtype),
            wall_accepted=accepted, least_eigenvalue=least, indefinite=indefinite.astype(dtype),
            unmeasured=unmeasured.astype(dtype), steps=steps,
            stopped_at=record[0], stopped_least=record[1], stopped_pose=record[2:]))

    # -- what the build and a run's end are told -------------------------------------------------------------------------

    def built_problems(self, start: Mapping[str, Any] | None = None) -> list[Problem]:
        """The resistance at the body's starting pose (``start``'s ``pose``), refused where the wall's reads there are
        refused or where it is indefinite past the floor: with a wall, a body whose local drag outweighs its true
        resistance (``drag.rft-indefinite``). A pose reached later is the step's to judge."""
        pose = None if start is None else start.get("pose")
        if pose is None:
            return []
        with cast(Any, jax).enable_x64(True), cast(Any, jax).ensure_compile_time_eval():
            found, accepted = self.resistance_at(self._built_leaves(), jnp.asarray(pose, dtype=jnp.float64))
            least_array, floor = self.judged(found, jnp.float64)
            least = float(least_array)
        if not bool(accepted):
            return [Problem("drag.rft-wall-range", "the wall's data was read outside its range at the body's starting "
                                                   "pose: a point of its centreline lies beyond what the wall kind's "
                                                   "table holds",
                            "start the body inside the wall's range, initial=State(position=...), or write a wall kind "
                            "that reaches it, Drag.RFT(wall_kind=...)")]
        if not least > -floor:
            return [Problem("drag.rft-indefinite", f"its resistance at the starting pose is indefinite: its least "
                                                   f"eigenvalue, each row and column scaled by its diagonal, is "
                                                   f"{least:.3g} ({self._settings_text()}); resistive force theory "
                                                   f"gives the body's segments no interaction through the fluid, so "
                                                   f"its local drag outweighs its true resistance, and the wall's "
                                                   f"image then feeds it back past the body's own drag",
                            "use the MRS rung, Drag.MRS(...), whose segments interact, for a body this large against "
                            "its wall; or a body small against the vessel")]
        return []

    def built_lines(self, start: Mapping[str, Any] | None, closest_approach: float | None) -> list[str]:
        """Whether the body lies within the points the rung's validity lines were measured over (never a refusal):
        where it does, the lines, estimates against the MRS at its resolution; where it does not, that its error
        against the MRS is not measured there, and that every step raises ``unmeasured``."""
        del closest_approach
        measured, measured_gap = self.measured_as_built()
        if not measured:
            return [f"the rft rung's validity lines were measured over other bodies than this one ({self._settings_text()}"
                    f"): its error against the MRS is not measured here, so each step raises unmeasured; the lines "
                    f"and the points they hold over are in docs/data/slender_validity"]
        line = (f"the rft rung's validity lines hold for this body's family (its points in "
                f"docs/data/slender_validity, estimates against the MRS at its resolution, never bounds)")
        if measured_gap > 0.0:
            gap = self._start_gap(start)
            if gap is not None and abs(gap - measured_gap) > GAP_TOLERANCE * measured_gap:
                return [f"the rft rung's validity lines were measured at a gap of {measured_gap:.3g} filament radii "
                        f"between the filament and the wall, and this body starts at {gap:.3g}: its error against the MRS "
                        f"is not measured there, so each step at another gap raises unmeasured; the lines and the "
                        f"points they hold over are in docs/data/slender_validity"]
            line += (f"; with its wall, at the gap measured, {measured_gap:.3g} filament radii between the "
                     f"filament and the wall (held to {GAP_TOLERANCE:.0%}), any other raising unmeasured")
        return [line]

    def measured_as_built(self) -> tuple[bool, float]:
        """:meth:`measured_here` at the leaves as built, in float64: what the build report says of the body."""
        with cast(Any, jax).enable_x64(True), cast(Any, jax).ensure_compile_time_eval():
            measured, gap = self.measured_here(self._built_leaves())
            return bool(measured), float(gap)

    def _built_leaves(self) -> dict[str, Any]:
        """The node's leaves as built, in float64 (called where float64 is enabled): what the build's checks evaluate
        the step's functions at."""
        leaves = {key: jnp.asarray(value, dtype=jnp.float64) if isinstance(value, (float, tuple, np.ndarray)) else value
                  for key, value in self.constants().items()}
        return {**leaves, **self._wall_arguments()}

    def _start_gap(self, start: Mapping[str, Any] | None) -> float | None:
        """The gap between the filament and the wall at the body's starting pose, in filament radii."""
        pose = None if start is None else start.get("pose")
        if pose is None:
            return None
        with cast(Any, jax).enable_x64(True), cast(Any, jax).ensure_compile_time_eval():
            return float(self.wall_gap(self._built_leaves(), jnp.asarray(pose, dtype=jnp.float64)))

    def _settings_text(self) -> str:
        held = self.constants()
        return (f"{held['kind']} {held['kind_version']}, coefficients {held['coefficients']!r}, "
                f"{self.point_count} points, wall {held['wall_kind']} {held['wall_kind_version']}")

    def stop_problem(self, state: Mapping[str, Any]) -> Problem | None:
        """Why the node stopped a run: the first step whose resistance was indefinite past the floor, which the node
        replaced by NaN before the body took it; ``None`` where it stopped none."""
        if "stopped_at" not in state or not float(np.asarray(state["stopped_at"])) > 0.0:
            return None
        if self.constants()["on_indefinite"] != "stop":
            return None
        step = int(round(float(np.asarray(state["stopped_at"]))))
        pose = np.asarray(state["stopped_pose"], dtype=np.float64)
        least = float(np.asarray(state["stopped_least"]))
        position = ", ".join(f"{c * 1e3:.5g}" for c in pose[:3])
        turn = ", ".join(f"{c:.5g}" for c in pose[3:])
        return Problem("drag.rft-indefinite",
                       f"its resistance turned indefinite at its step {step} (t = {step * self.delta_t:.6g} s), and "
                       f"the run stopped there, before the body took it: its least eigenvalue, each row and column "
                       f"scaled by its diagonal, was {least:.3g} ({self._settings_text()}), at the pose ({position}) "
                       f"mm, quaternion ({turn}): its local drag outweighs its true resistance against the wall",
                       "use the MRS rung, Drag.MRS(...), whose segments interact, for a body this large against its "
                       "wall")


__all__ = ["COEFFICIENT_SETS", "EQUATIONS", "GAP_TOLERANCE", "LIGHTHILL_Q", "MEASURED", "ON_INDEFINITE",
           "RFT_POINTS_PER_RADIUS", "SEMIDEFINITE_KINDS", "SHAPE_TOLERANCE",
           "Measured", "RFTDragInputs", "RFTDragNode", "RFTDragState", "coefficient_problem", "default_coefficients",
           "drag_coefficients", "measured_family", "shape_ratios"]
