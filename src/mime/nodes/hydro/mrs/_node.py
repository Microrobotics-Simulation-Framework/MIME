"""The MRS resistance of a rigid body: its hull kind's samples, placed by the body's pose, solved for the
resistance about its centre of mass at every step."""

from __future__ import annotations

import dataclasses
import math
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
from typing_extensions import NotRequired, ReadOnly, TypedDict

from mime.core import MimeNode, NodeFields, Symbol, SymbolKind, as_run_array, jnp, node_fields, node_input, \
    quaternion, typed_view
from mime.nodes.hydro.mrs._quadrature import (
    Chunks,
    asymmetry,
    cell_moments,
    empty_cells,
    held_or_made,
    separated_matrix,
    separated_resistance,
    turned,
    turned_resistance,
)
from mime.nodes.hydro.mrs._solve import PRECISION, blob_squared, resistance
from mime.nodes.hydro.mrs._walls import WallCorrection, wall_correction_named
from mime.registry import Direction, EntryKind, Geometry, Problem, Registered, Samples, Sampling, lookup
from mime.units import Quantity, unit_symbol

EQUATIONS = (r"\mathbf{R} = \mathbf{M}^{\top} (\mathbf{A} + \mathbf{G})^{-1} \mathbf{M}, \quad "
             r"\mathbf{A}_{mn} = \frac{(r^{2} + 2 \epsilon^{2}) \mathbf{I} + \mathbf{r} \mathbf{r}^{\top}}"
             r"{8 \pi \mu (r^{2} + \epsilon^{2})^{3/2}}, \quad \mathbf{r} = \mathbf{x}_{m} - \mathbf{x}_{n}, \quad "
             r"\epsilon^{2} = b^{2} \frac{1}{N} \sum_{n} w_{n}, \quad "
             r"\mathbf{x}_{n} = \mathbf{p} + R(\mathbf{q}) (\mathbf{P} \mathbf{y}_{n} + \mathbf{c})")

#: ``jax.checkpoint``, whose hints leave its result partly unknown to a strict checker, and the pytree functions.
checkpoint: Any = cast(Any, jax).checkpoint
_tree_util: Any = cast(Any, jax).tree_util
#: Where the symbol rows' functions are.
_ADVANCE = "mime.nodes.hydro.mrs._node.MRSDragNode.advance"
_SOLVE = "mime.nodes.hydro.mrs._solve.resistance"
#: The prefix of a node constant holding a setting of the body's hull kind, and of one holding a wall's parameter.
SHAPE, WALL = "shape_", "wall_"
#: The share of a separated quadrature's free resistance its symmetric part may drop before the build refuses it
#: (``asymmetry``). A labelled decision of phase 7m-b: at every setting measured there it was below 1e-3 (2e-6 on the
#: sphere at its default, 8e-4 on FL-9 at 206 force points and blob 1.0), and the rounding-dominated solves of the
#: helix reference reached 2e-2; the limit catches the gross failures, not every solve that rounding has reached.
ASYMMETRY_LIMIT = 1e-2
#: The ``params`` leaf holding, on a separated quadrature, the body-frame quantity made once per build: in free space
#: the resistance about the body's origin (6 x 6), with a wall the free part's matrix (3N x 3N), at unit viscosity.
HELD = "held_in_body_frame"


class MRSDragState(TypedDict, closed=True):
    resistance: ReadOnly[Float[Array, "6 6"]]  # noqa: F722 - a jaxtyping shape
    background_load: ReadOnly[Float[Array, "6"]]
    wall_accepted: ReadOnly[Bool[Array, ""]]  # noqa: F722 - a jaxtyping shape


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
    discretisation's error; the build reports what it drops (``asymmetry``)."""

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
                 quadrature_owners: np.ndarray | None = None, **constants: object) -> None:
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
        return node_fields(MRSDragState(resistance=as_run_array(jnp.zeros((6, 6))),
                                        background_load=as_run_array(jnp.zeros(6)), wall_accepted=jnp.asarray(True)))

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

    def body_leaf_names(self) -> tuple[str, ...]:
        """The leaves the body-frame quantity is made from: the kind's numbers, the part's placement and the blob."""
        return (*(f"{SHAPE}{name}" for name in self._traced), "part_rotation", "part_centre", "blob")

    def _quadrature_samples(self, p: Mapping[str, Any]) -> Samples:
        """The separated quadrature's points and weights in the body's frame: the kind's quadrature at
        ``quadrature_counts``, placed as the samples are."""
        if self._quadrature is None:
            return self.samples(p)
        constants = self.constants()
        parameters = {name: p[f"{SHAPE}{name}"] if name in self._traced else constants[f"{SHAPE}{name}"]
                      for name in (key.removeprefix(SHAPE) for key in constants if key.startswith(SHAPE))}
        found = Sampling(self._sampling.patches, self._quadrature, "quadrature").samples(self._kind, parameters)
        dtype = jnp.asarray(p["blob"]).dtype
        turn = jnp.asarray(p["part_rotation"], dtype=dtype).reshape(3, 3)
        centre = jnp.asarray(p["part_centre"], dtype=dtype)
        return Samples(jnp.matmul(jnp.asarray(found.points, dtype=dtype), turn.T, precision=PRECISION) + centre, None,
                       jnp.asarray(found.weights, dtype=dtype))

    def body(self, theta: Mapping[str, Any]) -> Body:
        """The body's discretisation in its own frame from the leaves ``theta`` (:meth:`_theta_keys`)."""
        p = {**theta, "viscosity": jnp.asarray(theta["blob"])}
        force = self.samples(p).points
        quadrature = self._quadrature_samples(p)
        owners = cast(np.ndarray, self._owners)
        area, centroid = cell_moments(quadrature.points, quadrature.weights, owners, self.sample_count)
        eps_squared = blob_squared(area, theta["blob"])
        return Body(force, quadrature.points, quadrature.weights, area, centroid, eps_squared)

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
    def _built(self) -> tuple[np.ndarray, float, float]:
        """The body-frame quantity made once, from the node's constants in float64 whatever the run's precision (the
        resistance in free space, the free part's matrix with a wall), and what the build judges of the free part's
        resistance: the share its symmetric part drops and its least eigenvalue, each row scaled by its diagonal."""
        with cast(Any, jax).enable_x64(True):
            theta = {key: jnp.asarray(value, dtype=jnp.float64) for key, value in self._built_theta().items()}
            body = self.body(theta)
            matrix = separated_matrix(body.force, body.points, body.weights, self.chunks, body.eps_squared)
            found = separated_resistance(matrix, body.force, body.area, body.centroid, jnp.zeros(3, dtype=jnp.float64))
            symmetric = (found + found.T) / 2.0
            scale = jnp.sqrt(jnp.abs(jnp.diag(symmetric)))
            least = float(jnp.min(jnp.linalg.eigvalsh(symmetric / (scale[:, None] * scale[None, :]))))
            held = symmetric if self._free() else matrix
            return np.asarray(held, dtype=np.float64), float(asymmetry(found)), least

    @property
    def held_in_body_frame(self) -> np.ndarray:
        """The body-frame quantity made once (``HELD``): the resistance in free space, the free part's matrix with a
        wall, at unit viscosity, in float64."""
        return self._built[0]

    @property
    def free_asymmetry(self) -> float:
        """The share of the free part's resistance its symmetric part drops (``asymmetry``), as built."""
        return self._built[1]

    def built_problems(self, start: Mapping[str, Any] | None = None) -> list[Problem]:
        """What the build refuses of the resistance on a separated quadrature: a free part whose symmetric part is not
        positive definite, or whose asymmetry is past :data:`ASYMMETRY_LIMIT`, each the sign of a discretisation that
        no longer resolves its blob (a blob too large for its force spacing makes the matrix rounding-dominated) or a
        quadrature too coarse for it; and, with a wall, the resistance at the body's starting pose (``start``'s
        ``pose``) that is not positive definite, the wall's image taken at force points that do not resolve it (a
        body within about a force spacing of the wall, MIME-ANO-129). A pose reached later is not checked."""
        if self._owners is None:
            return []
        empty = empty_cells(self._owners, self.sample_count)
        if empty:
            return [Problem("drag.empty-cell", f"{len(empty)} of the {self.sample_count} force points have no point of "
                                               f"the quadrature nearest them (the first {empty[:5]}), so their cells "
                                               f"are empty and the matrix singular",
                            "write a larger quadrature_refinement: Drag.MRS(quadrature_refinement=...)")]
        _, dropped, least = self._built
        found: list[Problem] = []
        settings = (f"{self.sample_count} force points, a quadrature of {self.quadrature_count} points, blob "
                    f"{self.constants()['blob']!r}")
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
        if not found and not self._free() and start is not None and "pose" in start:
            pose = as_run_array(start["pose"])
            resistance = np.asarray(self.update(self.initial_state(), {"pose": pose}, self.delta_t,
                                                params=self.params_pytree())["resistance"], dtype=np.float64)
            scale = np.sqrt(np.abs(np.diag(resistance)))
            confined = float(np.linalg.eigvalsh(resistance / np.outer(scale, scale))[0]) if np.all(scale > 0.0) \
                else float("nan")
            if not (math.isfinite(confined) and confined > 0.0):
                found.append(Problem(
                    "drag.mrs-resistance", f"the resistance at the body's starting pose, the wall's correction in it "
                                           f"({settings}), is not positive definite: its least eigenvalue, scaled, is "
                                           f"{confined!r}; the wall's image is taken at the force points, which do not "
                                           f"resolve it within about a force spacing of the wall (MIME-ANO-129)",
                    "write a smaller spacing, Drag.MRS(spacing=...), or start the body further from the wall"))
        return found

    def _held_value(self, p: Mapping[str, Any]) -> Array:
        """The body-frame quantity at the step's leaves: the held one where they are as built (``held_or_made``),
        made from them where one has moved or where the step was given no held value (a call with no ``params``)."""
        theta = {key: p[key] for key in self.body_leaf_names()}
        if HELD not in p:
            return self._make()(theta)
        return held_or_made(self._make(), self._built_theta())(theta, p[HELD])

    def _separated_step(self, p: Mapping[str, Any], pose: Array) -> NodeFields:
        viscosity = p["viscosity"]
        held = self._held_value(p)
        if self._free():
            found = viscosity * turned_resistance(held, quaternion.rotation_matrix(pose[3:]))
            return node_fields(MRSDragState(resistance=found.astype(pose.dtype),
                                            background_load=jnp.zeros(6, dtype=pose.dtype),
                                            wall_accepted=jnp.asarray(True)))
        wall = self._wall()
        theta = {key: p[key] for key in self.body_leaf_names()}
        parameters = {name: p[f"{WALL}{name}"] for name in wall.parameters}

        def solved(held: Array, theta: Mapping[str, Any], viscosity: Array, pose: Array,
                   parameters: Mapping[str, Any]) -> tuple[Array, Array]:
            turn = quaternion.rotation_matrix(pose[3:])
            body = self.body(theta)
            force = pose[:3] + jnp.matmul(body.force, turn.T, precision=PRECISION)
            centroid = pose[:3] + jnp.matmul(body.centroid, turn.T, precision=PRECISION)
            corrected = wall.correction(force, parameters, viscosity)
            matrix = turned(held, turn) / viscosity + corrected.matrix * jnp.repeat(body.area, 3)[None, :]
            found = separated_resistance(matrix, force, body.area, centroid, pose[:3])
            return (found + found.T) / 2.0, corrected.accepted

        found, accepted = checkpoint(solved)(held, theta, viscosity, pose, parameters)
        return node_fields(MRSDragState(resistance=found.astype(pose.dtype),
                                        background_load=jnp.zeros(6, dtype=pose.dtype), wall_accepted=accepted))

    @override
    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        given = typed_view(MRSDragInputs, boundary_inputs)
        pose = given.get("pose", as_run_array(jnp.asarray((0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))))
        if self._owners is not None:
            return self._separated_step(p, pose)
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


__all__ = ["ASYMMETRY_LIMIT", "EQUATIONS", "HELD", "SHAPE", "WALL", "Body", "MRSDragInputs", "MRSDragNode",
           "MRSDragState", "kind_from_settings"]
