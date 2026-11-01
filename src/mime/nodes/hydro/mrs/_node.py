"""The MRS resistance of a rigid body: its hull kind's samples, placed by the body's pose, solved for the
resistance about its centre of mass at every step."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any, cast, override

import jax
from jax import Array
from jaxtyping import Bool, Float
from maddening.core.compliance.metadata import NodeMeta, Reference
from maddening.core.node import BoundaryInputSpec
from maddening.core.params import ParamSpec
from typing_extensions import NotRequired, ReadOnly, TypedDict

from mime.core import MimeNode, NodeFields, Symbol, SymbolKind, as_run_array, jnp, node_fields, node_input, \
    quaternion, typed_view
from mime.nodes.hydro.mrs._solve import PRECISION, blob_squared, resistance
from mime.nodes.hydro.mrs._walls import WallCorrection, wall_correction_named
from mime.registry import Direction, EntryKind, Geometry, Registered, Samples, lookup
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


class MRSDragNode(MimeNode):
    """The regularised-Stokeslet resistance (6 x 6, world frame, SI) of a rigid body about its centre of mass at its
    pose, from its hull kind's samples on the named ``patches`` at ``counts`` (``mime.nodes.hydro.mrs``'s
    ``resistance``), with the wall correction ``wall_kind`` added; and the background load, zero (the medium is at
    rest).

    The samples are the kind's, in its own frame, from its settings (each a node constant ``shape_<setting>``): a
    number is a ``params`` leaf, so that a study's derivative in it flows through the samples; a direction and an
    integer are held as built (a part's direction is its frame, and turning the body is its pose). They are placed in
    the body's frame by the part's ``part_rotation`` (rows) and ``part_centre``, and in the world by the pose. The blob
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
    constant itself."""

    meta = NodeMeta(
        algorithm_id="MIME-NODE-108",
        description="The resistance of a rigid body in Stokes flow by the method of regularised Stokeslets "
                    "on its hull kind's samples, with a wall's correction added.",
        governing_equations=EQUATIONS,
        assumptions=("Stokes flow: quasi-steady, the limit of vanishing Reynolds number",
                     "The surface sampled by the kind's quadrature, a point force at each sample, regularised by a blob "
                     "of the samples' mean spacing times b",
                     "A medium at rest, so the background load is zero"),
        references=(Reference("Cortez2005", "the regularised Stokeslet and its method of regularised Stokeslets"),),
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
                 part_rotation: Sequence[float], wall_kind: str, wall_kind_version: str, **constants: object) -> None:
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
        super().__init__(name, timestep, viscosity=viscosity, blob=blob, kind=kind, kind_version=kind_version,
                         patches=tuple(patches), counts=tuple(tuple(int(c) for c in count) for count in counts),
                         part_centre=tuple(float(c) for c in part_centre),
                         part_rotation=tuple(float(c) for c in part_rotation), wall_kind=wall_kind,
                         wall_kind_version=wall_kind_version, **{key: value for key, value in constants.items()})

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
        return {**super().param_specs(), "viscosity": ParamSpec(units="Pa s"), "blob": ParamSpec(units="1"),
                "part_centre": ParamSpec(units="m", trainable=False, description=placed),
                "part_rotation": ParamSpec(units="1", trainable=False, description=placed), **shapes, **walls, **data}

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
        return {**cast(dict[str, Any], inherited.params_pytree()), **self._wall_arguments()}

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

    def samples(self, p: Mapping[str, Any]) -> Samples:
        """The kind's samples on every patch, in the body's frame: points, normals and weights."""
        constants = self.constants()
        parameters = {name: p[f"{SHAPE}{name}"] if name in self._traced else constants[f"{SHAPE}{name}"]
                      for name in (key.removeprefix(SHAPE) for key in constants if key.startswith(SHAPE))}
        found = [self._kind.samples(parameters, patch, tuple(count))
                 for patch, count in zip(constants["patches"], constants["counts"], strict=True)]
        dtype = jnp.asarray(p["viscosity"]).dtype
        turn = jnp.asarray(p["part_rotation"], dtype=dtype).reshape(3, 3)
        centre = jnp.asarray(p["part_centre"], dtype=dtype)
        points = jnp.concatenate([jnp.asarray(s.points, dtype=dtype) for s in found])
        normals = jnp.concatenate([jnp.asarray(s.normals, dtype=dtype) for s in found])
        weights = jnp.concatenate([jnp.asarray(s.weights, dtype=dtype) for s in found])
        return Samples(jnp.matmul(points, turn.T, precision=PRECISION) + centre,
                       jnp.matmul(normals, turn.T, precision=PRECISION), weights)

    @override
    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        given = typed_view(MRSDragInputs, boundary_inputs)
        pose = given.get("pose", as_run_array(jnp.asarray((0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))))
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


__all__ = ["EQUATIONS", "SHAPE", "WALL", "MRSDragInputs", "MRSDragNode", "MRSDragState", "kind_from_settings"]
