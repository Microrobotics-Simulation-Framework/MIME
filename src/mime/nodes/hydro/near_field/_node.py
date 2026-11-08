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

from mime.core import MimeNode, NodeFields, Symbol, SymbolKind, Units, as_run_array, jit, jnp, node_input
from mime.nodes.hydro.mrs._node import (
    CONFINED_ASYMMETRY_LIMIT,
    HELD,
    NEAR_WALL,
    WALL,
    MRSDragNode,
    array_tree,
    floating_leaves,
    path_name,
    least_distance,
    rerun_leaves,
)
from mime.nodes.hydro.mrs._quadrature import asymmetry, held_or_made, scaled_least_eigenvalue, unchanged
from mime.nodes.hydro.mrs._walls import WallCorrection, wall_correction_named
from mime.nodes.hydro.near_field._contract import (
    blocks_of,
    cross_loads,
    grand_resistance,
    joint_matrix,
    lagged_gain,
    quadratic_form,
)
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
#: The most joint unknowns (three a force point, summed over the members) the dense route takes: its matrix, its copy
#: and its LU held in float64 are about three times ``8 (3 sum N)^2`` bytes. A labelled line from W3's first step
#: (``predictions_w3a.json``, its outcome in the pull request): on a 24-thread laptop CPU, eight spheres of 882 force
#: points (21,168 unknowns) peaked at 11 GiB on the dense LU, six (15,876) at 6.5 GiB, where GMRES took the same time
#: in about half the memory. Past it, a cluster needs W3b's GMRES route.
DENSE_UNKNOWNS = 16_000
#: The stop's causes, as the state's ``stopped_why`` holds them.
WHY_MEMBER, WHY_CLUSTER, WHY_GAIN = 1.0, 2.0, 3.0
#: The ledger's floor on a quadratic form's rounding, in units of eps on its scale (``mime.core.ledger``'s).
FORM_ULPS = 16.0


def member_prefix(index: int) -> str:
    return MEMBER.format(index)


class JointDragState(TypedDict, extra_items=ReadOnly[Float[Array, "..."]]):
    """``resistance_<i>`` and ``background_load_<i>`` for each member (its delivery); the cluster's symmetrised grand
    resistance; the step's judgement (each member's scaled least eigenvalue and the cluster's, the asymmetry, the gain
    and its relaxation in steps, the dissipation rate at the starting twists) with the flags it raises; with a wall,
    each body's least distance from it; and the stop's record (the step, why, which member, the value, the poses)."""

    resistance: ReadOnly[Float[Array, "k k"]]  # noqa: F722 - a jaxtyping shape
    least_eigenvalue: ReadOnly[Annotated[Float[Array, "n"], Units("1")]]  # noqa: F722 - a jaxtyping shape
    cluster_least: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    asymmetry: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    gain: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    relaxation_steps: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    dissipation: ReadOnly[Annotated[Float[Array, ""], Units("W")]]  # noqa: F722 - a jaxtyping shape
    wall_accepted: ReadOnly[Annotated[Bool[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    indefinite: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    gain_exceeded: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    near_wall: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    asymmetric: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    cluster_gift: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    cluster_gift_steps: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    cross_cells_short: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    wall_gap: NotRequired[ReadOnly[Annotated[Float[Array, "n"], Units("m")]]]  # noqa: F722 - a jaxtyping shape
    wall_gap_spacings: NotRequired[ReadOnly[Annotated[Float[Array, "n"], Units("1")]]]  # noqa: F722 - a jaxtyping shape
    wall_cells_short: NotRequired[ReadOnly[Annotated[Float[Array, ""], Units("1")]]]  # noqa: F722 - a jaxtyping shape
    steps: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    stopped_at: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    stopped_why: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    stopped_member: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    stopped_value: ReadOnly[Annotated[Float[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape
    stopped_poses: ReadOnly[Annotated[Float[Array, "n 7"], Units("m; 1")]]  # noqa: F722 - a jaxtyping shape


class JointDragInputs(TypedDict, extra_items=ReadOnly[Float[Array, "..."]]):
    """``pose_<i>``, ``twist_<i>`` and ``mass_matrix_<i>`` for each member, from its body's state."""


class JointDragNode(MimeNode):
    """The drag of a medium's interacting MRS bodies, solved as one joint system each step (C1: one joint node per
    medium; Bundle 2: the bodies' steps lagged, with a gain check).

    **Its members** are the bodies of ``members``, each with the constructor arguments its per-body
    ``MRSDragNode`` takes, written ``m<i>_<argument>``; the node builds each one's ``MRSDragNode`` (never added to the
    graph) and takes its discretisation from it (:class:`MRSOperator`). The wall's constants are the medium's, once.

    **Each step**: the joint matrix at the members' poses (each one's free part, made once per build in its frame and
    turned to its pose, with its wall's self-image as dev1's; each pair's cross-kernel within its reach and the wall's
    image between them), the grand resistance ``R`` by one dense LU with every body's six rigid motions, its symmetric
    part taken; then each body ``a`` is delivered its own ``R_aa`` into ``resistance`` and its lagged cross load
    ``-sum_b R_ab V_b`` at the step's starting twists into ``background_load``. The cluster's ``R`` stays in the state,
    so that each body's block and load are derived from it.

    **Judged before the bodies take it** (each held out of every derivative): each body's ``R_aa`` and the cluster's
    ``R`` by their least eigenvalue, each row and column scaled by its diagonal, and the lagged step's gain
    ``rho((M/dt + R_d)^-1 R_off)`` from each body's mass matrix as it publishes it. Where a body's or the cluster's
    resistance is not positive definite, or the gain is not below one (the lagged step would grow), every delivery is
    replaced by NaN and the run stops there (``stop_problem`` says which, when and why). Reported: the scaled least
    eigenvalues (the margin), the gain and the lagged coupling's relaxation in steps ``1/(1-g)``, the cluster's
    dissipation rate ``V^T R V`` at the starting twists (``cluster_gift`` where it is negative past its rounding,
    counted in ``cluster_gift_steps``: the ledger's judgement of the near field's cross loads, ``CHECK_CLUSTER``), the
    asymmetry, each body's distance from the wall, and whether a cross-kernel or a wall image ran short of its
    budget."""

    meta = NodeMeta(
        algorithm_id="MIME-NODE-122",
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
    symbols = (
        Symbol(r"\mathbf{R}", SymbolKind.STATE, "resistance", "N s/m; N s; N m s", _ADVANCE),
        Symbol(r"\mu", SymbolKind.PARAMETER, "viscosity", "Pa s", _ADVANCE),
        Symbol(r"g", SymbolKind.STATE, "gain", "1", _ADVANCE),
    )

    def __init__(self, name: str, timestep: float, *, members: Sequence[str], viscosity: float, wall_kind: str,
                 wall_kind_version: str, reach: float = REACH, cross_share: float = CROSS_SHARE,
                 **constants: object) -> None:
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
        super().__init__(name, timestep, members=held, viscosity=viscosity, wall_kind=wall_kind,
                         wall_kind_version=wall_kind_version, reach=reach, cross_share=cross_share, **constants)

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
            for key, spec in member.param_specs().items():
                if key in ("viscosity", HELD) or key in self._wall_keys():
                    continue
                found[f"{prefix}{key}"] = spec
            found[f"{prefix}{HELD_MATRIX}"] = ParamSpec(units="1", trainable=False, description=made)
        first = self._members[0].param_specs()
        found.update({key: spec for key, spec in first.items() if key in self._wall_keys()})
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
        return {**own, **self._wall_arguments(), **held}

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
    def _made_compiled(self) -> tuple[Any, ...]:
        return tuple(jit(member.self_interaction) for member in self._members)

    def _held_value(self, index: int, p: Mapping[str, Any]) -> Array:
        """A member's free part at the step's leaves: the held one where they are as built (or as the save's
        resolution rerun moves them), made afresh where one has moved (``held_or_made``)."""
        member = self._members[index]
        theta = {key: p[f"{member_prefix(index)}{key}"] for key in member.body_leaf_names()}
        key = f"{member_prefix(index)}{HELD_MATRIX}"
        if key not in p:
            return member.self_interaction(theta)
        built = member.built_theta()
        return held_or_made(member.self_interaction, built, (rerun_leaves(theta, built),))(theta, p[key])

    @override
    def made_again(self, params: Mapping[str, Any]) -> dict[str, Any]:
        found: dict[str, Any] = {}
        for i, member in enumerate(self._members):
            key = f"{member_prefix(i)}{HELD_MATRIX}"
            if key not in params:
                continue
            theta = {name: params[f"{member_prefix(i)}{name}"] for name in member.body_leaf_names()}
            if bool(unchanged(theta, member.built_theta())):
                continue
            found[key] = self._made_compiled[i](theta).astype(jnp.asarray(params[key]).dtype)
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
                     relaxation_steps=zero, dissipation=zero, wall_accepted=jnp.asarray(True), indefinite=zero,
                     gain_exceeded=zero, near_wall=zero, asymmetric=zero, cluster_gift=zero, cluster_gift_steps=zero,
                     cross_cells_short=zero,
                     steps=zero, stopped_at=zero, stopped_why=zero, stopped_member=zero, stopped_value=zero,
                     stopped_poses=as_run_array(jnp.zeros((count, 7))))
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

    def solved(self, p: Mapping[str, Any], poses: Array) -> tuple[Array, Array, Array, Array, Array]:
        """The cluster's grand resistance at ``poses`` (``(K, 7)``) from the constants ``p``, not symmetrised; whether
        every wall read was accepted; whether any wall image or cross-kernel ran short; each body's least distance from
        the wall (zero without a wall's distance). Checkpointed: reverse mode keeps its inputs and solves again."""
        wall = self._wall()
        parameters = {name: p[f"{WALL}{name}"] for name in wall.parameters}
        leaves = [self._leaves(i, p) for i in range(len(self._members))]
        held = [self._held_value(i, p) for i in range(len(self._members))]
        operators, distance = self._operators, wall.distance

        def solve(held: Sequence[Array], leaves: Sequence[Mapping[str, Any]], poses: Array, viscosity: Array,
                  parameters: Mapping[str, Any], reach: Array) -> tuple[Array, Array, Array, Array, Array]:
            joint = joint_matrix(operators, leaves, held, poses, viscosity, wall, parameters, reach)
            gaps = jnp.zeros(len(operators), dtype=poses.dtype) if distance is None else jnp.stack(
                [least_distance(distance(cast(Quadrature, placed.own).points, parameters)) for placed in joint.placed])
            return grand_resistance(joint), joint.accepted, jnp.any(joint.wall_short), jnp.any(joint.cross_short), gaps

        return checkpoint(solve)(held, leaves, poses, p["viscosity"], parameters, p["reach"])

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
        found, accepted, wall_short, cross_short, gaps = self.solved(p, poses)
        found = found.astype(dtype)
        symmetric = 0.5 * (found + found.T)
        own, off = blocks_of(symmetric, count)
        loads = cross_loads(off, twists)
        judged = _lax.stop_gradient(found)
        held_own, held_off = blocks_of(0.5 * (judged + judged.T), count)
        least = jnp.stack([scaled_least_eigenvalue(judged[6 * i:6 * i + 6, 6 * i:6 * i + 6])
                           for i in range(count)]).astype(dtype)
        cluster = scaled_least_eigenvalue(judged).astype(dtype)
        dropped = asymmetry(judged).astype(dtype)
        gain = lagged_gain(held_own, held_off, _lax.stop_gradient(masses), dt).astype(dtype)
        form, scale = quadratic_form(0.5 * (judged + judged.T), _lax.stop_gradient(twists))
        gift = form < -FORM_ULPS * jnp.finfo(dtype).eps * scale
        member_stop, cluster_stop, gain_stop = ~(jnp.min(least) > 0.0), ~(cluster > 0.0), ~(gain < 1.0)
        stop = accepted & (member_stop | cluster_stop | gain_stop)
        why = jnp.where(member_stop, WHY_MEMBER, jnp.where(cluster_stop, WHY_CLUSTER, WHY_GAIN)).astype(dtype)
        value = jnp.where(member_stop, jnp.min(least), jnp.where(cluster_stop, cluster, gain)).astype(dtype)
        steps = state["steps"] + 1.0
        first = stop & (state["stopped_at"] == 0.0)
        record = jnp.where(first, jnp.concatenate([steps[None], why[None], jnp.argmin(least).astype(dtype)[None],
                                                   value[None], poses.reshape(-1)]),
                           jnp.concatenate([state["stopped_at"][None], state["stopped_why"][None],
                                            state["stopped_member"][None], state["stopped_value"][None],
                                            state["stopped_poses"].reshape(-1)]))
        out: dict[str, Array] = {}
        for i in range(count):
            out[f"resistance_{i}"] = jnp.where(stop, jnp.nan, own[i]).astype(dtype)
            out[f"background_load_{i}"] = jnp.where(stop, jnp.nan, loads[i]).astype(dtype)
        flags = jnp.stack([stop & (member_stop | cluster_stop), stop & gain_stop, dropped > CONFINED_ASYMMETRY_LIMIT,
                           gift, cross_short]).astype(dtype)
        relaxation = jnp.where(gain < 1.0, 1.0 / (1.0 - jnp.minimum(gain, 1.0 - jnp.finfo(dtype).eps)), jnp.inf)
        out.update(resistance=symmetric, least_eigenvalue=least, cluster_least=cluster, asymmetry=dropped, gain=gain,
                   relaxation_steps=relaxation.astype(dtype), dissipation=form.astype(dtype), wall_accepted=accepted,
                   indefinite=flags[0], gain_exceeded=flags[1], asymmetric=flags[2], cluster_gift=flags[3],
                   cluster_gift_steps=state["cluster_gift_steps"] + flags[3],
                   cross_cells_short=flags[4], steps=steps, stopped_at=record[0], stopped_why=record[1],
                   stopped_member=record[2], stopped_value=record[3], stopped_poses=record[4:].reshape(count, 7))
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
        """Each member's own refusals of its free part (``MRSDragNode.built_problems``); a joint system past the dense
        route's size (:data:`DENSE_UNKNOWNS`); and at the bodies' starting poses and twists, a body's or the cluster's
        resistance that is not positive definite, or a lagged gain not below one."""
        found: list[Problem] = []
        for body, member in zip(self.members, self._members, strict=True):
            found += [Problem(problem.code, f"{body}: {problem.message}", problem.action)
                      for problem in member.built_problems(None)]
        if found:
            return found
        if self.unknowns > DENSE_UNKNOWNS:
            return [Problem("drag.near-field-size", f"the medium's near field couples {len(self.members)} bodies "
                                                    f"({', '.join(self.members)}) into a joint system of "
                                                    f"{self.unknowns} unknowns, past the dense route's "
                                                    f"{DENSE_UNKNOWNS}: a memory line, where the dense LU's peak "
                                                    f"outgrows an iterative solve's (measured at 21,168 unknowns: 11 "
                                                    f"against 5.5 GiB in float64, in about the same time); this one's "
                                                    f"LU would hold about "
                                                    f"{3 * 8 * self.unknowns ** 2 / 2 ** 30:.3g} GiB",
                            "write a coarser spacing on the bodies' drag, Drag.MRS(spacing=...), or fewer bodies in "
                            "the medium; the near field's iterative route (GMRES, each body's own block its "
                            "preconditioner), planned and not in this release, lifts this line")]
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
        lines = [f"the medium's near field couples {', '.join(self.members)} in one joint system (force points: "
                 f"{counts}; {self.unknowns} unknowns), solved each step by one dense LU for the cluster's grand "
                 f"resistance, each body's step lagged: its own coupled 6 x 6 implicit, the others' loads from the "
                 f"step before",
                 f"the cross-kernel between bodies: the source's regularised Stokeslet at its force points where the "
                 f"pair is farther apart than {self.constants()['reach']:g} of the source's cells' own size, "
                 f"integrated over every cell within half that, blended between; integrated over at most {budgets} "
                 f"quadrature points a source, the nearest cells first (cross_cells_short where a pair within the "
                 f"reach left a cell out)",
                 f"its matrix in the run's dtype {8 * self.unknowns ** 2 / 2 ** 20:.4g} MiB in float64, the LU's "
                 f"peak about three times that; past {DENSE_UNKNOWNS} unknowns the build refuses the dense route"]
        if start is not None and all(f"pose_{i}" in start for i in range(len(self.members))):
            judged = self.judged_at(start)
            margins = ", ".join(f"{body} {float(v):.3g}" for body, v in zip(self.members, judged["least_eigenvalue"],
                                                                             strict=True))
            gain = float(judged["gain"])
            lines.append(f"at the start: the lagged gain {gain:.3g} (a step's error relaxes over about "
                         f"{float(judged['relaxation_steps']):.3g} steps; at one or above the run stops), the scaled "
                         f"least eigenvalue of each body's coupled resistance {margins} and of the cluster's "
                         f"{float(judged['cluster_least']):.3g} (its margin; not above zero, the run stops)")
        return lines

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
        if why == WHY_GAIN:
            return Problem(
                "drag.near-field-gain",
                f"the lagged coupling of {', '.join(self.members)} would grow: its gain, the spectral radius of "
                f"(M/dt + R_d)^-1 R_off, was {value:.4g} {when} ({where}); a lagged step above one amplifies its "
                f"error at every step (three overdamped bodies in a triangle near contact do)",
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


__all__ = ["DENSE_UNKNOWNS", "EQUATIONS", "HELD_MATRIX", "JointDragInputs", "JointDragNode", "JointDragState",
           "member_prefix"]
