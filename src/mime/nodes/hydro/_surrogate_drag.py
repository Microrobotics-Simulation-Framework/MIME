"""The resistance of a de Jongh screw body in a straight tube, from the Cholesky MLP surrogate, with the near-wall
lubrication correction blended in: ported from the 0.2 line's ``MLPResistanceNode`` and ``LubricationCorrectionNode``
as one law, with three fixes.

- **The tube's axis is a parameter**, with a point on it and the tube's radius, from the geometry of the region the
  medium fills. The 0.2 node took the tube to run along +z through the origin, so in a tube along x it read the
  body's position along the tube as a sideways offset and clamped it on every step.
- **The offset's direction is built from vectors**, not from ``arctan2``: on the axis, where it has none, the value
  and its derivative are finite (the 0.2 node's ``sqrt(0)`` and ``arctan2(0, 0)`` differentiated to NaN). An offset
  below ``AXIS_ULPS`` units in the last place of the tube's radius counts as on the axis, and there the canonical
  frame takes the body's own x axis across the tube (its y axis where x lies within 60 degrees of the tube's axis), so
  that turning the whole setup turns the law. The 0.2 node took the world's x axis there (``arctan2(0, 0) = 0``), which nothing turns; for a
  body on the axis with its x axis along the world's, the two agree. The surrogate's resistance is not symmetric about
  the axis at zero offset (by about 4% of its largest entry in each block), so it is discontinuous across the axis
  either way: a property of the trained model, which a point on the axis cannot remove.
- **The tube's radius is one number**: the 0.2 node's callable ``R_ves(z)`` was a closure, which cannot be saved.

The surrogate's resistance is non-dimensional, in the canonical frame (the tube's axis as z, the offset along +x), for
features ``(nu, L / R, kappa, offset / R, 0, log(gap / R), nu^2, kappa^2)`` with ``R`` the body's reference radius,
``kappa = R / R_tube`` and ``gap`` the distance from the body's outer radius ``(1 + modulation) R`` to the wall. It is
turned into the world frame block by block and scaled to SI by ``mu R``, ``mu R^2`` (both coupling blocks) and
``mu R^3``. The offset is clamped as in the 0.2 line to 0.95 of the room between the body's outer radius and the wall,
and ``clamp_fired`` reports when it is.

**Reference point** (MIME-ANO-112). The training's BEM took its torques about the origin of the body's mesh, the end
``zeta = 0`` of de Jongh's Eq. 1, with the body spanning ``zeta`` from 0 to ``L`` along the tube's axis: the surrogate's
resistance ``R_P`` is about that end. The node moves it to the body's centre of mass, where the body's dynamics applies
it. With ``d`` from the end ``P`` to the centre ``C``, a point of the body at ``P`` moves at ``U_P = U_C + d x Omega``
and the torque about ``C`` is ``T_C = T_P - d x F``, so ``[U_P; Omega] = T [U_C; Omega]`` and ``[F; T_C] = T^T [F;
T_P]`` with ``T = [[I, [d]x], [0, I]]``: ``R_C = T^T R_P T``. A congruence, so ``R_C`` stays symmetric positive
definite; a shift along the tube's axis leaves the axial block (the force along the axis and the torque about it,
against the velocity along it and the turning about it) unchanged.

- **Which end.** The helix's axis is the body's x axis, and ``zeta`` runs along +x: the body spans ``-L/2`` to ``+L/2``
  along its x axis, its end ``zeta = 0`` at ``-L/2``. The surrogate takes that axis to lie along the tube's, so the
  canonical frame's axis is the tube's axis on the side the body's x axis points to, ``s a`` with ``s = +1`` where
  ``x . a >= 0`` and ``-1`` otherwise, the frame ``[n, s a x n, s a]``, and ``d = (L/2) s a``. A body turned end for
  end (``s`` from +1 to -1) gets the half turn about ``n`` of the resistance it had: the mirror-image shift. ``s``
  jumps where the body's axis lies across the tube, outside what the surrogate was trained on; the derivative is
  that of the side taken.
- ``reference_point="end"`` keeps the resistance about the end, with ``s = +1`` whatever the orientation, as the 0.2
  line and drag 0.1.0 delivered it: a test-only switch for comparing with them; the entry builds ``"centre"``.

**Lubrication** (Cox & Brenner 1967; Goldman, Cox & Brenner 1967a, b; added to a far-field resistance as Stokesian
dynamics adds its lubrication terms, Brady & Bossis 1988), with the body's nearest point taken as a sphere of the
body's outer radius ``a``, the gap ``delta`` from it to the wall along the radial normal ``n``, and the axis ``t1``
and ``t2 = n x t1`` along the wall: ``R += w (R_nn n n^T + R_tt (t1 t1^T + t2 t2^T))`` in translation and
``w R_rr (t1 t1^T + t2 t2^T)`` in rotation, about the sphere's centre, which sits at the body's centre of mass across
the tube: so about the reference point the surrogate's resistance is now moved to, and consistent with it for a body
whose axis lies along the tube's (a tilted body's nearest point lies off its centre, where the terms would need the
same shift from that point; the model omits it, as it omits the sphere's own translation-rotation coupling), ``w = exp(-delta / epsilon)`` (the 0.2 line's blend, a modelling choice
no reference fixes), with ``R_nn = 6 pi mu a^2 / delta``, ``R_tt = 6 pi mu a (8/15) ln(a / delta)`` and ``R_rr = 8 pi
mu a^3 (2/5) ln(a / delta)``, each logarithm floored at zero; no translation-rotation coupling (as the 0.2 line). As
the gap closes, ``delta`` is floored at ``GAP_FLOOR`` (1 nm, the 0.2 line's): there ``R_nn`` is ``6 pi mu a^2 / 1 nm``
and the logarithms are finite. A contact's margin keeps a confined body's gap at the margin or above, far above the
floor.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, override

from jax import Array
from jaxtyping import Float
from maddening.core.node import BoundaryInputSpec
from maddening.core.static_data import StaticArray
from typing_extensions import NotRequired, ReadOnly, TypedDict

from mime.core import MimeNode, NodeFields, as_run_array, jnp, node_fields, node_input, quaternion, typed_view
from mime.core._quaternion import safe_norm
from mime.data import Asset, asset_bytes
from mime.surrogates import SURROGATES, CholeskyMLP, predict_resistance, read_cholesky_mlp

#: An offset below this many units in the last place of the tube's radius counts as on the axis: 1.6 micrometres in
#: A's 3.175 mm tube in float32, 3e-15 m in float64. Well above the rounding of a position (turning the world moves
#: a point on the axis off it by a few units in the last place of its distance from the tube's reference point), so
#: that a point on the axis stays on it when the world is turned.
AXIS_ULPS = 4096
#: The lubrication's smallest gap, in metres: the 0.2 line's ``delta_floor_mm = 1e-6``.
GAP_FLOOR = 1e-9
#: The non-dimensional gap the surrogate's feature ``log(gap / R)`` is floored at, as in the 0.2 line.
FEATURE_GAP_FLOOR = 1e-3
#: How far into the room between body and wall the offset may go before it is clamped, as in the 0.2 line.
CLAMP_FRACTION = 0.95
#: Where the delivered resistance is taken about: the body's centre of mass, or (test-only) the end of its axis.
REFERENCE_POINTS = ("centre", "end")


class SurrogateDragState(TypedDict, closed=True):
    resistance: ReadOnly[Float[Array, "6 6"]]  # noqa: F722 - a jaxtyping shape
    background_load: ReadOnly[Float[Array, "6"]]
    clamp_fired: ReadOnly[Float[Array, ""]]


class SurrogateDragInputs(TypedDict, closed=True):
    pose: NotRequired[ReadOnly[Float[Array, "7"]]]


def _pair(frame: Any) -> Any:
    """``diag(frame, frame)``: a 3 x 3 rotation acting on a [translation; rotation] pair."""
    held = jnp.zeros((6, 6), dtype=frame.dtype)
    return held.at[:3, :3].set(frame).at[3:, 3:].set(frame)


class SurrogateDragNode(MimeNode):
    """The surrogate's resistance (6 x 6, world frame, SI) at the body's pose, with lubrication when
    ``lubrication_epsilon`` is given; the background load, zero (the medium is at rest: no flow dynamics is built
    yet, so the ambient velocity at the body is zero); and ``clamp_fired``, 1 where the offset was clamped.

    The weights are the packaged asset ``weights``, read once their sha256 is ``weights_sha256``, and held as static
    arrays: the derived hash covers their contents. Every number the step reads is a ``params`` leaf."""

    def __init__(self, name: str, timestep: float, *, weights: str, weights_sha256: str, viscosity: float,
                 helix_wavenumber: float, length: float, radius: float, vessel_centre: Sequence[float],
                 vessel_axis: Sequence[float], vessel_radius: float, lubrication_epsilon: float | None = None,
                 reference_point: str = "centre") -> None:
        if reference_point not in REFERENCE_POINTS:
            raise ValueError(f"reference_point {reference_point!r} is not one of {REFERENCE_POINTS}")
        if weights not in SURROGATES:
            raise ValueError(f"no surrogate is declared for the asset {weights!r}; those declared are "
                             f"{sorted(SURROGATES)}")
        super().__init__(name, timestep, weights=weights, weights_sha256=weights_sha256, viscosity=viscosity,
                         helix_wavenumber=helix_wavenumber, length=length, radius=radius,
                         vessel_centre=tuple(vessel_centre), vessel_axis=tuple(vessel_axis),
                         vessel_radius=vessel_radius, lubrication_epsilon=lubrication_epsilon,
                         reference_point=reference_point)
        self._network: CholeskyMLP = read_cholesky_mlp(asset_bytes(Asset(weights, weights_sha256)))
        self._outer = 1.0 + SURROGATES[weights].modulation

    @property
    @override
    def static_data(self) -> dict[str, Any]:
        return {f"mlp_{key}": StaticArray(value) for key, value in self._network.arrays().items()}

    @override
    def initial_state(self) -> NodeFields:
        return node_fields(SurrogateDragState(resistance=as_run_array(jnp.zeros((6, 6))),
                                              background_load=as_run_array(jnp.zeros(6)),
                                              clamp_fired=as_run_array(jnp.zeros(()))))

    @override
    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        return {"pose": node_input("m; 1", (7,))}

    def frame(self, p: Mapping[str, Any], position: Array, orientation: Array) -> tuple[Array, Array]:
        """The body's distance from the tube's axis, and the canonical frame as columns ``[n, a x n, a]``: ``a`` the
        tube's axis, ``n`` the unit offset across it (on the axis, the body's own x axis across the tube, or its y
        axis where x lies within 60 degrees of the tube's axis)."""
        dtype = position.dtype
        axis = jnp.asarray(p["vessel_axis"], dtype=dtype)
        axis = axis / safe_norm(axis)
        relative = position - jnp.asarray(p["vessel_centre"], dtype=dtype)
        radial = relative - jnp.dot(relative, axis) * axis
        squared = jnp.sum(radial * radial)
        resolution = AXIS_ULPS * float(jnp.finfo(dtype).eps) * p["vessel_radius"]
        away = squared > resolution ** 2
        offset = jnp.where(away, jnp.sqrt(jnp.where(away, squared, 1.0)), 0.0)
        body = quaternion.rotation_matrix(orientation)
        across_x = body[:, 0] - jnp.dot(body[:, 0], axis) * axis
        across_y = body[:, 1] - jnp.dot(body[:, 1], axis) * axis
        across = jnp.where(jnp.sum(across_x * across_x) > 0.25, across_x, across_y)
        direction = jnp.where(away, radial, across)
        normal = direction / safe_norm(direction)
        return offset, jnp.stack([normal, jnp.cross(axis, normal), axis], axis=1)

    def canonical_resistance(self, p: Mapping[str, Any], offset: Array) -> tuple[Array, Array]:
        """The surrogate's non-dimensional resistance in the canonical frame at ``offset``, and whether the offset was
        clamped (1 or 0)."""
        dtype = offset.dtype
        reference = p["radius"]
        tube = p["vessel_radius"] / reference
        kappa = 1.0 / tube
        offset_nd = offset / reference
        largest = jnp.maximum((tube - self._outer) * CLAMP_FRACTION, 0.0)
        beyond = offset_nd > largest
        clamped = offset_nd * jnp.where(beyond, largest / jnp.maximum(offset_nd, 1e-10), 1.0)
        gap = jnp.maximum(tube - (clamped + self._outer), FEATURE_GAP_FLOOR)
        nu = p["helix_wavenumber"]
        base = [nu, p["length"] / reference, kappa, clamped, 0.0, jnp.log(gap)]
        squared = [nu ** 2, kappa ** 2] if self._network.squared_features else []
        features = jnp.stack([jnp.asarray(f, dtype=dtype) for f in (*base, *squared)])
        return predict_resistance(self._network, features), beyond.astype(dtype)

    def lubrication(self, p: Mapping[str, Any], offset: Array, frame: Array) -> Array:
        """The lubrication correction ``w R_lub`` (SI) at the body's offset, in the wall frame the canonical frame
        gives: ``n`` its first column, the axis its third."""
        dtype = offset.dtype
        mu, a = p["viscosity"], self._outer * p["radius"]
        gap = jnp.maximum(p["vessel_radius"] - (offset + a), GAP_FLOOR)
        weight = jnp.exp(-gap / p["lubrication_epsilon"])
        normal_term = 6.0 * math.pi * mu * a ** 2 / gap
        logarithm = jnp.maximum(jnp.log(a / gap), 0.0)
        tangential = 6.0 * math.pi * mu * a * (8.0 / 15.0) * logarithm
        rotational = 8.0 * math.pi * mu * a ** 3 * (2.0 / 5.0) * logarithm
        normal, along = frame[:, 0], frame[:, 2]
        across = jnp.cross(normal, along)
        wall = jnp.outer(along, along) + jnp.outer(across, across)
        correction = jnp.zeros((6, 6), dtype=dtype)
        correction = correction.at[:3, :3].set(normal_term * jnp.outer(normal, normal) + tangential * wall)
        return weight * correction.at[3:, 3:].set(rotational * wall)

    def end(self, p: Mapping[str, Any], orientation: Array) -> Array:
        """``s``: +1 where the body's x axis (the helix's, along which ``zeta`` runs) points along the tube's axis or
        across it, -1 where against it; +1 with the resistance about the end."""
        dtype = orientation.dtype
        if self.constants()["reference_point"] == "end":
            return jnp.ones((), dtype=dtype)
        axis = jnp.asarray(p["vessel_axis"], dtype=dtype)
        along = jnp.dot(quaternion.rotation_matrix(orientation)[:, 0], axis)
        return jnp.where(along >= 0, 1.0, -1.0).astype(dtype)

    def about_the_centre(self, p: Mapping[str, Any], canonical: Array) -> Array:
        """The non-dimensional ``canonical`` resistance moved from the end ``zeta = 0`` to the centre, ``L / 2``
        along the canonical frame's axis: ``T^T R T``, ``T = [[I, [d]x], [0, I]]``; unchanged with the resistance
        about the end."""
        if self.constants()["reference_point"] == "end":
            return canonical
        dtype = canonical.dtype
        half = jnp.asarray(p["length"] / (2.0 * p["radius"]), dtype=dtype)
        zero = jnp.zeros((), dtype=dtype)
        cross = jnp.stack([jnp.stack([zero, -half, zero]), jnp.stack([half, zero, zero]),
                           jnp.stack([zero, zero, zero])])
        move = jnp.eye(6, dtype=dtype).at[:3, 3:].set(cross)
        return move.T @ canonical @ move

    @override
    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        given = typed_view(SurrogateDragInputs, boundary_inputs)
        pose = given.get("pose", as_run_array(jnp.asarray((0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))))
        offset, frame = self.frame(p, pose[:3], pose[3:])
        side = self.end(p, pose[3:])
        frame = frame * jnp.stack([jnp.ones_like(side), side, side])
        canonical, clamp_fired = self.canonical_resistance(p, offset)
        canonical = self.about_the_centre(p, canonical)
        turn = _pair(frame)
        nondimensional = turn @ canonical @ turn.T
        mu, reference = p["viscosity"], p["radius"]
        blocks = jnp.stack([jnp.stack([mu * reference, mu * reference ** 2]),
                            jnp.stack([mu * reference ** 2, mu * reference ** 3])]).astype(offset.dtype)
        resistance = jnp.kron(blocks, jnp.ones((3, 3), dtype=offset.dtype)) * nondimensional
        if self.constants()["lubrication_epsilon"] is not None:
            resistance = resistance + self.lubrication(p, offset, frame)
        ambient = jnp.zeros(6, dtype=offset.dtype)
        return node_fields(SurrogateDragState(resistance=resistance, background_load=resistance @ ambient,
                                              clamp_fired=clamp_fired))


__all__ = ["AXIS_ULPS", "GAP_FLOOR", "REFERENCE_POINTS", "SurrogateDragInputs", "SurrogateDragNode",
           "SurrogateDragState"]
