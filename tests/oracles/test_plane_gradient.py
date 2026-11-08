"""The derivative through the MRS drag's node above a plane wall (the plane wall kind, ``mime.walls.PlaneWall``,
declared ``exact``): forward and reverse mode against a central difference, in float64, in the plane's offset (a leaf a
study may move, which lubrication's opening of a contact moves), the blob, the viscosity and the sphere's radius, at
poses away from the node's value-keyed choices (the image at the force points, which makes none; and integrated over the cells near the
wall at a gap where every cell within the reach is in the budget, so that the budget's choice moves no column).

The poses are the sphere unturned, whose samples tie in height pairwise across its tensor grid's symmetry (the ties
MIME-ANO-143 found the tube's table read differentiating with the wrong sign), and turned. The image is a closed form
with no table and no sign test, so no tie should move it [inference]: this is where that is shown.

The difference: Richardson's extrapolation of two central differences, each leaf stepped by a millionth of its size;
its truncation estimated by a third step. The loss weighs every entry of the resistance by a fixed matrix of order one
over the entry's scale, so that no entry's derivative is lost in another's.

**The control**, failing on every run (nothing is drawn): a ``stop_gradient`` on the offset where the image reads it
(``mime.nodes.hydro.mrs._plane.image_terms``), which leaves the value and takes the offset's derivative away."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from plane_cases import RADIUS, drag_node, sphere_above_plane

from mime.build._maddening import node_state, parameter_leaves
from mime.nodes.hydro.mrs import _plane

#: The leaves differentiated, each a scalar.
LEAVES = ("wall_offset", "blob", "viscosity", "shape_radius")
#: How far autodiff may lie from the difference, of the derivative's scale: the difference's truncation and rounding
#: measured at most 1.4e-8 of it (measured when the wall kind was built, in its pull request), autodiff within the difference's own resolution.
TOLERANCE = 1e-6
TURNED = (math.cos(0.4), 0.3 * math.sin(0.4), -0.5 * math.sin(0.4), math.sqrt(1 - 0.34) * math.sin(0.4))
#: The cases: the sphere's force spacing (in its radius), its gap (in its radius), wall_refinement and orientation.
CASES = {
    "point, unturned": (0.4, 1.0, None, (1.0, 0.0, 0.0, 0.0)),
    "integrated, turned": (0.4, 0.4, 2, TURNED),
}
#: The drag's blob in the cases: its default, half the force spacing, whose quadrature is the coarsest the rule gives
#: (the derivative's agreement does not rest on the blob: at a blob of 0.1 and a spacing of 0.3 radii it was within
#: 1.4e-8 of the scale in every leaf, measured when the wall kind was built).
BLOB = 0.5


def loss_of(case):
    """The case's node, its leaves, and the loss of the resistance at its pose as a function of the leaves."""
    spacing, gap, refinement, orientation = case
    built = sphere_above_plane(spacing, 2.0, blob=BLOB, wall_refinement=refinement).build()
    node = drag_node(built)
    state = node_state(built.graph, "drag:ball")
    leaves = parameter_leaves(built.graph)["drag:ball"]
    norm = math.sqrt(sum(c * c for c in orientation))
    pose = jnp.asarray((0.0, 0.0, RADIUS * (1.0 + gap), *(c / norm for c in orientation)))
    scale = np.asarray(node.update(state, {"pose": pose}, 1e-3, params=leaves)["resistance"])
    diagonal = np.sqrt(np.abs(np.outer(np.diag(scale), np.diag(scale))))
    weights = jnp.asarray((1.0 + 0.1 * np.arange(36).reshape(6, 6) / 36.0) / diagonal)

    def loss(values):
        held = {**leaves, **dict(zip(LEAVES, values, strict=True))}
        return jnp.sum(weights * node.update(state, {"pose": pose}, 1e-3, params=held)["resistance"])

    return jnp.asarray([leaves[name] for name in LEAVES]), jax.jit(loss)


def richardson(loss, at, index, step):
    def central(h):
        e = jnp.zeros_like(at).at[index].set(h)
        return (float(loss(at + e)) - float(loss(at - e))) / (2.0 * h)

    coarse, fine, finest = central(step), central(step / 2.0), central(step / 4.0)
    value = (4.0 * fine - coarse) / 3.0
    return value, abs(value - (4.0 * finest - fine) / 3.0)


#: Each leaf's size, for its step: the offset's is the sphere's radius (the plane passes through the origin).
SIZES = {"wall_offset": RADIUS}


def compared(case):
    at, loss = loss_of(case)
    reverse = np.asarray(jax.jit(jax.grad(loss))(at))
    forward = np.asarray(jax.jit(jax.jacfwd(loss))(at))
    found = []
    for i, name in enumerate(LEAVES):
        size = SIZES.get(name, abs(float(at[i])))
        difference, truncation = richardson(loss, at, i, 1e-6 * size)
        scale = max(abs(difference), abs(reverse[i]))
        found.append((name, reverse[i], forward[i], difference, truncation, scale))
    return found


@pytest.mark.parametrize("name", sorted(CASES))
def test_forward_and_reverse_equal_the_difference_in_the_offset_blob_viscosity_and_radius(name):
    for leaf, reverse, forward, difference, truncation, scale in compared(CASES[name]):
        print(f"\n{name} {leaf}: reverse {reverse:.12g}, forward {forward:.12g}, difference {difference:.12g} "
              f"(truncation {truncation:.2g}), |AD - FD| / scale {abs(reverse - difference) / scale:.2g}")
        assert scale > 0.0 and truncation <= 0.1 * TOLERANCE * scale, (leaf, truncation, scale)
        assert abs(reverse - difference) <= TOLERANCE * scale, (leaf, reverse, difference)
        assert abs(forward - difference) <= TOLERANCE * scale, (leaf, forward, difference)


def test_control_a_stop_gradient_on_the_offset_fails_the_offsets_derivative(monkeypatch):
    held = _plane.image_terms

    def stopped(targets, sources, parameters, eps_squared):
        return held(targets, sources, {**parameters, "offset": jax.lax.stop_gradient(parameters["offset"])},
                    eps_squared)

    monkeypatch.setattr(_plane, "image_terms", stopped)
    found = {leaf: (reverse, difference, scale) for leaf, reverse, _, difference, _, scale
             in compared(CASES["point, unturned"])}
    reverse, difference, scale = found["wall_offset"]
    print(f"\nthe offset stopped: reverse {reverse:.6g}, difference {difference:.6g}")
    assert abs(reverse - difference) > 1e3 * TOLERANCE * scale
