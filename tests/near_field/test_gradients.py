"""Gate 5, gradients through the dense route (float64, away from the reach's blend and the budget's choice, their
value-keyed boundaries): the squeeze's coupled resistance ``R_aa - R_ab`` of two spheres, forward (``jax.jvp``) and
reverse (``jax.grad``) in the neighbour's separation and in its radius at gaps of one and a quarter radius, against
central differences, within ten times the difference's own resolution (the difference of two step sizes); linear in
the loss's scale at 1e-12 and 1e6 (S3 found JAX's stock GMRES returning exact zeros or 57 % errors there; the dense LU
has no such stop, and W3b's GMRES is held to the same line); and the reverse gradient of a run of five steps through
``run_scan``, a sphere pulled towards another, in that one's radius. The control: a ``stop_gradient`` on the source's
placement where the cross-kernel sees it, which leaves the values alone and moves the gradient past the line (S3: 17 %
off)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from near_field_cases import NODE, bodies, joint_node, mrs, precision, spheres, start_inputs
from rigid_oracles import ConstantWrench

from mime.build._maddening import parameter_leaves
from mime.nodes.hydro.near_field import MRSOperator, Placed
from mime.nodes.hydro.near_field._mrs import Quadrature

SPACING = 0.4e-3
RADIUS_LEAF = "m1_shape_radius"


def squeeze_of(built):
    node, params = joint_node(built), dict(parameter_leaves(built.graph)[NODE])
    inputs = start_inputs(built)
    poses = jnp.stack([jnp.asarray(inputs[f"pose_{i}"]) for i in range(2)])

    def squeeze(shift, radius):
        held = {**params, RADIUS_LEAF: radius}
        found = node.solved(node.constants(held), poses.at[1, 0].add(shift))[0]
        grand = 0.5 * (found + found.T)
        return (grand[0, 0] - grand[0, 6]) / grand[0, 0]

    return squeeze, jnp.asarray(params[RADIUS_LEAF])


def central(f, x, steps):
    return [(float(f(x + h)) - float(f(x - h))) / (2.0 * h) for h in steps]


@pytest.mark.parametrize("gap", [1e-3, 0.25e-3])
def test_forward_and_reverse_match_central_differences_in_separation_and_radius_at_every_loss_scale(gap):
    with precision("float64"):
        squeeze, radius = squeeze_of(bodies("gradient", spheres(gap, 2), mrs(SPACING)).build())
        rows = {}
        for name, f, x, steps in (("separation", lambda s: squeeze(s, radius), 0.0, (4e-8, 2e-8)),
                                  ("radius", lambda r: squeeze(0.0, r), radius, (4e-8, 2e-8))):
            reverse = float(jax.grad(f)(jnp.asarray(x)))
            forward = float(jax.jvp(f, (jnp.asarray(x),), (jnp.asarray(1.0),))[1])
            differences = central(f, jnp.asarray(x), steps)
            scaled = [float(jax.grad(lambda y, c=c: c * f(y))(jnp.asarray(x))) / c for c in (1e-12, 1e6)]
            rows[name] = (reverse, forward, differences, scaled)
    for name, (reverse, forward, differences, scaled) in rows.items():
        resolution = abs(differences[0] - differences[1])
        print(f"\ngap {gap * 1e3:g} mm, d/d({name}): reverse {reverse:.10g}, forward {forward:.10g}, central "
              f"{differences[1]:.10g} (resolution {resolution:.2g}); over the loss's scale {scaled}")
        assert abs(reverse - differences[1]) <= 10.0 * resolution + 1e-9 * abs(reverse)
        assert abs(forward - reverse) <= 1e-9 * abs(reverse)
        assert all(abs(s - reverse) <= 1e-9 * abs(reverse) for s in scaled)


def pulled():
    experiment = bodies("pulled", spheres(0.5e-3, 2), mrs(SPACING), mass=2e-8, inertia=2e-17)
    experiment.attach(ConstantWrench(value=(-2e-7, 0.0, 0.0, 0.0, 0.0, 0.0)), body="s1")
    return experiment.build()


def run_gradient(built, plant: bool = False, monkeypatch=None) -> tuple[float, list[float]]:
    """The reverse gradient of the separation after five steps in the neighbour's radius, and its central differences
    through the parameter tree at two steps."""
    graph = built.graph
    base = graph.params

    def separation(radius):
        held = jax.tree.map(lambda x: x, base)
        held["nodes"][NODE][RADIUS_LEAF] = radius
        graph.reset_state()
        final = graph.run_scan(5, params=held)
        return 1e3 * (final["s1"]["pose"][0] - final["s0"]["pose"][0])

    radius = jnp.asarray(base["nodes"][NODE][RADIUS_LEAF])
    found = float(jax.grad(separation)(radius))
    differences = central(separation, radius, (4e-8, 2e-8))
    graph.reset_state()
    return found, differences


def test_the_reverse_gradient_through_five_scanned_steps_matches_central_differences(monkeypatch):
    with precision("float64"):
        found, differences = run_gradient(pulled())
    resolution = abs(differences[0] - differences[1])
    print(f"\nfive steps: reverse {found:.10g}, central {differences[1]:.10g} (resolution {resolution:.2g})")
    assert abs(found - differences[1]) <= 10.0 * resolution + 1e-9 * abs(found)


def test_control_a_stop_gradient_where_the_cross_kernel_sees_the_radius_moves_the_gradient(monkeypatch):
    held = MRSOperator.cross

    def cross(self, placed, targets, reach):
        stopped = Placed(*(jax.lax.stop_gradient(getattr(placed, name)) for name in
                           ("force", "centroid", "area", "eps_squared", "centre")),
                         own=Quadrature(jax.lax.stop_gradient(placed.own.points),
                                        jax.lax.stop_gradient(placed.own.weights)))
        return held(self, stopped, targets, reach)

    with precision("float64"):
        squeeze, radius = squeeze_of(bodies("planted", spheres(0.25e-3, 2), mrs(SPACING)).build())
        f = lambda r: squeeze(0.0, r)  # noqa: E731
        clean = float(jax.grad(f)(radius))
        value = float(f(radius))
        monkeypatch.setattr(MRSOperator, "cross", cross)
        planted = float(jax.grad(f)(radius))
        planted_value = float(f(radius))
    print(f"\nthe stop_gradient on the cross-kernel's source: gradient {planted:.6g} against {clean:.6g} "
          f"({abs(planted / clean - 1):.1%} off), the value moved {abs(planted_value - value):.2g}")
    assert planted_value == value
    assert abs(planted / clean - 1.0) > 0.01
