"""Gate 5 on the iterative route (first step (f): spike S3's wrapper results in the repository), float64, away from the
reach's blend and the budget's choice: the squeeze's coupled resistance ``R_aa - R_ab`` of two spheres past the takeover
line, forward (``jax.jvp``) and reverse (``jax.grad``) through the ``custom_linear_solve`` wrapper in the neighbour's
separation and in its radius at gaps of one and a quarter radius, against central differences (within ten times their
own resolution, the difference of two step sizes) and the dense route's autodiff; linear in the loss's scale at 1e-12
and 1e6; and the reverse gradient of runs of five and ten steps through ``run_scan``, a sphere pulled towards another,
in that one's radius, against central differences.

The controls, each failing every run: JAX's stock GMRES in the wrapper's place (its derivative solves stop at the
primal's absolute tolerance: S3 measured an exact zero, 57 % and 2.6 % off at small scales), at the loss scale 1e-12
off by a thousand times the scale test's line or more in reverse and forward mode (measured at a gap of a radius: an
exact zero and 2.3 % in the separation, 0.95 % and 2.0 % in the radius, whose derivative passes partly outside the
solve); and a ``stop_gradient`` on the source's placement where the
cross-kernel sees it, which leaves the values alone and moves the gradient past the line (S3: 17 %)."""

import jax
import jax.numpy as jnp
import pytest
from jax.scipy.sparse.linalg import gmres as stock_gmres
from near_field_cases import NODE, bodies, joint_node, mrs, precision, spheres, start_inputs
from rigid_oracles import ConstantWrench

from mime.build._maddening import parameter_leaves
from mime.interactions import NearField
from mime.nodes.hydro.near_field import MRSOperator, Placed
from mime.nodes.hydro.near_field import _contract
from mime.nodes.hydro.near_field._gmres import REPORT_SIZE
from mime.nodes.hydro.near_field._mrs import Quadrature

SPACING = 0.4e-3
RADIUS_LEAF = "m1_shape_radius"
ITERATIVE = {"near_field": NearField(solve="gmres")}
#: The scale test's line: the derivative over the loss's (or the tangent's) scale at 1e-12 and 1e6, relative.
SCALE_LINE = 1e-9


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


def compiled(f):
    """``f``'s value, its reverse derivative of ``c f`` and its tangent along ``c``, each compiled once, ``c`` traced."""
    value = jax.jit(f)
    reverse = jax.jit(jax.grad(lambda y, c: c * f(y)))
    tangent = jax.jit(lambda y, c: jax.jvp(f, (y,), (c,))[1])
    return value, reverse, tangent


def derivatives(gap: float, settings: dict, full: bool = True) -> dict:
    """Each derivative of the squeeze: reverse and forward, and over the loss's and the tangent's scale at 1e-12 and
    1e6; with ``full``, central differences at two steps too."""
    squeeze, radius = squeeze_of(bodies("gradient", spheres(gap, 2), mrs(SPACING, **settings)).build())
    rows = {}
    for name, f, x in (("separation", lambda s: squeeze(s, radius), jnp.asarray(0.0)),
                       ("radius", lambda r: squeeze(0.0, r), radius)):
        value, reverse, tangent = compiled(f)
        row = dict(reverse=float(reverse(x, 1.0)), forward=float(tangent(x, 1.0)),
                   scaled=[float(reverse(x, c)) / c for c in (1e-12, 1e6)],
                   tangents=[float(tangent(x, c)) / c for c in (1e-12, 1e6)])
        if full:
            row["central"] = central(value, x, (4e-8, 2e-8))
        rows[name] = row
    return rows


@pytest.mark.parametrize("gap", [1e-3, 0.25e-3])
def test_forward_and_reverse_match_central_differences_and_the_dense_route_at_every_loss_scale(gap):
    """The scale arm (the loss's and the tangent's scale at 1e-12, judged at ``SCALE_LINE``) is what guards the cold
    start (decision 10): every solve warm-started from the primal's answer passes central differences, forward against
    reverse and the dense route, and fails only it (measured: reverse 7.3 % and 4.8 % off at 1e-12 in the separation
    and the radius at a gap of a radius). It is never trimmed as redundant with the other three."""
    with precision("float64"):
        rows = derivatives(gap, ITERATIVE)
        dense = derivatives(gap, {}, full=False)
    for name, row in rows.items():
        differences = row["central"]
        resolution = abs(differences[0] - differences[1])
        reverse = row["reverse"]
        print(f"\ngap {gap * 1e3:g} mm, d/d({name}): reverse {reverse:.10g}, forward {row['forward']:.10g}, the dense "
              f"route's {dense[name]['reverse']:.10g}, central {differences[1]:.10g} (resolution {resolution:.2g}); "
              f"over the loss's scale {row['scaled']}, over the tangent's {row['tangents']}")
        assert abs(reverse - differences[1]) <= 10.0 * resolution + 1e-9 * abs(reverse)
        assert abs(row["forward"] - reverse) <= 1e-9 * abs(reverse)
        assert abs(dense[name]["reverse"] - reverse) <= 1e-8 * abs(reverse)
        assert all(abs(s - reverse) <= SCALE_LINE * abs(reverse) for s in (*row["scaled"], *row["tangents"]))


def pulled():
    experiment = bodies("pulled", spheres(0.5e-3, 2), mrs(SPACING, **ITERATIVE), mass=2e-8, inertia=2e-17)
    experiment.attach(ConstantWrench(value=(-2e-7, 0.0, 0.0, 0.0, 0.0, 0.0)), body="s1")
    return experiment.build()


def run_gradient(built, steps: int) -> tuple[float, list[float]]:
    """The reverse gradient of the separation after ``steps`` steps in the neighbour's radius, through ``run_scan``,
    and its central differences through the parameter tree at two steps."""
    graph = built.graph
    base = graph.params

    def separation(radius):
        held = jax.tree.map(lambda x: x, base)
        held["nodes"][NODE][RADIUS_LEAF] = radius
        graph.reset_state()
        final = graph.run_scan(steps, params=held)
        return 1e3 * (final["s1"]["pose"][0] - final["s0"]["pose"][0])

    radius = jnp.asarray(base["nodes"][NODE][RADIUS_LEAF])
    found = float(jax.grad(separation)(radius))
    differences = central(separation, radius, (4e-8, 2e-8))
    graph.reset_state()
    return found, differences


@pytest.mark.parametrize("steps", [5, 10])
def test_the_reverse_gradient_through_scanned_steps_matches_central_differences(steps):
    with precision("float64"):
        built = pulled()
        assert joint_node(built).iterative
        found, differences = run_gradient(built, steps)
    resolution = abs(differences[0] - differences[1])
    print(f"\n{steps} steps: reverse {found:.10g}, central {differences[1]:.10g} (resolution {resolution:.2g})")
    assert abs(found - differences[1]) <= 10.0 * resolution + 1e-9 * abs(found)


def stock(matrix, b, tap, precondition, transposed, stopping):
    """The plant: JAX's stock GMRES in the wrapper's place, each column its own ``custom_linear_solve`` whose tolerance
    is fixed from the primal right-hand side, with the same preconditioner, tolerance, restart and cap."""
    del transposed

    def column(rhs):
        return stock_gmres(lambda v: matrix @ v, rhs, tol=stopping.tolerance, atol=0.0, restart=stopping.restart,
                           maxiter=stopping.cap // stopping.restart, M=lambda r: precondition(r[:, None])[:, 0],
                           solve_method="incremental")[0]

    return jax.vmap(column, in_axes=1, out_axes=1)(b), jnp.zeros(REPORT_SIZE, dtype=b.dtype) + 0.0 * tap


@pytest.mark.parametrize("gap", [1e-3, 0.25e-3])
def test_control_jax_stock_gmres_in_the_wrappers_place_fails_the_scale_test(gap, monkeypatch):
    with precision("float64"):
        clean = derivatives(gap, ITERATIVE, full=False)
        monkeypatch.setattr(_contract, "reported_solve", stock)
        planted = derivatives(gap, ITERATIVE, full=False)
    for name in clean:
        reverse = clean[name]["reverse"]
        small_reverse, small_forward = planted[name]["scaled"][0], planted[name]["tangents"][0]
        print(f"\ngap {gap * 1e3:g} mm, d/d({name}): the stock solver at the loss scale 1e-12 {small_reverse:.6g} in "
              f"reverse and {small_forward:.6g} forward, against {reverse:.6g}")
        # The gate's line is 1e-9 of the derivative (the scale test above); the plant must fail it, by orders.
        assert abs(small_reverse / reverse - 1.0) > 1e3 * SCALE_LINE and abs(small_forward / reverse - 1.0) > 1e3 * SCALE_LINE


def test_control_a_stop_gradient_where_the_cross_kernel_sees_the_radius_moves_the_gradient(monkeypatch):
    held = MRSOperator.cross

    def cross(self, placed, targets, reach):
        stopped = Placed(*(jax.lax.stop_gradient(getattr(placed, name)) for name in
                           ("force", "centroid", "area", "eps_squared", "centre")),
                         own=Quadrature(jax.lax.stop_gradient(placed.own.points),
                                        jax.lax.stop_gradient(placed.own.weights)))
        return held(self, stopped, targets, reach)

    with precision("float64"):
        squeeze, radius = squeeze_of(bodies("planted", spheres(0.25e-3, 2), mrs(SPACING, **ITERATIVE)).build())
        f = lambda r: squeeze(0.0, r)  # noqa: E731
        clean, value = float(jax.grad(f)(radius)), float(f(radius))
        monkeypatch.setattr(MRSOperator, "cross", cross)
        planted, planted_value = float(jax.grad(f)(radius)), float(f(radius))
    print(f"\nthe stop_gradient on the cross-kernel's source: gradient {planted:.6g} against {clean:.6g} "
          f"({abs(planted / clean - 1):.1%} off), the value moved {abs(planted_value - value):.2g}")
    assert planted_value == value
    assert abs(planted / clean - 1.0) > 0.01
