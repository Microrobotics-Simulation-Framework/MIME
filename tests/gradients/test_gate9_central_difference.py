"""Gate 9, float64: a study's derivative, reverse and forward, against a central difference that rebuilds (each side
the experiment written again with the value moved, built and run through ``built.run``), at a relative step of
``H = 1e-4``, at loss scales 1e-12 and 1e6; its line ``|AD/CD - 1| <= 1e-7``.

The line from measurement (probe run 42763593758, before this gate): at ``h = 1e-4`` the difference agreed with the
derivative through the built graph's parameters to 5.7e-9 on the pulled sphere's radius and 7.4e-9 to 1.1e-8 on the
tube's viscosity (its truncation, and at the tube's default tolerance its solve's); about ten times under the line.

The cases: the gently pulled sphere's radius at 1.05 mm, inside its declared range [1.0, 1.15] mm, so that each side of
the difference builds the range's force points (away from every value-keyed boundary); and the periodic tube's
viscosity and its drive's amplitude (D/h 8, IMEX Euler under Stokes, 20 steps of 20 ms), its entry's ``unverified``
status acknowledged, since this gate is what checks it in these values. The step is far above ``tol x tau / dt``.

**Controls** (``gradient_plants.py``, each switched on by ``MIME_PLANT_GRADIENT`` for the gate kit's detection rate): a value
baked at build (``baked``) and ``stop_gradient`` on a leaf the path uses (``stopped``) each move the derivative by its
leaf's share, far past the line."""

from __future__ import annotations

import pytest
from gradient_rigs import RADIUS, STEPS, TUBE_STEPS, flux, gentle, pose_loss, tube
from gradient_plants import GRADIENT, from_environment, gradient_plant

from mime.study import Study, along
from mime.units import mm, mPa_s, ms

H = 1e-4
LINE = 1e-7
SCALES = (1e-12, 1e6)
MODES = ("reverse", "forward")
#: The tube's entry is unverified in its geometry; this gate checks its viscosity and drive (the entry's own record).
ACKNOWLEDGED = {"water.flow": "gate 9 checks the derivative in the viscosity and the drive against a central "
                              "difference that rebuilds; the entry's status is its geometry's"}


def sphere_loss(radius: float) -> float:
    return float(pose_loss(gentle(radius * 1e3).build().run(ms(STEPS)).final))


def tube_flux(viscosity: float, amplitude: float) -> float:
    return float(flux(tube(viscosity * 1e3, amplitude).build().run(ms(20.0 * TUBE_STEPS)).final))


def central(loss, theta: float) -> float:
    return (loss(theta * (1 + H)) - loss(theta * (1 - H))) / (2 * H * theta)


def derivatives(experiment, axis, value, of, wrt, duration, acknowledged=None):
    """The study's derivative at the one point, by mode and scale."""
    found = {}
    for mode in MODES:
        for scale in SCALES:
            result = Study(experiment).sweep(along({axis: [value]}), duration=duration).gradient(
                lambda final, scale=scale: scale * of(final), wrt=wrt, mode=mode, acknowledged=acknowledged)
            found[(mode, scale)] = {path: float(held) / scale for path, held in result.member(0).gradient.items()}
    return found


@pytest.fixture(scope="module")
def sphere():
    with pytest.MonkeyPatch.context() as patch:
        gradient_plant(patch, from_environment(GRADIENT))
        return central(sphere_loss, 1.05e-3), derivatives(gentle(1.05), RADIUS, mm(1.05), pose_loss, None, ms(STEPS))


@pytest.fixture(scope="module")
def flow():
    with pytest.MonkeyPatch.context() as patch:
        gradient_plant(patch, from_environment(GRADIENT))
        differences = {"water.viscosity": central(lambda mu: tube_flux(mu, 0.01), 1e-3),
                       "body_force:water.amplitude": central(lambda a: tube_flux(1e-3, a), 0.01)}
        return differences, derivatives(tube(), "water.viscosity", mPa_s(1.0), flux, list(differences),
                                        ms(20.0 * TUBE_STEPS), ACKNOWLEDGED)


def misses(differences, derivatives) -> list[str]:
    return [f"{path} {mode} at {scale:g}: {found[path]:.12e} against {difference:.12e}, "
            f"{abs(found[path] / difference - 1):.2e}"
            for (mode, scale), found in derivatives.items() for path, difference in differences.items()
            if not abs(found[path] / difference - 1) <= LINE]


def test_the_spheres_derivative_in_its_radius_is_the_rebuilt_central_difference(sphere):
    difference, found = sphere
    assert misses({RADIUS: difference}, found) == []


def test_the_tubes_derivative_in_its_viscosity_and_drive_is_the_rebuilt_central_difference(flow):
    differences, found = flow
    assert misses(differences, found) == []


@pytest.mark.parametrize("plant", ["baked", "stopped"])
def test_control_a_baked_value_or_a_stopped_leaf_fails_the_gate_on_the_sphere(plant):
    with pytest.MonkeyPatch.context() as patch:
        gradient_plant(patch, plant)
        result = Study(gentle(1.05)).sweep(along({RADIUS: [mm(1.05)]}), duration=ms(STEPS)).gradient(
            pose_loss, mode="reverse")
        found = {("reverse", 1.0): {RADIUS: float(result.member(0).gradient[RADIUS])}}
        difference = central(sphere_loss, 1.05e-3)
    assert misses({RADIUS: difference}, found), f"the {plant} plant passes the gate"


def test_the_gates_cases_run_inside_their_validity_lines_but_the_tubes_bypass(sphere, flow):
    """The operating point read before the comparison: the sphere's run raises nothing, and the tube's raises its
    bypass line alone (D/h 8 is past it, measured 2.6e-3 against the line's 1e-3; a mechanism gate, which compares the
    computed model with itself), not its Stokes line (its drive keeps its Reynolds number under one)."""
    sphere_result = Study(gentle(1.05)).sweep(along({RADIUS: [mm(1.05)]}), duration=ms(STEPS)).gradient(pose_loss)
    assert sphere_result.member(0).flags == ()
    flow_result = Study(tube()).sweep(along({"water.viscosity": [mPa_s(1.0)]}), duration=ms(20.0 * TUBE_STEPS)) \
        .gradient(flux, acknowledged=ACKNOWLEDGED)
    assert [flag.path for flag in flow_result.member(0).flags] == ["water.flow.bypass_flagged"]


def test_the_studys_derivative_is_the_graphs_own_through_its_parameters(sphere):
    """The study's path (its chunks composed, the held matrix made once a chunk) against the derivative through the
    built graph's parameters by MADDENING's own scan (the matrix made at every step): one function, two programs, so
    equal to rounding, in both modes and at both scales."""
    import jax

    built = gentle(1.05).build()
    graph = built.graph
    base = graph.params

    def of(value):
        held = jax.tree.map(lambda x: x, base)
        held["nodes"]["drag:ball"]["shape_radius"] = value
        graph.reset_state()
        return pose_loss(graph.run_scan(STEPS, params=held))

    own = float(jax.grad(of)(jax.numpy.asarray(base["nodes"]["drag:ball"]["shape_radius"])))
    _, found = sphere
    for (mode, scale), held in found.items():
        assert abs(held[RADIUS] / own - 1) < 1e-10, (mode, scale, held[RADIUS], own)


def test_a_part_of_a_field_is_refused():
    """The sphere's initial position moves the pose its body starts from (seven numbers, the position three of them):
    a part of a field, which a derivative here is not read through."""
    from mime.build import BuildRefused

    with pytest.raises(BuildRefused) as raised:
        Study(gentle(1.05)).sweep(along({RADIUS: [mm(1.05)]}), duration=ms(STEPS)).gradient(
            pose_loss, wrt=["ball.initial.position"])
    assert [refusal.code for refusal in raised.value.refusals] == ["gradient.shape"]
