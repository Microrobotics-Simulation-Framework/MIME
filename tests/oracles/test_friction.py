"""Friction at the wall (``Contact(friction=Coulomb(mu=...))``, W2b), against closed forms, in both dtypes, each with a
control that fails on every run (nothing here is drawn; the drawn suites are ``tests/oracles/test_friction_suites.py``):

- **A block on an incline** (a cylinder lying along the fall line on its rims, which cannot roll down it): it sticks for
  ``tan(theta) <= mu`` and slides at ``g (sin(theta) - mu cos(theta))`` past it, either side of breakaway at 0.9995 and
  1.0005 of it. The control: ``mu`` doubled moves breakaway as predicted, and a slope between ``mu`` and ``2 mu`` slides
  at ``mu`` and sticks at ``2 mu``.
- **A solid sphere on an incline**: it rolls without slipping (``omega a = v``) at ``(5/7) g sin(theta)`` for
  ``tan(theta) <= (7/2) mu``, and slips past it, at ``g (sin(theta) - mu cos(theta))`` with its spin growing at ``(5/2)
  mu g cos(theta) / a``. On the node's half-space (an exact plane) and through ``built.run`` on a tilted tube's floor. The
  control: frictionless, the sphere slides at ``g sin(theta)`` and never turns.
- **The ledger**: friction's loss is ``mu N`` times the slip distance while it slides, and zero while it rolls, booked
  under its own name; its numerical column (the sweeps' remainder) at rounding; the step identity closes.
- **Gradients** in ``mu`` through the run, reverse mode against the closed form (``-g cos(theta) T`` sliding, zero
  rolling: the rolling case cannot discriminate, said so) and against central differences in float64.
- **A body off the wall feels no friction**: a sphere a micrometre above the shrunk wall, its point never held, steps as
  the frictionless hold does, bit for bit.
- **float32 at sub-microsecond steps** (the first step's probe): a sphere spun on the plane under a glycerol-like
  resistance at the contact, at dt / tau 0.001 (0.36 us), against a DOP853 reference: float32's error is float64's, and
  the contact carries the whole load. The control: the gap's rounding band taken out, float32 loses most of the load to
  the squeeze film and misses."""

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from friction_cases import (
    BALL,
    MASS,
    MOMENT,
    ROD,
    ROD_MASS,
    ROD_MOMENTS,
    axis,
    incline_oracle,
    net,
    on_the_floor,
    plane_node,
    slope_inputs,
)

import mime.nodes.body._rigid_inertial as rigid_module
from mime.build import BuildRefused
from mime.interactions import Contact
from mime.interactions.friction import Coulomb
from mime.nodes.body._friction import FRICTION_CONVERGED
from mime.units import s

G = 9.81
DT, STEPS = 1e-3, 100
#: The closed forms' resolution in each dtype: a hundred steps of a step solved exactly but for the dtype's rounding.
EXACT = {jnp.float64: 1e-12, jnp.float32: 2e-5}
#: Through the build, a margin of 10 um and the hull's points 0.1 mm apart (2,016 of them, 2.7 um short of the sphere,
#: within the margin): the dry oracles do not depend on either, and the default spacing at the default margin (the
#: roughness cutoff, 1.4 um) seeds the sphere from about 51,000.
COARSE = {"margin": 1e-5, "spacing": 1e-4}


@pytest.fixture(params=[jnp.float64, jnp.float32], ids=["float64", "float32"])
def dtype(request):
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", request.param == jnp.float64)
    yield request.param
    jax.config.update("jax_enable_x64", held)


def slid(node, inputs, steps: int = STEPS, params=None):
    """The node's state after ``steps`` steps under ``inputs``, from its initial state."""
    def scan(state):
        return jax.lax.scan(lambda held, _: (node.update(held, inputs, DT, params=params), None), state, None,
                            length=steps)[0]
    return jax.jit(scan)(node.initial_state())


# -- a block and a sphere on an incline, on the node's plane ---------------------------------------------------------

@pytest.mark.parametrize("ratio", [0.5, 0.9995, 1.0005, 2.0])
def test_a_block_on_an_incline_sticks_below_tan_theta_mu_and_slides_past_it(dtype, ratio):
    mu = 0.3
    theta = math.atan(mu * ratio)
    node = plane_node(ROD, mu=mu, mass=ROD_MASS, inertia=ROD_MOMENTS)
    out = slid(node, slope_inputs(ROD_MASS, theta, dtype))
    state, acceleration = incline_oracle(theta, mu, rolls=False)
    speed = float(out["twist"][0]) / (STEPS * DT)
    print(f"\n{state}: {speed:.10g} against {acceleration:.10g} m/s^2; spin {np.asarray(out['twist'][3:])}")
    assert abs(speed - acceleration) <= EXACT[dtype] * G
    assert float(jnp.max(jnp.abs(out["twist"][3:]))) <= EXACT[dtype] * G * STEPS * DT / 1e-3


def test_control_mu_doubled_moves_breakaway_as_predicted(dtype):
    """A slope of ``tan(theta) = 1.5 mu`` slides at ``mu`` and sticks at ``2 mu``: the threshold moves with ``mu``."""
    mu = 0.3
    theta = math.atan(1.5 * mu)
    found = {}
    for coefficient in (mu, 2 * mu):
        node = plane_node(ROD, mu=coefficient, mass=ROD_MASS, inertia=ROD_MOMENTS)
        found[coefficient] = float(slid(node, slope_inputs(ROD_MASS, theta, dtype))["twist"][0]) / (STEPS * DT)
    assert abs(found[mu] - incline_oracle(theta, mu, rolls=False)[1]) <= EXACT[dtype] * G
    assert abs(found[2 * mu]) <= EXACT[dtype] * G


@pytest.mark.parametrize("ratio", [0.5, 0.9995, 1.0005, 2.0])
def test_a_sphere_on_an_incline_rolls_without_slip_below_its_threshold_and_slips_past_it(dtype, ratio):
    mu = 0.2
    theta = math.atan(3.5 * mu * ratio)
    out = slid(plane_node(BALL, mu=mu), slope_inputs(MASS, theta, dtype))
    state, acceleration = incline_oracle(theta, mu, rolls=True)
    T = STEPS * DT
    v, spin = float(out["twist"][0]), float(out["twist"][4])
    expected_spin = v / 1e-3 if state == "roll" else 2.5 * mu * G * math.cos(theta) / 1e-3 * T
    print(f"\n{state}: {v / T:.10g} against {acceleration:.10g} m/s^2, spin {spin:.10g} against {expected_spin:.10g}")
    assert abs(v / T - acceleration) <= EXACT[dtype] * G
    assert abs(spin - expected_spin) <= EXACT[dtype] * max(abs(expected_spin), 1.0)
    assert float(out["contact_residual"]) <= FRICTION_CONVERGED


def test_control_frictionless_the_sphere_slides_and_never_turns(dtype):
    theta = math.atan(0.35)
    out = slid(plane_node(BALL, friction=False), slope_inputs(MASS, theta, dtype))
    assert abs(float(out["twist"][0]) / (STEPS * DT) - G * math.sin(theta)) <= EXACT[dtype] * G
    assert float(jnp.max(jnp.abs(out["twist"][3:]))) == 0.0


# -- through the build, on a tilted tube's floor ----------------------------------------------------------------------

@pytest.mark.parametrize("ratio", [0.5, 2.0])
def test_through_the_build_the_sphere_on_a_tilted_tubes_floor_rolls_or_slips_as_its_closed_form(dtype, ratio):
    mu = 0.2
    theta = math.atan(3.5 * mu * ratio)
    built = on_the_floor(theta=theta, mu=mu, **COARSE).build()
    twist = np.asarray(built.run(s(STEPS * DT), history=True).history["ball"]["twist"], dtype=np.float64)
    state, acceleration = incline_oracle(theta, mu, rolls=True)
    down = -twist[-1, :3] @ np.asarray(axis(theta)) / (STEPS * DT)
    turned = float(np.linalg.norm(twist[-1, 3:])) * 1e-3 / (-twist[-1, :3] @ np.asarray(axis(theta)))
    print(f"\n{state}: {down:.10g} against {acceleration * net():.10g} m/s^2; omega a / v {turned:.10g}")
    assert abs(down - acceleration * net()) <= 10 * EXACT[dtype] * G
    expected = 1.0 if state == "roll" else 2.5 * mu / (math.tan(theta) - mu)
    assert abs(turned - expected) <= 10 * EXACT[dtype] * max(expected, 1.0)


def test_friction_is_refused_on_a_body_with_no_hull():
    """``confinement.no-hull``, one code for every setting read at a hull's points: a body held as a sphere of its
    extent about its centre has no point on its surface for friction to act at; the message names friction= and both
    routes."""
    from mime import Body, Experiment, Medium, State, Structure
    from mime.dynamics import RigidInertial
    from mime.interactions import Gravity
    from mime.scene import Scene, Tube, X, Z
    from mime.units import kg, kg_m2, kg_per_m3, m, m_per_s, m_per_s2, mm, rad_per_s

    experiment = Experiment("no_hull_friction", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(G), dt=s(DT))
    tube = experiment.add_structure(Structure("tube", geometry=Tube(axis=+X, inner_radius=mm(3.0), wall=mm(1.0),
                                                                    length=mm(50.0))))
    gas = experiment.add_medium(Medium("gas", fills=tube.lumen))
    ball = experiment.add_body(Body("ball", dynamics=RigidInertial(), initial=State(
        position=mm((0.0, 0.0, -1.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    for path, value in (("ball.mass", kg(MASS)), ("ball.inertia", kg_m2((MOMENT,) * 3)), ("ball.extent", mm(1.0)),
                        ("ball.buoyant_volume", m(1e-3) * m(1e-3) * m(1e-3)), ("gas.density", kg_per_m3(1.0)),
                        ("ball.centre_of_buoyancy", m((0.0, 0.0, 0.0)))):
        experiment.declare(path, value, reason="a body with no parts")
    experiment.attach(Gravity(), body=ball, medium=gas)
    experiment.attach(Contact.Projection(margin=mm(0.1), friction=Coulomb(mu=0.2)), body=ball,
                      surface=tube.inner_surface)
    with pytest.raises(BuildRefused) as refused:
        experiment.plan()
    assert refused.value.codes() == ["confinement.no-hull"]
    message = str(refused.value)
    assert "friction=" in message and "parts=" in message and "remove friction=" in message


def test_the_margin_defaults_to_the_roughness_cutoff_and_reads_the_declared_roughness():
    from mime.nodes.hydro.near_contact import roughness_cutoff

    assert Contact.engagement(None, None) == pytest.approx(math.sqrt(2.0) * 1e-6, rel=1e-15)
    built = on_the_floor(roughness=3e-6, spacing=1e-4).build()
    node = built.graph.get_node("ball")
    assert float(node.constants()["margin"]) == pytest.approx(float(roughness_cutoff(3e-6, 3e-6)), rel=1e-15)


# -- the ledger --------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("ratio", [0.5, 2.0])
def test_the_ledger_books_friction_as_mu_n_times_the_slip_distance_and_rolling_as_zero(ratio):
    from mime.observations import EnergyLedger

    mu = 0.2
    theta = math.atan(3.5 * mu * ratio)
    experiment = on_the_floor(theta=theta, mu=mu, **COARSE)
    experiment.observe(EnergyLedger())
    result = experiment.build().run(s(STEPS * DT), history=True)
    account = result.energy
    assert account is not None
    rows = {row.law: row for row in account.laws}
    law = "contact:ball.friction"
    assert law in rows, sorted(rows)
    history = result.history["ball"]
    twist = np.asarray(history["twist"], dtype=np.float64)
    load = MASS * net() * G * math.cos(theta)
    if ratio < 1.0:
        expected = 0.0
    else:
        # the slip grows linearly from zero: v - omega a at the point, each step's slip at its end
        slips = np.abs(-twist[:, :3] @ np.asarray(axis(theta)) - np.linalg.norm(twist[:, 3:], axis=1) * 1e-3)
        expected = mu * load * DT * float(np.sum(slips))
    print(f"\nfriction's loss {rows[law].lost:.12g} J against {expected:.12g} J; host term {rows[law].host_term}")
    assert abs(rows[law].lost - expected) <= 1e-9 * max(expected, MASS * G * 1e-3)
    assert rows[law].gift_steps == 0
    assert account.findings == (), account.findings


def flipped_solve(monkeypatch):
    """Friction's sign flipped in the solve (each pair's force negated, the review's planted flip), for a test's
    build: the step reads ``_friction.disk_update`` when it is traced."""
    import mime.nodes.body._friction as friction_module

    original = friction_module.disk_update
    monkeypatch.setattr(friction_module, "disk_update", lambda block, load, bound: -original(block, load, bound))


def test_control_friction_that_pushes_is_booked_with_its_sign_and_counted_a_gift(monkeypatch):
    """The control of the ledger's friction row: friction's sign flipped in the solve (the sphere sliding down the
    tilted floor, ratio 2) does positive work on every sliding step; it is booked so, a gain on friction's row, each
    such step counted a gift (``gift_steps``, named in the law's line), as any dissipative law's, where a book of
    ``-|l||u|`` read a loss whatever the solve did."""
    from mime.observations import EnergyLedger

    flipped_solve(monkeypatch)
    mu = 0.2
    experiment = on_the_floor(theta=math.atan(3.5 * mu * 2.0), mu=mu, **COARSE)
    experiment.observe(EnergyLedger())
    account = experiment.build().run(s(STEPS * DT)).energy
    assert account is not None
    row = {row.law: row for row in account.laws}["contact:ball.friction"]
    print(f"\n{row.line()}")
    assert row.lost < 0.0 and row.gift_steps >= STEPS // 2, row
    assert f"positive in {row.gift_steps} steps" in row.line(), row.line()


# -- gradients in mu ---------------------------------------------------------------------------------------------------

#: The derivative's resolution against the closed form, and the central difference's relative step, by dtype.
GRADIENT = {jnp.float64: (1e-10, 1e-5), jnp.float32: (1e-5, None)}


@pytest.mark.parametrize("ratio", [0.5, 2.0])
def test_the_derivative_in_mu_is_the_closed_forms(dtype, ratio):
    """Sliding, ``dv/dmu = -g cos(theta) T`` exactly (the normal force does not move with ``mu``); rolling without
    slip, zero: there the trajectory does not depend on ``mu`` above its threshold, so this half cannot discriminate a
    wrong derivative from a right one (said so; the sliding half does)."""
    mu, steps = 0.2, 50
    theta = math.atan(3.5 * mu * ratio)
    node = plane_node(BALL, mu=mu)
    leaves = node.params_pytree()
    inputs = slope_inputs(MASS, theta, dtype)

    def speed(coefficient):
        return slid(node, inputs, steps, params={**leaves, "friction_mu": coefficient})["twist"][0]

    held = jnp.asarray(mu, dtype=dtype)
    derivative = float(jax.grad(speed)(held))
    closed = 0.0 if ratio < 1.0 else -G * math.cos(theta) * steps * DT
    tolerance, step = GRADIENT[dtype]
    print(f"\nAD {derivative:.12g} against {closed:.12g}")
    assert abs(derivative - closed) <= tolerance * max(abs(closed), 1.0)
    if step is not None:
        h = step * mu
        central = (float(speed(held + h)) - float(speed(held - h))) / (2 * h)
        assert abs(derivative - central) <= 1e-7 * max(abs(closed), 1.0)


def test_control_mu_stopped_has_no_derivative():
    """The control: ``mu`` held as a constant (its derivative stopped) reads zero where the closed form is -0.285, so
    the sliding gate tells a missing derivative from the right one."""
    mu, steps = 0.2, 50
    theta = math.atan(3.5 * mu * 2.0)
    node = plane_node(BALL, mu=mu)
    leaves = node.params_pytree()
    inputs = slope_inputs(MASS, theta)

    def speed(coefficient):
        return slid(node, inputs, steps, params={**leaves, "friction_mu": jax.lax.stop_gradient(coefficient)})[
            "twist"][0]

    assert float(jax.grad(speed)(jnp.asarray(mu))) == 0.0 != -G * math.cos(theta) * steps * DT


#: The slip speed's derivative against central differences: the steps, and the difference's relative span and the
#: tolerance on the derivative's scale, by dtype.
SLIP_STEPS = 20
SLIP_DIFFERENCE = {jnp.float64: (1e-6, 1e-6), jnp.float32: (1e-3, 2e-2)}


#: Each body's shape, mass, moments and the factor on ``mu`` of its stick threshold's slope (``tan(theta)``).
SLIP_BODIES = {"sphere": (BALL, MASS, (MOMENT,) * 3, 3.5), "block": (ROD, ROD_MASS, ROD_MOMENTS, 1.0)}
SLIP_MU = 0.2


@functools.cache
def compiled_slip(body: str, dtype):
    """The slip after :data:`SLIP_STEPS` steps in ``(theta, mu)`` on ``body``'s node with its reverse-mode gradient,
    one program per body and dtype: the four ratios and the central differences' points are its arguments."""
    shape, mass, inertia, _ = SLIP_BODIES[body]
    slip = slip_after(plane_node(shape, mu=SLIP_MU, mass=mass, inertia=inertia), mass, dtype)
    return jax.jit(jax.value_and_grad(slip, argnums=(0, 1)))


@pytest.mark.parametrize("body", ["sphere", "block"])
@pytest.mark.parametrize("ratio", [0.5, 0.99, 1.01, 2.0])
def test_the_slip_speeds_derivative_is_its_central_differences_either_side_of_the_threshold(dtype, body, ratio):
    """``contact_slip_speed`` after 20 steps on the slope, its derivative in ``mu`` and in the slope's angle by reverse
    mode against central differences taken on the point's own side of the stick threshold (a span across it reads the
    kink): zero sticking, the sliding slip's slope past it (the review's -0.562 in ``mu`` on the sphere at 1.001). The
    control below takes the slip through a stopped value, as the field was, and reads zero against it."""
    mu = SLIP_MU
    theta = math.atan(SLIP_BODIES[body][3] * mu * ratio)
    with_gradient = compiled_slip(body, dtype)
    span, tolerance = SLIP_DIFFERENCE[dtype]
    at = (jnp.asarray(theta, dtype), jnp.asarray(mu, dtype))
    measured = [float(d) for d in with_gradient(*at)[1]]
    side = min(span, 0.25 * abs(ratio - 1.0))
    eps = float(jnp.finfo(dtype).eps)
    scale = G * SLIP_STEPS * DT  # the slip's slope's own size, m/s per unit of mu or of angle
    for index, (name, value) in enumerate((("theta", theta), ("mu", mu))):
        h = value * side
        up, down = list(at), list(at)
        up[index], down[index] = at[index] + h, at[index] - h
        central = (float(with_gradient(*up)[0]) - float(with_gradient(*down)[0])) / (2 * h)
        rounding = 64 * eps * scale * SLIP_STEPS * DT / h
        print(f"\n{body} at {ratio} of its threshold: d slip / d {name} AD {measured[index]:.9g}, CD {central:.9g}")
        assert abs(measured[index] - central) <= tolerance * scale + rounding, (name, measured[index], central)
    if ratio > 1.0:
        assert abs(measured[1]) > 0.1 * scale, measured  # sliding, the slip moves with mu: the gate discriminates


def slip_after(node, mass: float, dtype, stopped: bool = False):
    """``(theta, mu) ->`` the largest held point's slip speed after :data:`SLIP_STEPS` steps on the slope ``theta``
    (``stopped``: taken through a stopped value, the control)."""
    leaves = node.params_pytree()

    def slip(theta, mu):
        inputs = {"applied_wrench": jnp.stack([mass * G * jnp.sin(theta), jnp.zeros((), dtype),
                                               -mass * G * jnp.cos(theta), *(jnp.zeros((), dtype),) * 3]),
                  "resistance": jnp.zeros((6, 6), dtype), "background_load": jnp.zeros(6, dtype)}
        ended = slid(node, inputs, SLIP_STEPS, params={**leaves, "friction_mu": mu})
        speed = jnp.max(ended["contact_slip_speed"])
        return jax.lax.stop_gradient(speed) if stopped else speed

    return slip


def test_control_a_stopped_slip_speed_reads_no_derivative_against_its_central_difference(dtype):
    """The control: the slip speed taken through a stopped value (the field before the fix) reads zero where the
    central difference reads the sliding slope, so the gate above tells a silent zero from the derivative."""
    mu, ratio = 0.2, 2.0
    theta = math.atan(3.5 * mu * ratio)
    node = plane_node(BALL, mu=mu)
    stopped = slip_after(node, MASS, dtype, stopped=True)
    held = (jnp.asarray(theta, dtype), jnp.asarray(mu, dtype))
    slip = slip_after(node, MASS, dtype)
    h = SLIP_DIFFERENCE[dtype][0] * mu
    central = (float(slip(held[0], held[1] + h)) - float(slip(held[0], held[1] - h))) / (2 * h)
    assert float(jax.grad(stopped, argnums=1)(*held)) == 0.0
    assert abs(central) > 0.1 * G * SLIP_STEPS * DT, central


# -- a body off the wall ------------------------------------------------------------------------------------------------

def test_a_body_off_the_wall_feels_no_friction(dtype):
    """A sphere sliding a micrometre above the shrunk plane, held up by a load that balances its weight: no point is
    held, so the frictional hold's step is the frictionless one's, bit for bit (friction acts only on the contact's
    own normal force). The control: the same sphere on the plane, sliding, steps otherwise with friction."""
    lift = {"applied_wrench": jnp.zeros(6, dtype=dtype), "resistance": jnp.zeros((6, 6), dtype=dtype),
            "background_load": jnp.zeros(6, dtype=dtype)}
    rubbing, smooth = plane_node(BALL), plane_node(BALL, friction=False)

    def run(node, height):
        state = {**node.initial_state(), "pose": jnp.asarray([0.0, 0.0, 1e-3 + height, 1.0, 0.0, 0.0, 0.0], dtype),
                 "twist": jnp.asarray([2e-3, 1e-3, 0.0, 0.0, 0.0, 0.0], dtype)}
        return jax.lax.scan(lambda held, _: (node.update(held, lift, DT), None), state, None, length=20)[0]

    for field in ("pose", "twist"):
        assert np.array_equal(np.asarray(run(rubbing, 1e-6)[field]), np.asarray(run(smooth, 1e-6)[field]))
    pressed = slope_inputs(MASS, 0.0, dtype)

    def pressed_run(node):
        state = {**node.initial_state(), "twist": jnp.asarray([2e-3, 1e-3, 0.0, 0.0, 0.0, 0.0], dtype)}
        return jax.lax.scan(lambda held, _: (node.update(held, pressed, DT), None), state, None, length=20)[0]

    assert not np.array_equal(np.asarray(pressed_run(rubbing)["twist"]), np.asarray(pressed_run(smooth)["twist"]))


# -- float32 at sub-microsecond steps -------------------------------------------------------------------------------------

def spun_run(dtype, *, band: bool = True):
    """The spun sphere (``friction_cases.SPUN``) on the node's plane at dt/tau 0.001, mu 0.5: the worst |v - v_ref| over
    a omega0 against the DOP853 reference, and the contact's mean force over the load after its first contact."""
    from friction_cases import SPUN, spun_reference, spun_resistance

    mu, omega0 = 0.5, 100.0
    a, mass, moment, load = SPUN["radius"], SPUN["mass"], SPUN["moment"], SPUN["load"]
    tau = min(mass / SPUN["R_xx"], moment / SPUN["R_ryry"])
    stuck, reference = spun_reference(omega0, mu, 60 * tau)
    t_end = 2 * max(stuck or 0.0, 5 * tau)
    dt = 1e-3 * tau
    steps = int(round(t_end / dt))
    ulps = rigid_module.CONTACT_ULPS
    rigid_module.CONTACT_ULPS = ulps if band else 0
    try:
        kind = (BALL[0].__class__(radius=BALL[0].radius * 0.7), {"radius": a})
        node = plane_node(kind, mu=mu, dt=dt, mass=mass, inertia=(moment,) * 3, margin=SPUN["gap"])
        inputs = {"applied_wrench": jnp.asarray([0.0, 0.0, -load, 0.0, 0.0, 0.0], dtype),
                  "resistance": jnp.asarray(spun_resistance(), dtype), "background_load": jnp.zeros(6, dtype)}
        state = {**node.initial_state(), "twist": jnp.asarray([0.0, 0.0, 0.0, 0.0, omega0, 0.0], dtype)}

        def scan(held):
            def one(state, _):
                ended = node.update(state, inputs, dt)
                return ended, (ended["twist"][0], jnp.sum(ended["contact_forces"]))
            return jax.lax.scan(one, held, None, length=steps)[1]

        speeds, forces = (np.asarray(x, dtype=np.float64) for x in jax.jit(scan)(state))
    finally:
        rigid_module.CONTACT_ULPS = ulps
    times = (np.arange(steps) + 1) * dt
    expected = np.asarray([reference(t)[0] for t in times])
    first = int(np.argmax(forces > 0.0))
    return float(np.max(np.abs(speeds - expected)) / (a * omega0)), float(np.mean(forces[first:]) / load)


def test_float32_at_sub_microsecond_steps_matches_float64(dtype):
    error, carried = spun_run(dtype)
    print(f"\nworst |v - v_ref| / (a omega0) {error:.3g}; the contact carried {carried:.6f} of the load")
    assert 1e-5 <= error <= 1e-4
    assert abs(carried - 1.0) <= 1e-3


def test_control_without_the_gaps_band_float32_loses_the_load_to_the_squeeze():
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        error, carried = spun_run(jnp.float32, band=False)
    finally:
        jax.config.update("jax_enable_x64", held)
    print(f"\nno band: worst {error:.3g}, the contact carried {carried:.4f} of the load")
    assert error > 1e-3 and carried < 0.5
