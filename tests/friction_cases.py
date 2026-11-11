"""Bodies held with friction (``Contact(friction=Coulomb(mu=...))``): a sphere and a block-like cylinder on an incline,
as a node on a half-space (``plane_node``, a plane exact at the node, the scene's ``Plane`` having no contact surface
yet) and as an experiment on a tube's floor (``on_the_floor``, gravity turned along the tube's axis by the scene's
``up``); their closed forms (``incline_oracle``)."""

import math

import jax.numpy as jnp
import numpy as np

from mime import Body, Experiment, Medium, State, Structure
from mime.dynamics import RigidInertial
from mime.interactions import Contact, Gravity
from mime.interactions.friction import Coulomb
from mime.nodes.body import HULL, RigidInertialMidpointNode
from mime.scene import Cylinder, Scene, Sphere, Tube, Z
from mime.scene._sampling import quadrature_counts
from mime.things import Part
from mime.units import kg_per_m3, m, m_per_s, m_per_s2, mm, rad_per_s, s

G = 9.81
#: A millimetre sphere of 2000 kg/m^3 (the spike's case 2), and its mass and moment.
RADIUS, DENSITY = 1e-3, 2000.0
MASS = DENSITY * 4.0 / 3.0 * math.pi * RADIUS ** 3
MOMENT = 0.4 * MASS * RADIUS ** 2
BALL = (Sphere(radius=mm(1.0)), {"radius": RADIUS})
#: A cylinder lying along the slope's fall line on its rims: it cannot roll down the slope, so it is the block.
ROD = (Cylinder(radius=mm(0.5), length=mm(3.0), axis=(1.0, 0.0, 0.0)),
       {"radius": 5e-4, "length": 3e-3, "axis": (1.0, 0.0, 0.0)})
ROD_MASS = DENSITY * math.pi * 5e-4 ** 2 * 3e-3
ROD_MOMENTS = (0.5 * ROD_MASS * 5e-4 ** 2, ROD_MASS * (3 * 5e-4 ** 2 + 3e-3 ** 2) / 12.0,
               ROD_MASS * (3 * 5e-4 ** 2 + 3e-3 ** 2) / 12.0)
TUBE_RADIUS = 3.175e-3


def rolling_oracle(theta: float, mu: float, mu_r: float, a: float = RADIUS) -> tuple[str, float, float]:
    """A solid sphere on a slope of ``theta`` with friction ``mu`` and rolling resistance ``mu_r`` (a length): its state,
    its acceleration down the slope and its spin's: held still for ``tan(theta) <= mu_r / a`` (and ``<= mu``); rolling
    without slip for ``tan(theta) <= (7/2) mu - (5/2) mu_r / a`` at ``(5/7) g (sin - (mu_r/a) cos)``; else slipping at
    ``g (sin - mu cos)``, its spin growing at ``(mu a - mu_r) g cos / (2/5 a^2)``."""
    s, c = math.sin(theta), math.cos(theta)
    if math.tan(theta) <= mu_r / a and math.tan(theta) <= mu:
        return "still", 0.0, 0.0
    if math.tan(theta) <= 3.5 * mu - 2.5 * mu_r / a:
        acceleration = 5.0 / 7.0 * G * (s - mu_r / a * c)
        return "roll", acceleration, acceleration / a
    return "slip", G * (s - mu * c), max((mu * a - mu_r) * G * c / (0.4 * a * a), 0.0)


def incline_oracle(theta: float, mu: float, rolls: bool) -> tuple[str, float]:
    """The state and the acceleration down the slope of a solid sphere (``rolls``) or a body that cannot roll: a sphere
    rolls without slipping for ``tan(theta) <= (7/2) mu`` at ``(5/7) g sin(theta)``, else slips; a block sticks for
    ``tan(theta) <= mu``, else slides; sliding at ``g (sin(theta) - mu cos(theta))``."""
    if rolls and math.tan(theta) <= 3.5 * mu:
        return "roll", 5.0 / 7.0 * G * math.sin(theta)
    if not rolls and math.tan(theta) <= mu:
        return "stick", 0.0
    return "slide", G * (math.sin(theta) - mu * math.cos(theta))


def plane_node(body=BALL, *, mu: float = 0.2, dt: float = 1e-3, mass: float = MASS,
               inertia=(MOMENT,) * 3, friction: bool = True, spacing: float = 2e-4, margin: float = 0.0,
               rolling: float | None = None, cls=RigidInertialMidpointNode):
    """``body``'s node held by its hull on the half-space ``z >= 0`` (the contact 0.5.0's loops), with Coulomb friction
    ``mu`` (or none) and, where ``rolling`` is given, rolling resistance of that length, its lowest point on the
    shrunk plane."""
    kind, settings = body
    patches = tuple(kind.patches)
    lowest = settings["radius"]
    held = {"friction": "coulomb" if rolling is None else "coulomb_rolling", "friction_mu": mu,
            "friction_law": "contact:body.friction",
            **({} if rolling is None else {"friction_rolling": rolling})} if friction else {}
    return cls("body", dt, mass=mass, inertia=tuple(inertia), initial_pose=(0.0, 0.0, lowest + margin, 1, 0, 0, 0),
               initial_twist=(0.0,) * 6, surface_shape="half_space", surface_normal=(0.0, 0.0, 1.0),
               surface_offset=0.0, margin=margin, velocity_rule="remove_outward", hull_kind=kind.name,
               hull_kind_version=kind.version, hull_patches=patches,
               hull_counts=quadrature_counts(kind, settings, spacing, patches), contact_loops=True, **held,
               **{f"{HULL}{name}": value for name, value in settings.items()})


def slope_inputs(mass: float, theta: float, dtype=None) -> dict:
    """The weight on the plane ``z = 0`` turned into a slope of ``theta`` down ``+x``: ``m g (sin, 0, -cos)``."""
    held = {} if dtype is None else {"dtype": dtype}
    return {"applied_wrench": jnp.asarray([mass * G * math.sin(theta), 0.0, -mass * G * math.cos(theta), 0.0, 0.0,
                                           0.0], **held),
            "resistance": jnp.zeros((6, 6), **held), "background_load": jnp.zeros(6, **held)}


def on_the_floor(name: str = "friction_sphere", *, theta: float = 0.3, mu: float | None = 0.2, body=BALL,
                 density: float = DENSITY, fluid_density: float = 1.0, dt: float = 1e-3, margin: float | None = None,
                 roughness: float | None = None, rolling: float | None = None, along: bool = False) -> Experiment:
    """``body`` on the floor of a tube whose axis is tilted ``theta`` up from x about -y (``axis(theta)``), so that
    gravity pulls down the tube's lowest line by ``g sin(theta)`` and into the floor by ``g cos(theta)``, held by the contact with ``Coulomb(mu)`` (none where
    ``mu`` is ``None``; with rolling resistance ``rolling`` where given), its margin the contact's default (the roughness
    cutoff) unless written; in a gas of ``fluid_density`` (its buoyancy in the gravity law). ``along`` turns the body's
    x along the tube's axis (a cylinder lying on the floor down the slope: the block)."""
    kind, settings = body
    experiment = Experiment(name, scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(G), dt=s(dt))
    tube = experiment.add_structure(Structure("tube", geometry=Tube(axis=axis(theta), inner_radius=m(TUBE_RADIUS),
                                                                    wall=mm(1.0), length=mm(100.0))))
    gas = experiment.add_medium(Medium("gas", fills=tube.lumen))
    held = Contact.engagement(roughness, roughness) if margin is None else margin
    lowest = settings["radius"]
    ball = experiment.add_body(Body("ball", dynamics=RigidInertial(), initial=State(
        position=m(tuple(float(c) * (TUBE_RADIUS - held - lowest) for c in floor(theta))),
        orientation=(math.cos(theta / 2), 0.0, -math.sin(theta / 2), 0.0) if along else (1.0, 0.0, 0.0, 0.0),
        velocity=m_per_s((0.0, 0.0, 0.0)), angular_velocity=rad_per_s((0.0, 0.0, 0.0))),
        parts={"shell": Part(kind)}))
    experiment.declare("ball.parts.shell.density", kg_per_m3(density), reason="the spike's sphere")
    experiment.declare("gas.density", kg_per_m3(fluid_density), reason="a gas, its buoyancy small")
    if roughness is not None:
        experiment.declare("ball.roughness", m(roughness), reason="a declared roughness")
        experiment.declare("tube.roughness", m(roughness), reason="a declared roughness")
    experiment.attach(Gravity(), body=ball, medium=gas)
    contact = Contact.Projection(**({} if margin is None else {"margin": m(margin)}),
                                 **({} if mu is None else {"friction": Coulomb(
                                     mu=mu, **({} if rolling is None else {"rolling": m(rolling)}))}))
    experiment.attach(contact, body=ball, surface=tube.inner_surface)
    return experiment


def axis(theta: float) -> tuple[float, float, float]:
    """The tube's axis tilted ``theta`` up from x: down its lowest line is ``-axis``."""
    return (math.cos(theta), 0.0, math.sin(theta))


def floor(theta: float) -> tuple[float, float, float]:
    """The direction from the tilted tube's axis to its lowest line, across the axis."""
    return (math.sin(theta), 0.0, -math.cos(theta))


def net(density: float = DENSITY, fluid_density: float = 1.0) -> float:
    """The share of the weight left by buoyancy, which scales both the slope's pull and the load on the floor."""
    return 1.0 - fluid_density / density


__all__ = ["BALL", "MASS", "MOMENT", "ROD", "axis", "floor", "incline_oracle", "rolling_oracle", "net", "on_the_floor", "plane_node", "slope_inputs"]


#: The done case's sphere in glycerol-water (the friction spike's case 3 with fluid): radius, mass, moment, net load,
#: and its resistance at the contact gap (0.002 radii, the roughness cutoff), the spike's composed table there.
SPUN = {"radius": 0.7e-3, "mass": 4.58e-6, "moment": 8.98e-13, "load": 27.8e-6, "gap": 1.4e-6,
        "R_xx": 5.65e-3, "R_xry": -5.32e-7, "R_ryry": 2.48e-9, "R_zz": 0.663}


def spun_resistance() -> np.ndarray:
    """The spun sphere's 6 x 6 resistance at its contact gap, about its centre: the spike's entries, with the plane's
    symmetry for the rest (y as x, the coupling of y with rotation about x of the other sign, spin as rolling)."""
    R = np.zeros((6, 6))
    R[0, 0] = R[1, 1] = SPUN["R_xx"]
    R[2, 2] = SPUN["R_zz"]
    R[3, 3] = R[4, 4] = R[5, 5] = SPUN["R_ryry"]
    R[0, 4] = R[4, 0] = SPUN["R_xry"]
    R[1, 3] = R[3, 1] = -SPUN["R_xry"]
    return R


def spun_reference(omega0: float, mu: float, t_end: float):
    """The planar sphere-on-a-wall model the node's step solves, as an ODE, by DOP853 at rtol 1e-12 with event
    detection at sticking (the friction spike's reference): sliding ``f = -mu N sign(u)``, ``u = v - a omega``, the
    resistance's ``x`` and ``ry`` block; once ``u`` reaches zero with the needed traction within ``mu N`` it rolls,
    ``(m a^2 + I) omega' = -(a^2 R_xx + 2 a R_xry + R_ryry) omega``. Returns (the stick time, ``t -> (v, omega)``)."""
    from scipy.integrate import solve_ivp

    a, m, I, N = SPUN["radius"], SPUN["mass"], SPUN["moment"], SPUN["load"]
    Rxx, Rxw, Rww = SPUN["R_xx"], SPUN["R_xry"], SPUN["R_ryry"]

    def sliding(sign: float):
        def rhs(t, y):
            v, w = y
            f = -mu * N * sign
            return [(-Rxx * v - Rxw * w + f) / m, (-Rxw * v - Rww * w - a * f) / I]
        return rhs

    def needed(v: float, w: float) -> float:
        w_dot = -(a * a * Rxx + 2 * a * Rxw + Rww) * w / (m * a * a + I)
        return m * a * w_dot + Rxx * v + Rxw * w

    y0 = np.array([0.0, omega0])
    sign = float(np.sign(-a * omega0))
    pieces = []
    t0, stuck = 0.0, None
    while t0 < t_end:
        def event(t, y):
            return y[0] - a * y[1]
        event.terminal, event.direction = True, 0
        sol = solve_ivp(sliding(sign), (t0, t_end), y0, method="DOP853", rtol=1e-12, atol=1e-16, events=event,
                        dense_output=True)
        pieces.append((t0, sol.t[-1], sol.sol))
        if sol.status != 1:
            break
        t0, y0 = sol.t_events[0][0], sol.y_events[0][0]
        if abs(needed(*y0)) <= mu * N:
            stuck = t0
            k = (a * a * Rxx + 2 * a * Rxw + Rww) / (m * a * a + I)
            w_s = y0[1]
            pieces.append((t0, t_end, lambda t, ts=t0, w=w_s: np.array([a * w * np.exp(-k * (t - ts)),
                                                                       w * np.exp(-k * (t - ts))])))
            break
        sign = -sign

    def at(t: float) -> np.ndarray:
        for low, high, f in pieces:
            if low <= t <= high + 1e-15:
                return np.asarray(f(t))
        return np.asarray(pieces[-1][2](t))
    return stuck, at
