"""The contact's conformance suites: what every friction law, normal law and friction row kind passes, judged through the
body node's own frictional hold (``mime.interactions.friction``'s seams; decided Fr6), on draws at physical scales in
the run's dtype. A law of a lab's passes them with its own oracle beside them (Coulomb's: ``tests/oracles/
test_friction.py``, breakaway at ``tan(theta) = mu``).

- :func:`friction_law_suite`: **passivity**, each held point's friction power at the solved twist never above its floor
  (``_friction.power_floor``: the spike's guard, which caught a planted sign flip in every draw where the kinetic
  energy missed most); **the sticking limits**, a sphere's point whose sticking traction is within the law's limit
  sticks (its slip at rounding) and one past it slides on the limit, anti-parallel to its slip (the law's
  ``limit(load, speed)`` read against the solve); **its derivative** in the law's coefficient, reverse mode against
  central differences, on sliding draws.
- :func:`normal_law_suite`: the normal force never below the law's least (``NormalLaw.least``), no point past the
  shrunk wall after the step beyond rounding, and the hold's complementarity met (its residual under ``CONVERGED``).
- :func:`row_kind_suite`: a built row kind's rows measure what it declares (sliding: the slip of the body's own point,
  ``t . (v + omega x r)``, on a frame across the normal; rolling: the rotation about the tangents, ``t . omega``); a
  reserved kind is refused by the contact with its reason.

Draws (a sphere of 1 mm and a cylinder 1 mm by 3 mm lying on the plane ``z >= 0``, 2000 kg/m^3, a step of 1 ms): a
load of 0.3 to 3 weights at up to 1.4 rad from the normal in any azimuth, linear speeds along the plane of 1e-6 to 5e-2
m/s or none, angular speeds of 1e-3 to 50 rad/s or none, any orientation, and a coefficient log-uniform on [0.01, 1.2]
(the register's range). Nothing below a body's scale is drawn: the numerical extremes (a bound at rounding, a normal
force of zero) are tested on purpose in ``tests/oracles/test_friction_suites.py``."""

from __future__ import annotations

import math
import operator
from collections.abc import Callable
from typing import Any, cast

import numpy as np
from maddening.testing.verification import VerificationResult

from mime.core import jnp

#: The bodies the suites hold on the plane, by name: kind, settings, mass and principal moments.
BODIES = ("sphere", "cylinder")
_DT = 1e-3
_G = 9.81


def _body(name: str) -> tuple[Any, dict[str, Any], float, tuple[float, ...], float]:
    from mime.scene import Cylinder, Sphere
    from mime.units import mm

    if name == "sphere":
        mass = 2000.0 * 4.0 / 3.0 * math.pi * 1e-3 ** 3
        return Sphere(radius=mm(1.0)), {"radius": 1e-3}, mass, (0.4 * mass * 1e-6,) * 3, 1e-3
    radius, length = 5e-4, 3e-3
    mass = 2000.0 * math.pi * radius ** 2 * length
    side = mass * (3 * radius ** 2 + length ** 2) / 12.0
    return (Cylinder(radius=mm(0.5), length=mm(3.0), axis=(1.0, 0.0, 0.0)),
            {"radius": radius, "length": length, "axis": (1.0, 0.0, 0.0)}, mass, (0.5 * mass * radius ** 2, side, side),
            radius)


def held_node(name: str, *, mu: float = 0.3, friction: bool = True) -> Any:
    """A body of :data:`BODIES` held by its hull on the half-space ``z >= 0`` with Coulomb friction (``mu``, a leaf),
    its lowest point on the plane (the contact 0.5.0's loops)."""
    from mime.nodes.body import HULL, RigidInertialMidpointNode
    from mime.scene._sampling import quadrature_counts

    kind, settings, mass, moments, lowest = _body(name)
    patches = tuple(kind.patches)
    held = {"friction": "coulomb", "friction_mu": mu, "friction_law": "contact:body.friction"} if friction else {}
    return RigidInertialMidpointNode(
        "body", _DT, mass=mass, inertia=moments, initial_pose=(0.0, 0.0, lowest, 1.0, 0.0, 0.0, 0.0),
        initial_twist=(0.0,) * 6, surface_shape="half_space", surface_normal=(0.0, 0.0, 1.0), surface_offset=0.0,
        margin=0.0, velocity_rule="remove_outward", hull_kind=kind.name, hull_kind_version=kind.version,
        hull_patches=patches, hull_counts=quadrature_counts(kind, settings, 2e-4, patches), contact_loops=True,
        **held, **{f"{HULL}{key}": value for key, value in settings.items()})


def _draws() -> Any:
    from hypothesis import strategies as st

    def log(low: float, high: float) -> Any:
        return st.floats(math.log(low), math.log(high)).map(math.exp)

    def signed_or_zero(low: float, high: float) -> Any:
        return st.one_of(st.just(0.0), log(low, high), log(low, high).map(operator.neg))

    turn = st.tuples(st.floats(0.0, math.pi), st.floats(0.0, 2 * math.pi), st.floats(0.0, 2 * math.pi)).map(_turn)
    return st.tuples(log(0.3, 3.0), st.floats(0.0, 1.4), st.floats(0.0, 2 * math.pi),
                     st.tuples(signed_or_zero(1e-6, 5e-2), signed_or_zero(1e-6, 5e-2)),
                     st.tuples(*(signed_or_zero(1e-3, 50.0) for _ in range(3))), turn, log(0.01, 1.2))


def _unit(polar: float, azimuth: float) -> tuple[float, float, float]:
    """The unit direction at ``polar`` from +z and ``azimuth`` about it: every direction drawn, none filtered."""
    return (math.sin(polar) * math.cos(azimuth), math.sin(polar) * math.sin(azimuth), math.cos(polar))


def _turn(angles: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """The unit quaternion turning by ``angles[2]`` about the axis at ``angles[0], angles[1]``."""
    axis = _unit(angles[0], angles[1])
    half = 0.5 * angles[2]
    return (math.cos(half), math.sin(half) * axis[0], math.sin(half) * axis[1], math.sin(half) * axis[2])


def _placed(name: str, turn: tuple[float, ...]) -> Any:
    """The body at the orientation ``turn`` (normalised), lowered until its lowest point is on the plane: a sphere's
    centre a radius up; a cylinder's by its support along -z."""
    q = np.asarray(turn, dtype=np.float64)
    q = q / np.linalg.norm(q)
    _, settings, _, _, lowest = _body(name)
    if name == "cylinder":
        w, x, y, z = q
        rot = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        axis = rot @ np.asarray(settings["axis"])
        across = math.sqrt(max(0.0, 1.0 - axis[2] ** 2))
        lowest = settings["length"] / 2 * abs(axis[2]) + settings["radius"] * across
    return np.concatenate([[0.0, 0.0, lowest], q])


def _stepper(node: Any, dtype: Any) -> Callable[..., dict[str, Any]]:
    """One step of ``node`` from a drawn pose, twist, load and coefficient, compiled once."""
    import jax

    leaves = node.params_pytree()
    start = node.initial_state()

    def step(pose: Any, twist: Any, load: Any, mu: Any) -> dict[str, Any]:
        state = {**start, "pose": pose, "twist": twist}
        given = {"applied_wrench": load, "resistance": jnp.zeros((6, 6), dtype=dtype),
                 "background_load": jnp.zeros(6, dtype=dtype)}
        return node.advance({**node._step_constants(None), **leaves, "friction_mu": mu}, state, given, _DT)  # noqa: SLF001

    return cast(Callable[..., dict[str, Any]], cast(Any, jax).jit(step))


def _gaps(node: Any, dtype: Any) -> Callable[[Any], Any]:
    """The gaps of ``node``'s held points to the shrunk plane at a pose, compiled once."""
    import jax

    constants = {**node._step_constants(None), **node.params_pytree()}  # noqa: SLF001

    def gaps(pose: Any) -> Any:
        return node.contacts(constants, node.hull_points(constants, dtype), pose).gaps

    return cast(Callable[[Any], Any], cast(Any, jax).jit(gaps))


def _drawn(name: str, drawn: tuple[Any, ...], dtype: Any) -> tuple[Any, Any, Any, Any, float]:
    _, _, mass, _, _ = _body(name)
    size, polar, azimuth, linear, angular, turn, mu = drawn
    weight = mass * _G * size
    load = weight * np.array([math.sin(polar) * math.cos(azimuth), math.sin(polar) * math.sin(azimuth),
                              -math.cos(polar), 0.0, 0.0, 0.0])
    twist = np.array([linear[0], linear[1], 0.0, *angular])
    return (jnp.asarray(_placed(name, turn), dtype=dtype), jnp.asarray(twist, dtype=dtype),
            jnp.asarray(load, dtype=dtype), jnp.asarray(mu, dtype=dtype), weight)


#: The sweeps the suites solve at: enough to converge every drawn problem (the spike's reference was 4,000; its
#: four-corner box, the slowest, reached 1e-6 of its change by 89 at the 99th percentile), so that a law's passivity is
#: judged at its solution. At the declared count a trip of the guard on an unconverged iterate raises unconverged
#: (the frictional hold's residual), which the sweep-count measurement reports.
SUITE_SWEEPS = 400


def _converged(names: tuple[str, ...], sweeps: int) -> dict[str, int | None]:
    """Each body's hull kind's declared sweep count set to ``sweeps``, returning what each was (to restore)."""
    import mime.nodes.body._rigid_inertial as rigid

    held: dict[str, int | None] = {}
    for name in names:
        kind = _body(name)[0].name
        held[kind] = rigid.FRICTION_SWEEPS.get(kind)
        rigid.FRICTION_SWEEPS[kind] = sweeps
    return held


def _restored(held: dict[str, int | None]) -> None:
    import mime.nodes.body._rigid_inertial as rigid

    for kind, count in held.items():
        if count is None:
            rigid.FRICTION_SWEEPS.pop(kind, None)
        else:
            rigid.FRICTION_SWEEPS[kind] = count


def friction_law_suite(law: Any = None, *, bodies: tuple[str, ...] = BODIES, max_examples: int = 200,
                       plant: Callable[[Any], Any] | None = None, sweeps: int = SUITE_SWEEPS) -> VerificationResult:
    """``law`` (a ``FrictionLaw``; ``Coulomb`` where none is given) through the body's frictional hold on draws in the
    run's dtype (the module's cases), solved at ``sweeps`` (converged): passivity on every body, the sticking limits
    and the derivative on the sphere. Every draw is judged, none stops the run: the detail counts the draws that
    failed, a control's rate. ``plant``, for a control, replaces ``_friction.disk_update`` while the suite runs (the
    sign flip: ``lambda x: -x`` on its force)."""
    import jax
    from hypothesis import HealthCheck, Phase, given, settings

    import mime.nodes.body._friction as friction_module
    from mime.interactions.friction import Coulomb
    from mime.nodes.body._friction import power_floor

    held = Coulomb(mu=0.3) if law is None else law
    dtype = jnp.zeros(()).dtype
    floor = power_floor(dtype)
    eps = float(jnp.finfo(dtype).eps)
    original = friction_module.disk_update
    if plant is not None:
        def planted(block: Any, load: Any, bound: Any) -> Any:
            return plant(original(block, load, bound))

        friction_module.disk_update = planted
    failures: list[str] = []
    judged: list[bool] = []
    rubbed: list[bool] = []
    declared = _converged(bodies, sweeps)
    try:
        steppers = {name: _stepper(held_node(name), dtype) for name in bodies}

        @settings(max_examples=max_examples, deadline=None, database=None, phases=(Phase.generate,),
                  suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
        @given(_draws())
        def check(drawn: tuple[Any, ...]) -> None:
            found: list[str] = []
            any_friction = False
            for name, stepper in steppers.items():
                pose, twist, load, mu, weight = _drawn(name, drawn, dtype)
                out = stepper(pose, twist, load, mu)
                power = float(out["contact_friction_power"])
                traction = np.asarray(out["contact_traction"], dtype=np.float64)
                any_friction = any_friction or bool(np.max(np.linalg.norm(traction, axis=-1)) > 0.0)
                if not power <= floor:
                    found.append(f"{name}: friction's power at the solved twist {power:.3g} over its scale, above "
                                 f"its floor {floor:.3g}, at {drawn}")
                if name == "sphere" and plant is None:
                    found += _sphere_limits(held, stepper, (pose, twist, load, mu), out, weight, eps, drawn)
            rubbed.append(any_friction)
            judged.append(not found)
            failures.extend(found)

        check()
    finally:
        friction_module.disk_update = original
        _restored(declared)
        jax.clear_caches()
    failed = judged.count(False)
    acted = rubbed.count(True)
    if failed:
        return VerificationResult("friction law", "FAIL", detail=f"{type(held).__name__}: failed on {failed} of "
                                                                 f"{len(judged)} draws ({acted} with friction); "
                                                                 f"first: {failures[0]}")
    return VerificationResult("friction law", "PASS", detail=f"{type(held).__name__}: passive, its limits held and its "
                                                             f"derivative right on {len(judged)} draws ({acted} with "
                                                             f"friction)")


def _sphere_limits(law: Any, stepper: Callable[..., Any], drawn: tuple[Any, ...], out: dict[str, Any], weight: float,
                   eps: float, shown: tuple[Any, ...]) -> list[str]:
    """The sphere's sticking limits and derivative at one draw: the traction that sticks it (the solve at a coefficient
    far past any limit), against the law's limit at the normal force the solve took."""
    import jax

    pose, twist, load, mu = drawn
    normal = float(np.sum(np.asarray(out["contact_forces"], dtype=np.float64)))
    if normal <= 0.0:
        return []
    stuck = stepper(pose, twist, load, jnp.asarray(1e6, dtype=mu.dtype))
    needed = float(np.max(np.linalg.norm(np.asarray(stuck["contact_traction"], dtype=np.float64), axis=-1)))
    traction = np.asarray(out["contact_traction"], dtype=np.float64)
    held = float(np.max(np.linalg.norm(traction, axis=-1)))
    slip = float(np.max(np.asarray(out["contact_slip"], dtype=np.float64)))
    speed = float(np.linalg.norm(np.asarray(twist[:3], dtype=np.float64))
                  + 1e-3 * np.linalg.norm(np.asarray(twist[3:], dtype=np.float64))) + _G * _DT
    limit = float(law.limit(normal, slip)) if hasattr(law, "limit") else float(mu) * normal
    limit = float(mu) / float(getattr(law, "mu", float(mu))) * limit  # the drawn coefficient in the law's form
    found: list[str] = []
    if needed < (1.0 - 1e-3) * limit and slip > 1e3 * eps * speed:
        found.append(f"sphere: its sticking traction {needed:.6g} N is within the limit {limit:.6g} N, and it slips "
                     f"at {slip:.3g} m/s, at {shown}")
    if needed > (1.0 + 1e-3) * limit:
        if abs(held - limit) > 1e3 * eps * limit:
            found.append(f"sphere: sliding, its traction {held:.9g} N is not the limit {limit:.9g} N, at {shown}")

        def along(coefficient: Any) -> Any:
            ended = stepper(pose, twist, load, coefficient)
            return jnp.sum(ended["twist"][:3] * stuck["contact_traction"][0])

        derivative = float(cast(Any, jax).grad(along)(mu))
        h = float(mu) * (1e-6 if mu.dtype == jnp.float64 else 1e-2)
        central = (float(along(mu + h)) - float(along(mu - h))) / (2 * h)
        tolerance = 1e-6 if mu.dtype == jnp.float64 else 2e-2
        if not abs(derivative - central) <= tolerance * max(abs(central), 1e-30):
            found.append(f"sphere: sliding, its derivative in mu {derivative:.9g} against the central difference "
                         f"{central:.9g}, at {shown}")
    del weight
    return found


def normal_law_suite(law: Any = None, *, bodies: tuple[str, ...] = BODIES, max_examples: int = 100,
                     sweeps: int = SUITE_SWEEPS) -> VerificationResult:
    """``law`` (a ``NormalLaw``; ``Hold()`` where none is given) through the frictional hold on the module's draws: the
    normal force never below the law's least, no point past the shrunk plane after the step beyond rounding, and the
    hold's residual under ``CONVERGED``."""
    import jax
    from hypothesis import HealthCheck, Phase, given, settings

    from mime.interactions.friction import Hold
    from mime.nodes.body._contact import CONVERGED

    held = Hold() if law is None else law
    dtype = jnp.zeros(()).dtype
    eps = float(jnp.finfo(dtype).eps)
    failures: list[str] = []
    judged: list[bool] = []
    declared = _converged(bodies, sweeps)
    try:
        steppers = {name: (_stepper(held_node(name), dtype), _gaps(held_node(name), dtype)) for name in bodies}

        @settings(max_examples=max_examples, deadline=None, database=None, phases=(Phase.generate,),
                  suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
        @given(_draws())
        def check(drawn: tuple[Any, ...]) -> None:
            found: list[str] = []
            for name, (stepper, gaps) in steppers.items():
                pose, twist, load, mu, _ = _drawn(name, drawn, dtype)
                out = stepper(pose, twist, load, mu)
                least = float(np.min(np.asarray(out["contact_forces"], dtype=np.float64)))
                if least < held.least() - 64 * eps * float(np.max(np.abs(np.asarray(load)))):
                    found.append(f"{name}: a normal force {least:.3g} N below the law's least {held.least()} N")
                deepest = float(np.min(np.asarray(gaps(out["pose"]), dtype=np.float64)))
                if deepest < -1e3 * eps * 1e-2:
                    found.append(f"{name}: a point {-deepest:.3g} m past the shrunk plane after the step")
                if float(out["contact_residual"]) > CONVERGED and name == "sphere":
                    found.append(f"{name}: the hold's residual {float(out['contact_residual']):.3g} past CONVERGED")
            judged.append(not found)
            failures.extend(f"{failure}, at {drawn}" for failure in found)

        check()
    finally:
        _restored(declared)
        jax.clear_caches()
    failed = judged.count(False)
    if failed:
        return VerificationResult("normal law", "FAIL", detail=f"{type(held).__name__}: failed on {failed} of "
                                                               f"{len(judged)} draws; first: {failures[0]}")
    return VerificationResult("normal law", "PASS", detail=f"{type(held).__name__}: no pull, no penetration and the "
                                                           f"hold met on {len(judged)} draws")


def row_kind_suite(kind: str, *, max_examples: int = 100) -> VerificationResult:
    """The row kind ``kind`` (``mime.interactions.friction.ROW_KINDS``): built, its rows measure what it declares on
    drawn normals, arms and twists (sliding: ``t . (v + omega x r)``, ``t`` a unit frame across the normal); reserved,
    a law adding it is refused by the contact with the kind's reason."""
    from hypothesis import HealthCheck, Phase, given, settings
    from hypothesis import strategies as st

    from mime.interactions.friction import ROW_KINDS
    from mime.nodes.body._friction import traction_rows

    record = ROW_KINDS.get(kind)
    if record is None:
        return VerificationResult("row kind", "FAIL", detail=f"no row kind {kind!r}; the kinds are {sorted(ROW_KINDS)}")
    if not record.built:
        from mime.interactions import Contact
        from mime.interactions.friction import Coulomb

        law = type("Reserved", (Coulomb,), {"rows": (kind,), "held_rows": lambda self: (kind,)})(mu=0.3)
        found = Contact.Projection(margin=1e-4, friction=law).setting_problems({"margin": 1e-4})
        if record.reason in found:
            return VerificationResult("row kind", "PASS", detail=f"{kind}: reserved, refused with its reason")
        return VerificationResult("row kind", "FAIL", detail=f"{kind}: reserved, and a law adding it is not refused "
                                                             f"with its reason: {found}")
    if kind not in ("sliding", "rolling"):
        return VerificationResult("row kind", "FAIL", detail=f"{kind}: built, and the suite has no check of its rows")
    dtype = jnp.zeros(()).dtype
    eps = float(jnp.finfo(dtype).eps)
    failures: list[str] = []
    unit = st.tuples(st.floats(0.0, math.pi), st.floats(0.0, 2 * math.pi)).map(lambda a: _unit(a[0], a[1]))
    # lengths at physical scales (a body's speeds and levers), never a subnormal one (the module's draws' rule)
    size = st.floats(math.log(1e-6), 0.0).map(math.exp)
    vector = st.tuples(unit, size).map(lambda held: tuple(held[1] * c for c in held[0]))

    @settings(max_examples=max_examples, deadline=None, database=None, phases=(Phase.generate,),
              suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
    @given(unit, vector, st.floats(0.0, 2 * math.pi), vector, vector)
    def check(normal: Any, arm: Any, turn: float, linear: Any, angular: Any) -> None:
        n = np.asarray(normal, dtype=np.float64)
        # the frame's direction across the normal, turned about it by the drawn angle: never along it, as at a valid
        # state (a tube's axis across its wall's normals)
        first = np.cross(n, np.eye(3)[int(np.argmin(np.abs(n)))])
        first /= np.linalg.norm(first)
        along = math.cos(turn) * first + math.sin(turn) * np.cross(n, first) + 0.3 * n
        r = 1e-3 * np.asarray(arm, dtype=np.float64)
        rows = jnp.concatenate([jnp.asarray(n, dtype=dtype), jnp.asarray(np.cross(r, n), dtype=dtype)])[None]
        held = np.asarray(traction_rows(rows, jnp.asarray(r[None], dtype=dtype), jnp.zeros(3, dtype=dtype),
                                        jnp.asarray(along, dtype=dtype), rolling=kind == "rolling"),
                          dtype=np.float64)[0]
        v, w = 1e-2 * np.asarray(linear, dtype=np.float64), 10.0 * np.asarray(angular, dtype=np.float64)
        point = v + np.cross(w, r)
        if kind == "rolling":
            for sliding_row, rolling_row in zip(held[1:3], held[3:5], strict=True):
                t = np.asarray(sliding_row[:3], dtype=np.float64)
                measured = float(rolling_row @ np.concatenate([v, w]))
                if float(np.max(np.abs(rolling_row[:3]))) != 0.0 or \
                        abs(measured - float(t @ w)) > 64 * eps * float(np.linalg.norm(w)):
                    failures.append(f"a rolling row {rolling_row} does not measure the rotation about {t}")
            return
        for row in held[1:]:
            t = np.asarray(row[:3], dtype=np.float64)
            if abs(float(np.linalg.norm(t)) - 1.0) > 64 * eps or abs(float(np.dot(t, n))) > 64 * eps:
                failures.append(f"a frame row {t} is not a unit direction across the normal {n}")
            measured = float(row @ np.concatenate([v, w]))
            if abs(measured - float(t @ point)) > 64 * eps * (np.linalg.norm(v) + np.linalg.norm(w) * 1e-3):
                failures.append(f"a sliding row measures {measured:.9g}, the point's slip along it is "
                                f"{float(t @ point):.9g}")
        if abs(np.dot(held[1, :3], held[2, :3])) > 64 * eps:
            failures.append("the sliding pair's directions are not orthogonal")

    check()
    if failures:
        return VerificationResult("row kind", "FAIL", detail=f"{kind}: {len(failures)} failures; first: {failures[0]}")
    measured = "the slip of the body's own point" if kind == "sliding" else "the body's rotation about the tangents"
    return VerificationResult("row kind", "PASS", detail=f"{kind}: its rows measure {measured}")


__all__ = ["BODIES", "SUITE_SWEEPS", "friction_law_suite", "held_node", "normal_law_suite", "row_kind_suite"]
