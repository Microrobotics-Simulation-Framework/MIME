"""The contact's conformance suites: what every friction law, normal law and friction row kind passes, judged through the
body node's own frictional hold (``mime.interactions.friction``'s seams; decided Fr6), on draws at physical scales in
the run's dtype. A law of a lab's passes them with its own oracle beside them (Coulomb's: ``tests/oracles/
test_friction.py``, breakaway at ``tan(theta) = mu``).

- :func:`friction_law_suite`: **passivity, never silent**, each held point's friction power at the solved twist never
  above its floor (``_friction.power_floor``: the spike's guard, which caught a planted sign flip in every draw where
  the kinetic energy missed most) and the step not unconverged, at a converged count (but where rigid Coulomb holds no
  solution, a draw counted apart: ``NO_SOLUTION_DRAWS``); a planted sign flip raises one of
  the two in every draw of every body where friction acts, the clean solve's friction above the guard's line of the
  load (the guard, or ``unconverged`` where the flip lifts the body off and leaves no friction to read: two draws in
  915 of a box's in W2b's probe); **the sticking limits**, a sphere's point whose sticking traction is within the
  law's limit sticks (its slip at rounding) and one past it slides on the limit, anti-parallel to its slip (the law's
  ``limit(load, speed)`` read against the solve); **its derivative** in the law's coefficient, reverse mode against
  central differences, on sliding draws.
- :func:`normal_law_suite`: the normal force never below the law's least (``NormalLaw.least``), no point past the
  shrunk wall after the step beyond rounding, and the hold's complementarity met (its residual under
  ``FRICTION_CONVERGED``).

Each suite runs the law it is given through the solve, so a law the contact's solve does not hold (refused by the
contact by name, ``mime.interactions.friction.unheld``) fails its suite unrun, never judged as the default in its
place.
- :func:`row_kind_suite`: a built row kind's rows measure what it declares (sliding: the slip of the body's own point,
  ``t . (v + omega x r)``, on a frame across the normal; rolling: the rotation about the tangents, ``t . omega``); a
  reserved kind is refused by the contact with its reason.

Draws (a sphere of 1 mm, a cylinder 1 mm by 3 mm and a box 1 by 1 by 0.5 mm held by its samples, on the plane
``z >= 0``, 2000 kg/m^3, a step of 1 ms): a
load of 0.3 to 3 weights at up to 1.4 rad from the normal in any azimuth, linear speeds along the plane of 1e-6 to 5e-2
m/s or none, angular speeds of 1e-3 to 50 rad/s or none, any orientation, and a coefficient log-uniform on [0.01, 1.2]
(the register's range). Nothing below a body's scale is drawn: the numerical extremes (a bound at rounding, a normal
force of zero) are tested on purpose in ``tests/oracles/test_friction_suites.py``. Two regions are not drawn, so these
suites are no evidence there: a load past 1.4 rad from the normal (near the tangent), and the sphere's limits within
1e-3 of its stick threshold (judged on neither side). W2b's review ran the first apart, 540 cases a dtype (polar 1.4
to 1.5707 rad, the three bodies at rest and moving, mu 0.01 to 1.2): no guard trip, no residual past its line and the
sphere's limits held, in float64 and float32."""

from __future__ import annotations

import math
import operator
from collections.abc import Callable
from typing import Any, cast

import numpy as np
from maddening.testing.verification import VerificationResult

from mime.core import jnp

#: The bodies the suites hold on the plane, by name: kind, settings, mass and principal moments.
BODIES = ("sphere", "cylinder", "box")
_DT = 1e-3
_G = 9.81


def _body(name: str) -> tuple[Any, dict[str, Any], float, tuple[float, ...], float]:
    from mime.scene import Box, Cylinder, Sphere
    from mime.units import mm

    if name == "sphere":
        mass = 2000.0 * 4.0 / 3.0 * math.pi * 1e-3 ** 3
        return Sphere(radius=mm(1.0)), {"radius": 1e-3}, mass, (0.4 * mass * 1e-6,) * 3, 1e-3
    if name == "box":
        a, b, c = 1e-3, 1e-3, 5e-4
        mass = 2000.0 * a * b * c
        return (Box(size=mm((1.0, 1.0, 0.5))), {"size": (a, b, c)}, mass,
                (mass * (b * b + c * c) / 12.0, mass * (a * a + c * c) / 12.0, mass * (a * a + b * b) / 12.0), c / 2)
    radius, length = 5e-4, 3e-3
    mass = 2000.0 * math.pi * radius ** 2 * length
    side = mass * (3 * radius ** 2 + length ** 2) / 12.0
    return (Cylinder(radius=mm(0.5), length=mm(3.0), axis=(1.0, 0.0, 0.0)),
            {"radius": radius, "length": length, "axis": (1.0, 0.0, 0.0)}, mass, (0.5 * mass * radius ** 2, side, side),
            radius)


def held_node(name: str, *, mu: float = 0.3, friction: bool = True, rolling: float | None = None) -> Any:
    """A body of :data:`BODIES` held by its hull on the half-space ``z >= 0`` with Coulomb friction (``mu``, a leaf;
    with ``rolling``, rolling resistance of that length, a leaf too), its lowest point on the plane (the contact
    0.5.0's loops)."""
    from mime.nodes.body import HULL, RigidInertialMidpointNode
    from mime.scene._sampling import quadrature_counts

    kind, settings, mass, moments, lowest = _body(name)
    patches = tuple(kind.patches)
    held = {"friction": "coulomb" if rolling is None else "coulomb_rolling", "friction_mu": mu,
            "friction_law": "contact:body.friction",
            **({} if rolling is None else {"friction_rolling": rolling})} if friction else {}
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
    kind, settings, _, _, lowest = _body(name)
    w, x, y, z = q
    rot = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    if name == "box":
        # held by its samples (a kind with no support): its lowest sample on the plane
        from mime.scene._sampling import quadrature_arrays, quadrature_counts

        patches = tuple(kind.patches)
        points = quadrature_arrays(kind, settings, patches, quadrature_counts(kind, settings, 2e-4, patches))[0]
        lowest = float(np.max(-(np.asarray(points, dtype=np.float64) @ rot.T)[:, 2]))
    if name == "cylinder":
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


#: The sweeps the suites solve at: enough to converge almost every drawn problem (the spike's reference was 4,000; its
#: four-corner box, the slowest, reached 1e-6 of its change by 89 at the 99th percentile), so that a law's passivity is
#: judged at its solution. At the declared count a trip of the guard on an unconverged iterate raises unconverged
#: (the frictional hold's residual), which the sweep-count measurement reports.
SUITE_SWEEPS = 400
#: Where a clean solve is left unconverged at the suite's count (its residual past ``FRICTION_CONVERGED``), the counts
#: it is solved again at, in turn, and judged at the first it converges at, or the last: a box face down on the plane,
#: rocking with mu 1, five or six of its eight points held in no unique share, converges slowly and not monotonically
#: (rocking at 1 rad/s: 0.12 at 400 sweeps, 0.013 at 1,600, 3e-5 at 6,400; at 0.086 rad/s: 0.016 at 400, 0.020 at
#: 1,600, 0.017 at 6,400, 0.0057 at 25,600, 6.3e-4 at 102,400; float64, about one box draw in 1,000 of the suite's
#: past 400 and one in 4,000 past 25,600).
SUITE_DEEPER = (1600, 6400, 25600, 102400, 409600)
#: The share of a body's draws the suite solves again at the deeper counts, at most: past it a draw left unconverged is
#: judged at the suite's count, where it fails, so a solve broken into slowness fails naming its count rather than
#: running the deeper counts on every draw (the review's planted bugs took 59 to 158 minutes there). At the frictional
#: line a clean run needs them on 0 to 5 of a body's 200 draws in most runs (four runs, both dtypes: the box the most),
#: but Hypothesis's draws can cluster: one run of W2b PR 1's CI (Python 3.13, float32) needed them on more than 41 of
#: 187 box draws, each converging, so the share is a half, 100 draws, and a draw whose residual stalls stops early
#: (:data:`STALLED`), so that a broken solve's draws each cost two deeper counts, not five.
DEEPER_SHARE = 0.5
#: A deeper count that leaves the residual above this share of the count before it makes no progress; two in a row and
#: the draw stops there, judged a configuration with no solution where its guard is quiet (:data:`NO_SOLUTION_DRAWS`).
#: The slowest converging box of the measurement above (0.016 at 400, 0.020 at 1,600, 0.017 at 6,400, 0.0057 at
#: 25,600) falls by more than a tenth at every other count; the Painleve-type box stalls at 0.0247 from 1,600 on.
STALLED = 0.9
#: The share of the load above which a planted sign flip of friction is seen, a control's "friction acts", by dtype:
#: the flip measured raised down to 3e-11 of the load on the sphere in float64 (1e-17 on the cylinder and the box) and
#: 3e-6 in float32 (W2b's review: a body at rest under a load drawn off the normal, three bodies), each line three times
#: that. Under it the flip moves the body by 5.6e-15 m/s a step in float64 and 5.6e-9 m/s in float32, under any
#: check's resolution; the guard's own line (``_friction.power_floor``) is a power's share, not this.
FLIP_SEEN = {"float64": 1e-10, "float32": 1e-5}
#: A draw at a high coefficient under a load far off the normal can be a configuration rigid Coulomb holds no solution
#: for (Painleve-type, the validity line's: the review's box at mu 1.18 and 78 deg, held at a residual of 0.017 at 80
#: and 4,000 sweeps; a stress run's box at mu 1.2 and 82 deg, 0.025 at every count to 409,600, both dtypes alike). Its
#: step raises unconverged, as it should, and its friction stays passive. Unplanted, a draw whose deeper counts stall
#: (:data:`STALLED`) or reach the deepest unconverged, with the guard quiet, is counted apart as such, at most this many
#: a body a run: past them the draws fail, as a broken solve leaves more.
NO_SOLUTION_DRAWS = 2


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
    sign flip: ``lambda x: -x`` on its force); the clean solve beside it says where friction acts, its friction above
    :data:`FLIP_SEEN` of the load (below it a plant moves the step by less than any check resolves: a load within
    rounding of the normal on a body at rest), and the detail counts those draws apart, as it counts
    apart the draws whose clean solve is itself unconverged at ``sweeps`` (where the clean step raises the flag the
    plant is judged by). Unplanted, a draw left unconverged at ``sweeps`` is solved again at :data:`SUITE_DEEPER`'s
    counts and judged at its solution, for up to :data:`DEEPER_SHARE` of a body's draws (the detail counts them); one
    whose residual stalls (:data:`STALLED`) unconverged with the guard quiet is a configuration rigid Coulomb holds no
    solution for, counted apart for up to :data:`NO_SOLUTION_DRAWS` a body."""
    import jax
    from hypothesis import HealthCheck, Phase, given, settings

    import mime.nodes.body._friction as friction_module
    from mime.interactions.friction import Coulomb, unheld
    from mime.nodes.body._friction import FRICTION_CONVERGED
    from mime.nodes.body._friction import power_floor

    held = Coulomb(mu=0.3) if law is None else law
    refused = unheld(held, None)
    if refused:
        return VerificationResult("friction law", "FAIL", detail=f"{type(held).__name__}: not run, the contact refuses "
                                                                 f"it: {refused[0]}")
    dtype = jnp.zeros(()).dtype
    floor = power_floor(dtype)
    seen = FLIP_SEEN[jnp.dtype(dtype).name]
    eps = float(jnp.finfo(dtype).eps)
    original = friction_module.disk_update
    failures: list[str] = []
    judged: list[bool] = []
    rubbed: list[bool] = []
    acted = dict.fromkeys(bodies, 0)
    flagged = dict.fromkeys(bodies, 0)
    under = dict.fromkeys(bodies, 0)
    solved_deeper = dict.fromkeys(bodies, 0)
    unsettled = dict.fromkeys(bodies, 0)
    no_solution = dict.fromkeys(bodies, 0)
    quiet: list[str] = []
    declared = _converged(bodies, sweeps)
    try:
        clean: dict[str, Callable[..., dict[str, Any]]] = {}
        if plant is not None:
            # the clean solve, traced before the plant goes in (a step is traced at its first call)
            clean = {name: _stepper(held_node(name), dtype) for name in bodies}
            for name, stepper in clean.items():
                stepper(*_drawn(name, _AT_REST, dtype)[:4])

            def planted(block: Any, load: Any, bound: Any) -> Any:
                return plant(original(block, load, bound))

            friction_module.disk_update = planted
        steppers = {name: _stepper(held_node(name), dtype) for name in bodies}
        deeper: dict[tuple[str, int], Callable[..., dict[str, Any]]] = {}

        @settings(max_examples=max_examples, deadline=None, database=None, phases=(Phase.generate,),
                  suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
        @given(_draws())
        def check(drawn: tuple[Any, ...]) -> None:
            found: list[str] = []
            any_friction = False
            for name, stepper in steppers.items():
                pose, twist, load, mu, weight = _drawn(name, drawn, dtype)
                out = stepper(pose, twist, load, mu)
                # judged at its solution: solved again at the deeper counts while unconverged (a planted arm is
                # judged at the suite's count, where its flags must rise)
                deepened, stalled, strikes = False, False, 0
                room = solved_deeper[name] < max(1, int(DEEPER_SHARE * max_examples))
                for count in [count for count in SUITE_DEEPER if count > sweeps and plant is None and room]:
                    before = float(out["contact_residual"])
                    if not before > FRICTION_CONVERGED:
                        break
                    if (name, count) not in deeper:
                        deeper[name, count] = _traced_at(name, count, dtype)
                    out = deeper[name, count](pose, twist, load, mu)
                    deepened = True
                    # two deeper counts in a row that leave the residual where it was: no solution to come
                    strikes = strikes + 1 if float(out["contact_residual"]) > STALLED * before else 0
                    stalled = strikes == 2 or count == SUITE_DEEPER[-1]
                    if strikes == 2:
                        break
                solved_deeper[name] += int(deepened)
                power = float(out["contact_friction_power_ratio"])
                left = float(out["contact_residual"])
                loaded = float(np.linalg.norm(np.asarray(load[:3], dtype=np.float64)))
                reference = clean[name](pose, twist, load, mu) if name in clean else out
                pulled = _pulled(reference)
                # a planted arm is judged where the clean solve is itself converged at the suite's count: where it is
                # not, the clean step raises the flag the plant is judged by, and the draw is counted apart
                settled = name not in clean or not (float(reference["contact_residual"]) > FRICTION_CONVERGED
                                                    or float(reference["contact_friction_power_ratio"]) > floor)
                unsettled[name] += int(not settled)
                acts = pulled > seen * loaded and settled
                any_friction = any_friction or acts
                # the step is never silent: the guard reads friction pushing, or unconverged rises (the residual,
                # where a planted flip lifts the body off and leaves no friction to read)
                raised = not power <= floor or left > FRICTION_CONVERGED
                # unplanted, stalled unconverged with the guard quiet: no solution to hold (counted apart)
                unsolved = (stalled and power <= floor and left > FRICTION_CONVERGED
                            and no_solution[name] < NO_SOLUTION_DRAWS)
                no_solution[name] += int(unsolved)
                acted[name] += int(acts and not unsolved)
                flagged[name] += int(acts and raised and not unsolved)
                under[name] += int(not acts and settled)
                if acts and not raised:
                    quiet.append(f"{name} at {drawn}: friction {pulled:.3g} N over the load {loaded:.3g} N, power "
                                 f"{power:.3g}, residual {left:.3g}")
                if raised and not unsolved:
                    found.append(f"{name}: flagged, friction's power at the solved twist {power:.3g} over its scale "
                                 f"(its floor {floor:.3g}) and the residual {left:.3g} (its line "
                                 f"{FRICTION_CONVERGED:g}), at "
                                 f"{drawn}")
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
    counts = (f"flagged/with friction by body: "
              f"{', '.join(f'{name} {flagged[name]}/{acted[name]}' for name in bodies)}; "
              f"friction under the flip's resolution of the load ({seen:g}), by body: "
              f"{', '.join(f'{name} {under[name]}' for name in bodies)}; solved past {sweeps} sweeps to converge, by "
              f"body: {', '.join(f'{name} {solved_deeper[name]}' for name in bodies)} (at most "
              f"{max(1, int(DEEPER_SHARE * max_examples))} a body, past which a draw is judged at {sweeps}); "
              f"stalled unconverged with the guard quiet, a configuration rigid Coulomb holds no "
              f"solution for, counted apart (at most {NO_SOLUTION_DRAWS} a body), by body: "
              f"{', '.join(f'{name} {no_solution[name]}' for name in bodies)}"
              + (f"; the clean solve unconverged at {sweeps}, counted apart from the plant's, by body: "
                 f"{', '.join(f'{name} {unsettled[name]}' for name in bodies)}" if clean else ""))
    if failed:
        return VerificationResult("friction law", "FAIL", detail=f"{type(held).__name__}: failed on {failed} of "
                                                                 f"{len(judged)} draws ({rubbed.count(True)} with "
                                                                 f"friction); {counts}; first: {failures[0]}"
                                                                 + (f"; with friction and quiet: {quiet[:3]}" if quiet
                                                                    else ""))
    return VerificationResult("friction law", "PASS", detail=f"{type(held).__name__}: passive, its limits held and its "
                                                             f"derivative right on {len(judged)} draws "
                                                             f"({rubbed.count(True)} with friction); {counts}")


#: A draw at rest on the plane (a weight straight down, no motion), the clean steps' first call.
_AT_REST = (1.0, 0.0, 0.0, (0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), 0.3)


def _traced_at(name: str, count: int, dtype: Any) -> Callable[..., dict[str, Any]]:
    """``name``'s step at ``count`` sweeps, traced there (at its first call), the suite's count restored after."""
    held = _converged((name,), count)
    try:
        stepper = _stepper(held_node(name), dtype)
        stepper(*_drawn(name, _AT_REST, dtype)[:4])
    finally:
        _restored(held)
    return stepper


def _pulled(out: dict[str, Any]) -> float:
    """The body's whole friction, the length of its points' tractions' sum."""
    return float(np.linalg.norm(np.sum(np.asarray(out["contact_friction_forces"], dtype=np.float64), axis=0)))


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
    needed = float(np.max(np.linalg.norm(np.asarray(stuck["contact_friction_forces"], dtype=np.float64), axis=-1)))
    traction = np.asarray(out["contact_friction_forces"], dtype=np.float64)
    held = float(np.max(np.linalg.norm(traction, axis=-1)))
    slip = float(np.max(np.asarray(out["contact_slip_speed"], dtype=np.float64)))
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

        # along the whole sticking traction's direction (one point shared by eight groups splits it in no unique
        # way, so no one group's is the body's): the sliding force's direction, where the speed's derivative in mu is
        # about the normal force over the mass, over the step
        pulled = jnp.sum(stuck["contact_friction_forces"], axis=0)
        direction = pulled / jnp.sqrt(jnp.sum(pulled * pulled))

        def along(coefficient: Any) -> Any:
            ended = stepper(pose, twist, load, coefficient)
            return jnp.sum(ended["twist"][:3] * direction)

        derivative = float(cast(Any, jax).grad(along)(mu))
        # the sweeps leave the solve's dependence on mu non-smooth at their remainder (eight groups share the sphere's
        # point, and which takes its force is a tie the sweeps break), measured at about 1e-8 of the speed at the
        # suite's count: the difference is taken at 1e-4 of mu and judged at 1e-3 of the scale (float32: 1e-2, 2e-2),
        # its span kept on the sliding side of the stick threshold (mu times needed over limit; a span across it reads
        # the kink, not the derivative: a float32 draw at mu 1.0005, its threshold 1.004, read 0.69 of it at 1e-2)
        h = float(mu) * min(1e-4 if mu.dtype == jnp.float64 else 1e-2, 0.5 * (needed / limit - 1.0))
        central = (float(along(mu + h)) - float(along(mu - h))) / (2 * h)
        tolerance = 1e-3 if mu.dtype == jnp.float64 else 2e-2
        # judged on the scale it came from: the derivative's own (the normal force over the mass, over the step) and
        # the central difference's rounding (the speed's rounding over its step)
        scale = normal * _DT / _body("sphere")[2]
        rounding = 64 * eps * float(np.linalg.norm(np.asarray(twist[:3], dtype=np.float64)) + _G * _DT) / h
        if not abs(derivative - central) <= tolerance * scale + rounding:
            found.append(f"sphere: sliding, its derivative in mu {derivative:.9g} against the central difference "
                         f"{central:.9g} (its scale {scale:.3g}), at {shown}")
    del weight
    return found


#: The bodies the rolling suite holds (a sphere and a cylinder lying on its side: the two that roll on the plane), and
#: the drawn rolling resistance as a share of the body's radius, log-uniform (a 1 mm sphere's mu_r from 0.1 um to
#: 0.1 mm: the friction spike's range, its stopping deceleration (5/7) mu_r g / a up to a fourteenth of g).
ROLLING_BODIES = ("sphere", "cylinder")
ROLLING_SHARE = (1e-4, 0.1)
#: The share of a body's draws the rolling suite solves again at the deeper counts, at most: a cylinder rocking on its
#: line contact with its rolling rows converges slowly (one float64 run of 200 draws needed the deeper counts on 11 of
#: its cylinder draws, at mu 1 turning about every axis; another on none): a quarter, 50 draws, past which a broken
#: solve fails naming the count (the friction suite's share and its stall stop are its own, :data:`DEEPER_SHARE`).
ROLLING_DEEPER_SHARE = 0.25


def rolling_resistance_suite(law: Any = None, *, bodies: tuple[str, ...] = ROLLING_BODIES, max_examples: int = 200,
                             planted: bool = False, sweeps: int = SUITE_SWEEPS,
                             deeper_share: float = ROLLING_DEEPER_SHARE) -> VerificationResult:
    """**Rolling resistance dissipates, never injects**, through the body's frictional hold with its rolling rows
    (``law``, a ``Coulomb`` with ``rolling``; one of 1 um where none is given), on the friction law suite's draws with
    a rolling resistance drawn too (:data:`ROLLING_SHARE` of the body's radius), in the run's dtype, solved at
    ``sweeps``: each held point's friction power at the solved twist, its rolling pair's included, never above the
    guard's floor and the step not unconverged (a draw unconverged at ``sweeps`` solved again at
    :data:`SUITE_DEEPER`'s counts, for up to ``deeper_share`` of a body's draws, :data:`ROLLING_DEEPER_SHARE`).
    ``planted``, the control: the rolling resistance's sign flipped in the solve (its bound ``-mu_r N``: the rolling
    pair's torque along the rotation it acts on), which must raise the guard or ``unconverged`` on every draw where rolling resistance acts at
    its limit (the body turning about the plane's tangents at the solved twist), the clean step settled (the draws
    where it holds the rotation inside its limit, where a flipped bound moves nothing, where it moves the step's
    turning by less than the guard's line of the load's moment, or where the clean step itself is flagged, counted
    apart)."""
    import jax
    from hypothesis import HealthCheck, Phase, given, settings

    from mime.interactions.friction import Coulomb, unheld
    from mime.nodes.body._friction import FRICTION_CONVERGED, power_floor
    from mime.units import um

    held = Coulomb(mu=0.3, rolling=um(1.0)) if law is None else law
    refused = unheld(held, None)
    if refused or held.rolling is None:
        return VerificationResult("rolling resistance", "FAIL", detail=f"{type(held).__name__}: not run: "
                                  f"{refused[0] if refused else 'it holds no rolling resistance (rolling=...)'}")
    dtype = jnp.zeros(()).dtype
    floor = power_floor(dtype)
    declared = _converged(bodies, sweeps)
    counts = {key: dict.fromkeys(bodies, 0) for key in ("acted", "flagged", "under", "unsettled", "deeper")}
    failures: list[str] = []
    judged: list[bool] = []
    try:
        steppers = {name: _rolling_stepper(name, dtype) for name in bodies}
        deeper: dict[tuple[str, int], Callable[..., dict[str, Any]]] = {}

        @settings(max_examples=max_examples, deadline=None, database=None, phases=(Phase.generate,),
                  suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
        @given(_draws(), _rolling_shares())
        def check(drawn: tuple[Any, ...], share: float) -> None:
            found: list[str] = []
            for name, stepper in steppers.items():
                pose, twist, load, mu, _ = _drawn(name, drawn, dtype)
                _, _, _, moments, radius = _body(name)
                mu_r = jnp.asarray(share * radius, dtype=dtype)
                clean = stepper(pose, twist, load, mu, mu_r)
                room = counts["deeper"][name] < max(1, int(deeper_share * max_examples))
                for count in [count for count in SUITE_DEEPER if count > sweeps and room]:
                    if not (float(clean["contact_residual"]) > FRICTION_CONVERGED
                            or float(clean["contact_friction_power_ratio"]) > floor):
                        break
                    if (name, count) not in deeper:
                        held_count = _converged((name,), count)
                        try:
                            deeper[name, count] = _rolling_stepper(name, dtype)
                            deeper[name, count](*_drawn(name, _AT_REST, dtype)[:4], mu_r)
                        finally:
                            _restored(held_count)
                    clean = deeper[name, count](pose, twist, load, mu, mu_r)
                    counts["deeper"][name] += 1
                settled = not (float(clean["contact_residual"]) > FRICTION_CONVERGED
                               or float(clean["contact_friction_power_ratio"]) > floor)
                if not planted:
                    if not settled:
                        found.append(f"{name}: flagged (power {float(clean['contact_friction_power_ratio']):.3g}, "
                                     f"residual {float(clean['contact_residual']):.3g}) at {drawn}, mu_r {share:g} a")
                    continue
                # where rolling resistance acts and is at its limit (a flipped bound moves only a pair on it): the
                # body turns about the plane's tangents at the solved twist, and rolling resistance changes the step's
                # turning, as a moment, by more than the guard's line of the load's
                free = stepper(pose, twist, load, mu, jnp.zeros_like(mu_r))
                spun = np.asarray(clean["twist"][3:], dtype=np.float64)
                turned = spun - np.asarray(free["twist"][3:], dtype=np.float64)
                moment = float(np.linalg.norm(turned * np.asarray(moments))) / _DT
                loaded = float(np.linalg.norm(np.asarray(load[:3], dtype=np.float64))) * radius
                speed = float(np.linalg.norm(np.asarray(clean["twist"][:3], dtype=np.float64))) + radius * float(
                    np.linalg.norm(spun)) + _G * _DT
                rolls = float(np.linalg.norm(spun[:2])) * radius > math.sqrt(float(jnp.finfo(dtype).eps)) * speed
                acts = moment > floor * loaded and rolls and settled
                counts["unsettled"][name] += int(not settled)
                counts["under"][name] += int(settled and not acts)
                if not acts:
                    continue
                out = stepper(pose, twist, load, mu, -mu_r)
                raised = (float(out["contact_friction_power_ratio"]) > floor
                          or float(out["contact_residual"]) > FRICTION_CONVERGED)
                counts["acted"][name] += 1
                counts["flagged"][name] += int(raised)
                if not raised:
                    found.append(f"{name}: rolling resistance's sign flipped and nothing raised at {drawn}, mu_r "
                                 f"{share:g} a")
            judged.append(not found)
            failures.extend(found)

        check()
    finally:
        _restored(declared)
        jax.clear_caches()
    detail = "; ".join(f"{key} by body: {', '.join(f'{name} {value[name]}' for name in bodies)}"
                       for key, value in counts.items())
    failed = judged.count(False)
    if planted:
        missed = sum(counts["acted"][name] - counts["flagged"][name] for name in bodies)
        status = "FAIL" if missed == 0 and all(counts["acted"].values()) else "PASS"
        return VerificationResult("rolling resistance", status, detail=f"planted: {detail}; first quiet: "
                                                                      f"{failures[0] if failures else 'none'}")
    if failed:
        return VerificationResult("rolling resistance", "FAIL", detail=f"failed on {failed} of {len(judged)} draws; "
                                                                      f"{detail}; first: {failures[0]}")
    return VerificationResult("rolling resistance", "PASS", detail=f"passive on {len(judged)} draws; {detail}")


def _rolling_shares() -> Any:
    from hypothesis import strategies as st

    return st.floats(math.log(ROLLING_SHARE[0]), math.log(ROLLING_SHARE[1])).map(math.exp)


def _rolling_stepper(name: str, dtype: Any) -> Callable[..., dict[str, Any]]:
    """One step of ``name`` with rolling resistance from a drawn pose, twist, load, coefficient and rolling resistance,
    compiled once."""
    import jax

    node = held_node(name, rolling=1e-6)
    leaves = node.params_pytree()
    start = node.initial_state()

    def step(pose: Any, twist: Any, load: Any, mu: Any, mu_r: Any) -> dict[str, Any]:
        state = {**start, "pose": pose, "twist": twist}
        given = {"applied_wrench": load, "resistance": jnp.zeros((6, 6), dtype=dtype),
                 "background_load": jnp.zeros(6, dtype=dtype)}
        return node.advance({**node._step_constants(None), **leaves, "friction_mu": mu,  # noqa: SLF001
                             "friction_rolling": mu_r}, state, given, _DT)

    return cast(Callable[..., dict[str, Any]], cast(Any, jax).jit(step))


def normal_law_suite(law: Any = None, *, bodies: tuple[str, ...] = BODIES, max_examples: int = 100,
                     sweeps: int = SUITE_SWEEPS) -> VerificationResult:
    """``law`` (a ``NormalLaw``; ``Hold()`` where none is given) through the frictional hold on the module's draws: the
    normal force never below the law's least, no point past the shrunk plane after the step beyond rounding, and the
    hold's residual under ``FRICTION_CONVERGED``."""
    import jax
    from hypothesis import HealthCheck, Phase, given, settings

    from mime.interactions.friction import Hold, unheld
    from mime.nodes.body._friction import FRICTION_CONVERGED

    held = Hold() if law is None else law
    refused = unheld(None, held)
    if refused:
        return VerificationResult("normal law", "FAIL", detail=f"{type(held).__name__}: not run, the contact refuses "
                                                               f"it: {refused[0]}")
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
                if float(out["contact_residual"]) > FRICTION_CONVERGED and name == "sphere":
                    found.append(f"{name}: the hold's residual {float(out['contact_residual']):.3g} past its line "
                                 f"{FRICTION_CONVERGED:g}")
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

        def reserved(self: object) -> tuple[str, ...]:
            del self
            return (kind,)

        law = type("Reserved", (Coulomb,), {"rows": (kind,), "held_rows": reserved})(mu=0.3)
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


__all__ = ["BODIES", "DEEPER_SHARE", "ROLLING_BODIES", "ROLLING_DEEPER_SHARE", "ROLLING_SHARE", "SUITE_DEEPER",
           "SUITE_SWEEPS", "friction_law_suite", "held_node", "normal_law_suite", "rolling_resistance_suite", "row_kind_suite"]
