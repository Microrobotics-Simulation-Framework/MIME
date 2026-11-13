"""The reverse-mode gates' harness (``test_in_loop_reverse.py``): camera spike R1's ballistic case differentiated through
MIME's differentiated run (what ``Study.gradient`` compiles: reverse mode in checkpointed chunks, forward mode in one),
and the checks, each tolerance from its mechanism.

**The case.** A 20 um steel ball moving along a tube (bore 60 um, wall 10 um, absorbing 3000 per m), seen across it from
1 mm by a camera fixed in the world, 160 x 120 px, fx = fy = 750 px, a frame every ``period`` base steps of 1 ms. The
run is 12 steps from its start (an arm's), and its objective is ``J = sum_t |c_t - p_t|^2`` (pixels squared): ``c_t``
the darkness-weighted centroid of ``expected`` after step ``t``, ``p_t`` the ball's centre after the step projected at
the run's own leaves. The parameters: the ball's speed along the tube and its height at the run's start (``v0x``,
``z0``), and three of the detector's leaves (``fy``, ``cy``, ``k1``). Seed 0 is R1's fixed point; any other seed draws
one at physical scales (the ball in the lumen and in view over the run).

**The arms.** ``reset``: from the build's state. ``continued``: from the state after 6 steps (frames made, the static
structure the tree's). ``overridden``: the same, with ``fy`` written into the tree after the build (3 px above the
structure's numbers), so that every frame step makes the structure from the step's leaves. ``chunked``: the continued
state, differentiated in the three chunks of 4 steps a CALLBACK controller's run is split into (each stretch its own
chunk, the run-constants made once for the run).

**The checks** (tolerances from mechanisms, never from the result's own size):

* **T1, reverse against forward mode**: ``128 eps S``, ``S`` the sum of every pixel's and every projected centre's
  absolute contribution to ``dJ/dtheta`` (from the forward tangents). A pixel's chain is about 50 local Jacobians deep
  and the sums are pairwise over about 2^18 terms (depth 18): (50 + 18) eps, doubled.
* **T2, against central differences**: on a ladder of steps sqrt(10) apart (0.1 to 3e-5 of a pixel's motion), the
  error of ``D(h)`` is bounded by the finer rung's distance over ``1 - 1 / sqrt(10)`` (exact for an error falling as
  ``h``: the box edge is C1 and piecewise in its second derivative, the disc's coverage rises as a power 3/2 at a
  tangent), the coarser rung's Richardson term, and ``2 r_J / h`` at both rungs, ``r_J`` J's rounding: float32's
  measured against float64 over the ladder's points, float64's that times ``eps64 / eps32``. The step judged is the one
  of least bound, chosen without the AD value.
* **T3, float32 against float64 where central differences cannot discriminate**: a float32 parameter whose T2 bound
  exceeds :data:`DISCRIMINATES` of its derivative (a planted error under that share would pass) is judged against the
  float64 reverse derivative at the same point instead: ``|g32 - g64| <= 128 eps32 S + 1.5 sqrt(WIDTH eps32) S``. The
  first term is the chain's rounding (T1's); the second the derivative's own change over float32's forward rounding,
  which moves an edge by about ``WIDTH eps32`` pixels (a coordinate of up to the image's width, rounded): the disc's
  coverage rises as ``d^(3/2)`` past a pixel's side, so its derivative moves by ``1.5 sqrt(delta)`` of its size over
  ``delta`` pixels. Every parameter is judged so in float32 (as R2 found, float32 against float64 separates where
  differences cannot); T2 judges float32 only where it discriminates.
* **H1, held steps**: a held step's centroid derivative (reverse mode) equals its frame step's to ``128 eps S_t``,
  ``S_t`` the frame's terms' scale (each pixel's absolute contribution to the centroid's derivative); and the ball's
  speed's derivative reaches every held step: its reverse value there against its own central difference, as T2.
* **H2, per step, reverse against forward**: ``128 eps S_t``. R1's spike judged H2 on the parameter's largest per-step
  derivative, which ``dc/dfy`` (a difference of the ball's and the tube's parts) undercuts, and flagged one clean draw;
  ``S_t`` is the scale the terms came from.

**The plants** (``CAMERA_PLANT``, before anything is traced; the camera's step is traced once per node): ``edge-stop``,
``stop_gradient`` on the ball's exact disc coverage; ``structure-stop``, on the static structure the tree holds (the
run-constant's derivative); ``made-stop``, on the structure a frame step makes (the overridden arm's path);
``held-stop``, on the held frame (a step that is not a frame step)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from in_loop_cases import DETECTOR, LOOK_ACROSS, PLANT, RADIUS_UM

from mime import Body, Experiment, State, Structure
from mime.build._maddening import differentiated_run, full_state, graph_params
from mime.dynamics import RigidInertial
from mime.observations import Camera, World
from mime.scene import Scene, Sphere, Tube, X, Z
from mime.sensing._detector import CameraPose, project
from mime.sensing._node import INTRINSICS, detector_of
from mime.things import Part
from mime.units import kg_per_m3, m, m_per_s, ms, rad_per_s, um

NAMES = ("v0x", "z0", "fy", "cy", "k1")
STEPS = 12
CAMERA, BALL = "camera:", "ball"
RATIO = math.sqrt(10.0)
EPS = {"float64": float(np.finfo(np.float64).eps), "float32": float(np.finfo(np.float32).eps)}
#: The central differences' steps as fractions of a pixel of image motion (:data:`PIXEL_SCALE`), sqrt(10) apart.
LADDER = tuple(10.0 ** -(1 + k / 2) for k in range(8))
#: One pixel of image motion in each parameter at 160 x 120: the speed's a pixel at the end of a 12-step run, the height's
#: a pixel at 1 mm, the focal length's at the ball's distance from the centre row and the tube's walls (11 to 50 px),
#: k1's at the walls.
PIXEL_SCALE = {"v0x": 1.333e-6 / 12e-3, "z0": 1.333e-6, "fy": 30.0, "cy": 1.0, "k1": 1.0}
#: The share of a derivative under which a float32 central difference's bound must lie for it to judge (R1's smallest
#: planted effect, the static structure's share of a leaf's derivative, was 0.34; a bound of a tenth leaves a third of
#: that as margin).
DISCRIMINATES = 0.1
#: The chain's rounding, in eps of the terms' scale: (50 + 18) eps, doubled.
CHAIN = 128.0


@dataclass(frozen=True)
class Point:
    x0_um: float
    z0_um: float
    speed: float
    period: int


@dataclass(frozen=True)
class Arm:
    start: int
    override: float
    chunks: tuple[int, ...] | None


FIXED = Point(-30.0, 15.0, 2e-3, 3)
ARMS = {"reset": Arm(0, 0.0, None), "continued": Arm(6, 0.0, None), "overridden": Arm(6, 3.0, None),
        "chunked": Arm(6, 0.0, (4, 4, 4))}


def point_of(seed: int) -> Point:
    """R1's fixed point at seed 0, else a draw: the ball inside the lumen (radius 60 um, the ball's 20) and in view over
    the run (half-field 107 um at 160 px)."""
    if seed == 0:
        return FIXED
    rng = np.random.default_rng(seed)
    return Point(float(rng.uniform(-45.0, 0.0)), float(rng.choice([-1.0, 1.0]) * rng.uniform(8.0, 30.0)),
                 float(rng.uniform(0.5e-3, 3e-3)), int(rng.integers(2, 5)))


def case(point: Point) -> Experiment:
    exp = Experiment("ballistic", scene=Scene.new(up=Z, units="m"), dt=ms(1))
    exp.add_structure(Structure("tube", geometry=Tube(axis=+X, inner_radius=um(60), wall=um(10), length=um(480))))
    exp.add_body(Body(BALL, dynamics=RigidInertial(), initial=State(
        position=um((point.x0_um, 0.0, point.z0_um)), orientation=(1.0, 0.0, 0.0, 0.0),
        velocity=m_per_s((point.speed, 0.0, 0.0)), angular_velocity=rad_per_s((0.0, 0.0, 0.0))),
        parts={"steel": Part(Sphere(radius=um(RADIUS_UM)))}))
    exp.declare(f"{BALL}.parts.steel.density", kg_per_m3(7850.0), reason="steel")
    exp.observe(Camera(detector=DETECTOR, mount=World(position=m((0.0, -1e-3, 0.0)), orientation=LOOK_ACROSS),
                       frame_period=ms(point.period), absorbers=(("tube", 3000.0),)))
    return exp


def plant() -> None:
    """Switch on the planted error ``CAMERA_PLANT`` names, before anything is traced."""
    from mime.sensing import _coverage, _node

    if PLANT == "edge-stop":
        disc = _coverage.disc
        _coverage.disc = lambda *a, **k: jax.lax.stop_gradient(disc(*a, **k))  # type: ignore[assignment]
    elif PLANT == "structure-stop":
        carrying = _node.carrying
        _node.carrying = lambda make: (lambda held, numbers: jax.lax.stop_gradient(  # type: ignore[assignment]
            carrying(make)(held, numbers)))
    elif PLANT == "made-stop":
        make = _node.CameraNode._make
        _node.CameraNode._make = lambda self, numbers: jax.lax.stop_gradient(  # type: ignore[method-assign]
            make(self, numbers))
    elif PLANT == "held-stop":
        cond = _node.lax.cond

        class Held:
            def __getattr__(self, name: str) -> Any:
                return getattr(jax.lax, name)

            @staticmethod
            def cond(predicate: Any, made: Any, kept: Any, *operands: Any) -> Any:
                if getattr(kept, "__name__", "") == "keep":
                    held = kept

                    def kept_stopped() -> Any:
                        return jax.lax.stop_gradient(held())

                    return cond(predicate, made, kept_stopped, *operands)
                return cond(predicate, made, kept, *operands)

        _node.lax = Held()


def backlight() -> float:
    return 1000.0


def soft_centroid(expected: Any) -> Any:
    """The darkness-weighted centroid ``(u, v)`` of a frame: each pixel weighted by its share of the backlight it does
    not pass (the tube's faint walls weigh in with the ball)."""
    dark = (backlight() - expected) / backlight()
    rows, columns = expected.shape
    v = jnp.arange(rows, dtype=expected.dtype)[:, None]
    u = jnp.arange(columns, dtype=expected.dtype)[None, :]
    total = jnp.sum(dark)
    return jnp.stack([jnp.sum(dark * u) / total, jnp.sum(dark * v) / total])


class Gate:
    """The arm's run at a point in the process's dtype: its objective, its per-step centroids, their derivatives in
    both modes and the central differences' ladders, as plain numbers."""

    def __init__(self, point: Point, arm: Arm) -> None:
        built = case(point).build()
        if arm.start:
            built.run(ms(arm.start))
        graph = built.graph
        state, params = full_state(graph), graph_params(graph)
        if arm.override:
            camera = dict(params["nodes"][CAMERA])
            camera["fy"] = jnp.asarray(camera["fy"]) + arm.override
            params = {**params, "nodes": {**params["nodes"], CAMERA: camera}}
        self.point, self.arm, self.state, self.params = point, arm, state, params
        self.node = graph.get_node(CAMERA)
        kept = ((CAMERA, "expected"), (BALL, "pose"))
        fed = None if arm.chunks is None else [(length, {}) for length in arm.chunks]
        self.reverse_run = differentiated_run(graph, STEPS, (), kept, checkpoint=True, fed=fed)
        self.forward_run = differentiated_run(graph, STEPS, (), kept, checkpoint=False, fed=fed)
        self.dtype = state[CAMERA]["expected"].dtype
        self.theta0 = jnp.asarray([state[BALL]["twist"][0], state[BALL]["pose"][2],
                                   *[params["nodes"][CAMERA][name] for name in NAMES[2:]]], dtype=self.dtype)
        self.frame_step_of = [((arm.start + t) // point.period) * point.period - arm.start for t in range(STEPS)]

    def _inserted(self, theta: Any) -> tuple[Any, Any]:
        ball = dict(self.state[BALL])
        ball["twist"] = ball["twist"].at[0].set(theta[0])
        ball["pose"] = ball["pose"].at[2].set(theta[1])
        camera = dict(self.params["nodes"][CAMERA])
        for index, name in enumerate(NAMES[2:], start=2):
            camera[name] = theta[index].astype(jnp.asarray(camera[name]).dtype)
        return {**self.state, BALL: ball}, {**self.params, "nodes": {**self.params["nodes"], CAMERA: camera}}

    def outputs(self, theta: Any, run: Any) -> tuple[Any, Any]:
        """Every step's frame and the ball's projected centre at the run's leaves."""
        state, params = self._inserted(theta)
        _, _, kept = run(state, params)
        own = params["nodes"][CAMERA]
        camera = detector_of(self.node._spec(), [jnp.asarray(own[name], dtype=self.dtype)  # noqa: SLF001
                                                  for name in INTRINSICS])
        placement = np.asarray(self.node.constants()["placement"], dtype=np.float64)
        centres = project(camera, CameraPose(placement[:3], placement[3:]), kept[BALL]["pose"][:, :3])[0]
        return kept[CAMERA]["expected"], centres

    def objective(self, theta: Any, run: Any) -> tuple[Any, Any]:
        frames, centres = self.outputs(theta, run)
        c = jax.vmap(soft_centroid)(frames)
        return jnp.sum((c - centres) ** 2), c

    def measured(self) -> dict[str, Any]:
        theta0 = self.theta0
        report: dict[str, Any] = {"dtype": str(self.dtype), "point": self.point.__dict__, "names": list(NAMES),
                                  "frame_step_of": self.frame_step_of}
        value = jax.jit(lambda theta: self.objective(theta, self.forward_run))
        j0, c0 = value(theta0)
        report["J"], report["c"] = float(j0), np.asarray(c0, dtype=np.float64).tolist()
        report["g_vjp"] = np.asarray(jax.jit(jax.grad(lambda theta: self.objective(theta, self.reverse_run)[0]))(
            theta0), dtype=np.float64).tolist()
        # Each step's centroid's derivative by reverse mode, one cotangent a call (one compiled program; a jacrev would
        # batch the backward pass over the 24 cotangents).
        pulled = jax.jit(lambda theta, cotangent: jax.vjp(lambda at: self.objective(at, self.reverse_run)[1], theta)[1](
            cotangent)[0])
        rows = [[np.asarray(pulled(theta0, jnp.zeros((STEPS, 2), dtype=self.dtype).at[t, c].set(1.0)),
                            dtype=np.float64) for c in (0, 1)] for t in range(STEPS)]
        report["dc_rev"] = np.moveaxis(np.asarray(rows), -1, 0).tolist()

        def traced(theta: Any) -> Any:
            frames, centres = self.outputs(theta, self.forward_run)
            return frames, centres

        tangent = jax.jit(lambda theta, e: jax.jvp(traced, (theta,), (e,)))

        def terms(frames: Any, centres: Any) -> tuple[Any, Any]:
            return jax.grad(lambda f, p: jnp.sum((jax.vmap(soft_centroid)(f) - p) ** 2), argnums=(0, 1))(frames,
                                                                                                    centres)

        per_frame = jax.jit(jax.vmap(jax.jacrev(soft_centroid)))
        bar = jax.jit(terms)
        g_jvp, scale, dc_fwd, step_scale = [], [], [], []
        for index in range(len(NAMES)):
            e = jnp.zeros_like(theta0).at[index].set(1.0)
            (frames, centres), (dframes, dcentres) = tangent(theta0, e)
            bar_f, bar_p = bar(frames, centres)
            g_jvp.append(float(jnp.sum(bar_f * dframes) + jnp.sum(bar_p * dcentres)))
            scale.append(float(jnp.sum(jnp.abs(bar_f * dframes)) + jnp.sum(jnp.abs(bar_p * dcentres))))
            jac = per_frame(frames)  # (steps, 2, rows, columns)
            dc_fwd.append(np.asarray(jnp.sum(jac * dframes[:, None], axis=(2, 3)), dtype=np.float64).tolist())
            step_scale.append(np.asarray(jnp.sum(jnp.abs(jac * dframes[:, None]), axis=(2, 3)),
                                         dtype=np.float64).tolist())
        report["g_jvp"], report["scale"], report["dc_fwd"], report["step_scale"] = g_jvp, scale, dc_fwd, step_scale
        ladder: dict[str, list[dict[str, Any]]] = {}
        for index, name in enumerate(NAMES):
            rows = []
            for fraction in LADDER:
                e = jnp.zeros_like(theta0).at[index].set(1.0)
                up, down = theta0 + fraction * PIXEL_SCALE[name] * e, theta0 - fraction * PIXEL_SCALE[name] * e
                h = float(up[index] - theta0[index])
                (j_up, c_up), (j_down, c_down) = value(up), value(down)
                rows.append({"h": h, "J_up": float(j_up), "J_down": float(j_down),
                             "D": (float(j_up) - float(j_down)) / (2.0 * h),
                             "dc": ((np.asarray(c_up, dtype=np.float64) - np.asarray(c_down, dtype=np.float64))
                                    / (2.0 * h)).tolist()})
            ladder[name] = rows
        report["ladder"] = ladder
        return report


def measured(point: Point, arm: Arm, dtype: str) -> dict[str, Any]:
    """The arm's measurements in ``dtype`` (the run's precision set for it and restored after)."""
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", dtype == "float64")
    try:
        return Gate(point, arm).measured()
    finally:
        jax.config.update("jax_enable_x64", held)


def _rounding_of_j(dtype: str, f64: dict[str, Any], f32: dict[str, Any]) -> float:
    gap = abs(f32["J"] - f64["J"])
    for name in NAMES:
        for a, b in zip(f32["ladder"][name], f64["ladder"][name], strict=True):
            gap = max(gap, abs(a["J_up"] - b["J_up"]), abs(a["J_down"] - b["J_down"]))
    return gap if dtype == "float32" else gap * EPS["float64"] / EPS["float32"]


def _value(row: dict[str, Any], key: str, index: tuple[int, ...]) -> float:
    held = row[key]
    for i in index:
        held = held[i]
    return float(held)


def cd_bound(rows: list[dict[str, Any]], k: int, rounding: float, key: str = "D", index: tuple[int, ...] = ()) -> float:
    """The bound on ``|D(h_k) - D|`` from the ladder's neighbours (the finer one must exist)."""
    finer = abs(_value(rows[k], key, index) - _value(rows[k + 1], key, index)) / (1.0 - 1.0 / RATIO)
    coarser = abs(_value(rows[k - 1], key, index) - _value(rows[k], key, index)) / (RATIO ** 2 - 1.0) if k else 0.0
    return finer + coarser + 2.0 * rounding / abs(rows[k]["h"]) + 2.0 * rounding / abs(rows[k + 1]["h"]) / RATIO


def best_step(rows: list[dict[str, Any]], rounding: float, key: str = "D", index: tuple[int, ...] = ()
              ) -> tuple[int, float]:
    bound, k = min((cd_bound(rows, k, rounding, key, index), k) for k in range(len(rows) - 1))
    return k, bound


def judged(run: dict[str, Any], f64: dict[str, Any], f32: dict[str, Any], width: int) -> dict[str, dict[str, Any]]:
    """Every check of one dtype's run, by name: each a ``pass`` with what it compared."""
    dtype = run["dtype"]
    eps = EPS[dtype]
    rounding = _rounding_of_j(dtype, f64, f32)
    checks: dict[str, dict[str, Any]] = {}
    for i, name in enumerate(NAMES):
        g, s = run["g_vjp"][i], run["scale"][i]
        line = CHAIN * eps * s
        checks[f"T1 {name}"] = {"pass": abs(g - run["g_jvp"][i]) <= line, "diff": g - run["g_jvp"][i], "bound": line}
        k, bound = best_step(run["ladder"][name], rounding)
        cd = run["ladder"][name][k]["D"]
        reference = abs(f64["g_vjp"][i])
        discriminates = dtype == "float64" or bound <= DISCRIMINATES * reference
        checks[f"T2 {name}"] = {"pass": (abs(g - cd) <= bound) if discriminates else True, "judged": discriminates,
                                "ad": g, "cd": cd, "bound": bound, "relative_bound": bound / reference if reference
                                else None}
        if dtype == "float32":
            tolerance = (CHAIN * eps + 1.5 * math.sqrt(width * eps)) * s
            gap = abs(g - f64["g_vjp"][i])
            checks[f"T3 {name}"] = {"pass": gap <= tolerance, "gap": gap, "bound": tolerance,
                                    "relative_bound": tolerance / reference if reference else None}
    rev, fwd, scale, frame_of = run["dc_rev"], run["dc_fwd"], run["step_scale"], run["frame_step_of"]
    held = [t for t in range(len(frame_of)) if frame_of[t] != t and frame_of[t] >= 0]
    for i, name in enumerate(NAMES):
        worst_held = max(((abs(rev[i][t][c] - rev[i][frame_of[t]][c]), CHAIN * eps * scale[i][frame_of[t]][c])
                          for t in held for c in (0, 1)), key=lambda pair: pair[0] - pair[1], default=(0.0, 0.0))
        checks[f"H1 {name} held equals its frame"] = {"pass": worst_held[0] <= worst_held[1], "worst": worst_held[0],
                                                      "bound": worst_held[1], "held_steps": len(held)}
        worst = max(((abs(rev[i][t][c] - fwd[i][t][c]), CHAIN * eps * scale[i][t][c]) for t in range(len(frame_of))
                     for c in (0, 1)), key=lambda pair: pair[0] - pair[1])
        checks[f"H2 {name}"] = {"pass": worst[0] <= worst[1], "worst": worst[0], "bound": worst[1]}
    if held:
        c_round = max(abs(a - b) for sa, sb in zip(f32["c"], f64["c"], strict=True) for a, b in zip(sa, sb, strict=True))
        if dtype == "float64":
            c_round *= EPS["float64"] / EPS["float32"]
        failed = []
        for t in held:
            k, bound = best_step(run["ladder"]["v0x"], c_round, "dc", (t, 0))
            ad = rev[0][t][0]
            if not (abs(ad - run["ladder"]["v0x"][k]["dc"][t][0]) <= bound and ad != 0.0):
                failed.append({"step": t, "ad": ad, "cd": run["ladder"]["v0x"][k]["dc"][t][0], "bound": bound})
        checks["H1 v0x reaches every held step"] = {"pass": not failed, "failed": failed}
    return checks
