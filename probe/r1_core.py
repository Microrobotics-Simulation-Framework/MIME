"""Camera spike R1: reverse mode through the in-loop camera over a run. The shared pieces.

* ``case``: the in-loop camera's ballistic scene (a steel ball moving along a tube, seen across it by a camera fixed in the
  world), at a width, a frame period, and the ball's start and speed;
* ``scalar_run``: the compiled graph's own step (MADDENING's ``_build_step_fn``, as MIME's ``differentiated_run`` takes it)
  scanned over a run with a per-step objective accumulated in the carry, chunked under ``jax.checkpoint`` as
  ``differentiated_run`` chunks it when ``checkpoint`` is set;
* the run-level scalar: a darkness-weighted (soft) centroid of the camera's ``expected`` against the ball's true projected
  centre, its squared distance summed over the trajectory;
* the planted controls, switched on by ``CAMERA_R1_PLANT`` before anything is traced (a node's step is traced once a
  process).

Every number here is the spike's own; nothing is a MIME default unless named so.
"""

from __future__ import annotations

import math
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "sensing"))

PLANT = os.environ.get("CAMERA_R1_PLANT", "")
#: The parameters the gradient is taken in: the ball's starting speed along the tube and its starting height (two that
#: move a silhouette), and three of the detector's leaves (one focal length, one centre, the first radial coefficient).
NAMES = ("v0x", "z0", "fy", "cy", "k1")
CAMERA = "camera:"
BALL = "ball"


def plant() -> None:
    """Switch on the planted error ``CAMERA_R1_PLANT`` names (before any trace)."""
    if PLANT == "edge-stop":
        from mime.sensing import _coverage

        original = _coverage.disc

        def stopped(*args: Any, **kwargs: Any) -> Any:
            return jax.lax.stop_gradient(original(*args, **kwargs))

        _coverage.disc = stopped
    elif PLANT == "cache-stop":
        from mime.sensing._node import CameraNode

        original_part = CameraNode._static_part

        def held_stopped(self: Any, intrinsics: Any, numbers: Any, pose: Any, dtype: Any, count: Any, held: Any) -> Any:
            return original_part(self, intrinsics, numbers, pose, dtype, count, jax.lax.stop_gradient(held))

        CameraNode._static_part = held_stopped
    elif PLANT not in ("", "held-stop"):
        raise SystemExit(f"unknown plant {PLANT!r}")


def case(width: int = 160, period_steps: int = 3, x0_um: float = -30.0, z0_um: float = 15.0, speed: float = 2e-3,
         sub_frames: int = 1) -> Any:
    """The ballistic scene at ``width`` x ``3 width / 4`` (the done case's camera scaled, fx = 750 px at 160 wide), the
    ball starting at ``(x0, 0, z0)`` um moving at ``speed`` m/s along the tube, a frame every ``period_steps`` base steps
    of 1 ms."""
    from in_loop_cases import DETECTOR, LOOK_ACROSS, RADIUS_UM

    from mime import Body, Experiment, State, Structure
    from mime.dynamics import RigidInertial
    from mime.observations import Camera, Detector, World
    from mime.scene import Scene, Sphere, Tube, X, Z
    from mime.things import Part
    from mime.units import kg_per_m3, m, m_per_s, ms, rad_per_s, um

    scale = width / DETECTOR.width
    height = width * 3 // 4
    detector = Detector(width=width, height=height, fx=DETECTOR.fx * scale, fy=DETECTOR.fy * scale,
                        cx=(width - 1) / 2, cy=(height - 1) / 2)
    exp = Experiment("ballistic", scene=Scene.new(up=Z, units="m"), dt=ms(1))
    exp.add_structure(Structure("tube", geometry=Tube(axis=+X, inner_radius=um(60), wall=um(10), length=um(480))))
    exp.add_body(Body(BALL, dynamics=RigidInertial(), initial=State(
        position=um((x0_um, 0.0, z0_um)), orientation=(1.0, 0.0, 0.0, 0.0),
        velocity=m_per_s((speed, 0.0, 0.0)), angular_velocity=rad_per_s((0.0, 0.0, 0.0))),
        parts={"steel": Part(Sphere(radius=um(RADIUS_UM)))}))
    exp.declare(f"{BALL}.parts.steel.density", kg_per_m3(7850.0), reason="steel")
    exp.observe(Camera(detector=detector, mount=World(position=m((0.0, -1e-3, 0.0)), orientation=LOOK_ACROSS),
                       frame_period=ms(period_steps), sub_frames=sub_frames, absorbers=(("tube", 3000.0),)))
    return exp


def cost_case(width: int = 320, period_steps: int = 1, steps: int = 100, render_checkpoint: int = 0,
              subsample_checkpoint: int = 0, cache_checkpoint: int = 0) -> Any:
    """The cost arms' scene: the ball starts 40 um left of the axis's foot and crosses about 60 um over the run (it stays
    in view at every run length), a frame every ``period_steps`` base steps; with ``render_checkpoint`` the render inside
    the frame's cond under ``jax.checkpoint`` (set here, before anything is traced, so that the memory estimate and the
    measurement both see it); ``subsample_checkpoint`` and ``cache_checkpoint`` the spike's other placements: each corner
    sub-sample of the box edge, and the static cache's remake, under ``jax.checkpoint``."""
    for name, on in (("MIME_SPIKE_RENDER_CHECKPOINT", render_checkpoint), ("MIME_SPIKE_SUBSAMPLE_CHECKPOINT",
                                                                            subsample_checkpoint),
                     ("MIME_SPIKE_CACHE_CHECKPOINT", cache_checkpoint)):
        if on:
            os.environ[name] = "1"
        else:
            os.environ.pop(name, None)
    return case(width=width, period_steps=period_steps, x0_um=-40.0, z0_um=15.0, speed=min(2e-3, 60e-6 / (steps * 1e-3)))


def scalar_run(graph: Any, steps: int, per_step: Callable[[Any, Any], tuple[Any, Any]], *, checkpoint: bool
               ) -> Callable[[Any, Any], tuple[Any, Any, Any]]:
    """``(state, params) -> (sum of per_step's first output over the steps, per_step's second output at every step, the
    final state)``: MIME's ``differentiated_run`` with the objective accumulated in the carry (its chunks, run-constants
    and differentiated-node context as it has them)."""
    from mime.build import _maddening as mad
    from mime.core import differentiated_nodes, run_constants_made

    held = mad._held(graph)  # noqa: SLF001 - the spike drives the graph's own step as differentiated_run does
    held._recover_from_escaped_tracers()  # noqa: SLF001
    held._check_static_data_dirty()  # noqa: SLF001
    if held._dirty or held._compiled_step is None:  # noqa: SLF001
        held.compile()
    held._refuse_xla_loop_hazards("run_scan", scan=True)  # noqa: SLF001
    names = mad.run_constant_nodes(graph)
    every = tuple(graph.node_names)
    step_fn = held._build_step_fn()  # noqa: SLF001
    fed = held._resolve_external_inputs({})  # noqa: SLF001 - the ballistic graph's only external inputs are the
    # world-mounted camera's mount poses and rates, which it never reads (mount_index < 0)
    lax = jax.lax
    jnp = jax.numpy

    def chunk(length: int) -> Callable[[Any, Any], tuple[Any, Any]]:
        def scanned(carry: Any, made: Any) -> tuple[Any, Any]:
            def body(inner: Any, _: Any) -> tuple[Any, Any]:
                state, total = inner
                state = step_fn(state, fed, made)
                value, kept = per_step(state, made)
                return (state, total + value), kept

            return lax.scan(body, carry, None, length=length)

        return jax.checkpoint(scanned) if checkpoint else scanned

    def run(state: Any, params: Any) -> tuple[Any, Any, Any]:
        made = mad.with_run_constants(graph, params)
        real = state[CAMERA]["expected"].dtype
        carry = (state, jnp.zeros((), dtype=real))
        parts: list[Any] = []
        with run_constants_made(names), differentiated_nodes(every):
            length = max(1, math.ceil(math.sqrt(steps))) if checkpoint else max(1, steps)
            chunks, rest = divmod(steps, length)
            if chunks:
                step = chunk(length)

                def outer(inner: Any, _: Any) -> tuple[Any, Any]:
                    return step(inner, made)

                carry, out = lax.scan(outer, carry, None, length=chunks)
                parts.append(jax.tree_util.tree_map(lambda a: jnp.reshape(a, (chunks * length, *a.shape[2:])), out))
            if rest:
                carry, out = chunk(rest)(carry, made)
                parts.append(out)
        kept = jax.tree_util.tree_map(lambda *each: jnp.concatenate(each), *parts) if len(parts) > 1 else parts[0]
        return carry[1], kept, carry[0]

    return run


def backlight(spec: dict[str, Any]) -> float:
    response = spec["response"]
    return float(response["full_scale"]) * float(response["gain"]) + float(response["offset"])


def soft_centroid(expected: Any, light: float) -> Any:
    """The darkness-weighted centroid ``(u, v)`` of a frame: each pixel weighted by its share of the backlight it does not
    pass (the spike's stand-in for the tracker's soft centroid; the tube's faint walls weigh in with the ball)."""
    jnp = jax.numpy
    dark = (light - expected) / light
    rows, columns = expected.shape
    v = jnp.arange(rows, dtype=expected.dtype)[:, None]
    u = jnp.arange(columns, dtype=expected.dtype)[None, :]
    total = jnp.sum(dark)
    return jnp.stack([jnp.sum(dark * u) / total, jnp.sum(dark * v) / total])


def projected(graph: Any, params: Any, position: Any) -> Any:
    """The ball's centre projected by the camera at the run's own leaves (its pinhole and distortion)."""
    from mime.sensing._detector import CameraPose, project
    from mime.sensing._node import INTRINSICS, detector_of

    node = graph.get_node(CAMERA)
    own = params["nodes"][CAMERA]
    dtype = position.dtype
    camera = detector_of(node._spec(), [jax.numpy.asarray(own[name], dtype=dtype) for name in INTRINSICS])  # noqa: SLF001
    placement = np.asarray(node.constants()["placement"], dtype=np.float64)
    return project(camera, CameraPose(placement[:3], placement[3:]), position)[0]


def per_step_objective(graph: Any, keep_frames: bool = False) -> Callable[[Any, Any], tuple[Any, Any]]:
    """The per-step term ``|c - p|^2`` (pixels squared) of the soft centroid ``c`` of the camera's ``expected`` after the
    step against the ball's projected centre ``p`` after the step, and what each step keeps: ``c``, ``p`` and, with
    ``keep_frames``, the frame itself."""
    light = backlight(graph.get_node(CAMERA)._spec())  # noqa: SLF001

    def term(state: Any, params: Any) -> tuple[Any, Any]:
        expected = state[CAMERA]["expected"]
        c = soft_centroid(expected, light)
        p = projected(graph, params, state[BALL]["pose"][:3])
        kept = {"c": c, "p": p}
        if keep_frames:
            kept["frame"] = expected
        return jax.numpy.sum((c - p) ** 2), kept

    return term


def inserted(theta: Any, state: Any, params: Any) -> tuple[Any, Any]:
    """``state`` and ``params`` with the five parameters of ``NAMES`` taken from ``theta``."""
    ball = dict(state[BALL])
    ball["twist"] = ball["twist"].at[0].set(theta[0])
    ball["pose"] = ball["pose"].at[2].set(theta[1])
    nodes = {node: dict(leaves) for node, leaves in params["nodes"].items()}
    for index, name in enumerate(NAMES[2:], start=2):
        nodes[CAMERA][name] = theta[index].astype(nodes[CAMERA][name].dtype)
    return {**state, BALL: ball}, {**params, "nodes": nodes}


def start_of(built: Any) -> tuple[Any, Any, Any]:
    """The graph's state at its start and its parameters, and ``theta`` read from them."""
    from mime.build import _maddening as mad

    graph = built.graph
    graph.reset_state()
    state, params = mad.full_state(graph), mad.graph_params(graph)
    jnp = jax.numpy
    real = state[CAMERA]["expected"].dtype
    theta = jnp.asarray([state[BALL]["twist"][0], state[BALL]["pose"][2],
                         *[params["nodes"][CAMERA][name] for name in NAMES[2:]]], dtype=real)
    return state, params, theta


#: One pixel of image motion in each parameter at 160 x 120 (scaled with the width): the steps of the central differences
#: are fractions of it. The speed's is a pixel at the end of a 12-step run; the focal length's at the ball's distance from
#: the centre row and the tube's walls (11 to 50 px); k1's at the tube's walls (a pixel at k1 of about 0.7).
PIXEL_SCALE = {"v0x": 1.333e-6 / 12e-3, "z0": 1.333e-6, "fy": 30.0, "cy": 1.0, "k1": 1.0}


def raw_grad(width: int = 160, steps: int = 12, kind: str = "jacrev", render_checkpoint: int = 0, period_steps: int = 3
             ) -> tuple[Any, tuple[Any, ...]]:
    """``mem_estimate.py --raw``'s factory: the gradient scripts' largest program at a size, ``(function, (theta,))``:
    the per-step centroids' Jacobian by reverse mode (``jacrev``), the run-level scalar's gradient (``vjp``) or its value
    (``forward``), through the chunked, checkpointed run as ``r1_grad.py`` compiles it."""
    if render_checkpoint:
        os.environ["MIME_SPIKE_RENDER_CHECKPOINT"] = "1"
    plant()
    built = case(width=width, period_steps=period_steps).build()
    graph = built.graph
    state0, params0, theta0 = start_of(built)
    run = scalar_run(graph, steps, per_step_objective(graph), checkpoint=True)

    def objective(theta: Any) -> Any:
        return run(*inserted(theta, state0, params0))

    if kind == "jacrev":
        function = jax.jit(jax.jacrev(lambda theta: objective(theta)[1]["c"]))
    elif kind == "vjp":
        function = jax.jit(jax.grad(lambda theta: objective(theta)[0]))
    else:
        function = jax.jit(lambda theta: objective(theta)[0])
    return function, (theta0,)
